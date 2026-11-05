import argparse
import datetime
import json
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, get_state_dict

import utils
from datasets import build_dataset
from engine import train_one_epoch, validate_one_epoch
from models.hybridnet import HybridNet

IN_CHANS = 4


def get_args_parser():
    p = argparse.ArgumentParser("RNAtomy pretraining", add_help=False)
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--epochs", default=250, type=int)
    p.add_argument("--accum_iter", default=2, type=int, help="gradient accumulation steps")
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--device", default="cuda")

    # optimization
    p.add_argument("--opt", default="adamw", type=str)
    p.add_argument("--opt-eps", default=1e-8, type=float)
    p.add_argument("--opt-betas", default=None, type=float, nargs="+")
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--weight-decay", default=0.05, type=float)
    p.add_argument("--clip-grad", default=1.0, type=float)
    p.add_argument("--sched", default="cosine", type=str)
    p.add_argument("--lr", default=2e-4, type=float, help="peak LR at a 256-sample batch")
    p.add_argument("--lr-noise", default=None, type=float, nargs="+")
    p.add_argument("--lr-noise-pct", default=0.67, type=float)
    p.add_argument("--lr-noise-std", default=1.0, type=float)
    p.add_argument("--warmup-lr", default=1e-6, type=float)
    p.add_argument("--min-lr", default=1e-5, type=float)
    p.add_argument("--decay-epochs", default=30, type=float)
    p.add_argument("--warmup-epochs", default=20, type=int)
    p.add_argument("--cooldown-epochs", default=10, type=int)
    p.add_argument("--patience-epochs", default=10, type=int)
    p.add_argument("--decay-rate", "--dr", default=0.1, type=float)
    p.add_argument("--model-ema", action="store_true", default=True)
    p.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    p.add_argument("--model-ema-decay", default=0.99996, type=float)

    # data
    p.add_argument("--data-path", required=True, help="folder with train/ and val/ holding <id>.npy maps")
    p.add_argument("--feat1d-root", default="", help="folder with <id>/<feat1d-file> residue descriptors")
    p.add_argument("--feat1d-file", default="feat1d_norm.npy")
    p.add_argument("--feat1d-dim", default=23, type=int)
    p.add_argument("--use-1d-feats", action="store_true", default=True)
    p.add_argument("--no-use-1d-feats", action="store_false", dest="use_1d_feats")
    p.add_argument("--patch-size", default=8, type=int)
    p.add_argument("--token-budget", default=0, type=int,
                   help="max patch tokens per batch of mixed-size maps; 0 = batch size x median token count")
    p.add_argument("--length-curriculum", action="store_true", default=True)
    p.add_argument("--no-length-curriculum", action="store_false", dest="length_curriculum")
    p.add_argument("--curriculum-start-percentile", default=40.0, type=float)
    p.add_argument("--curriculum-end-percentile", default=100.0, type=float)
    p.add_argument("--curriculum-epochs", default=30, type=int)
    p.add_argument("--latb", action="store_true", default=True, help="length-adaptive token budget")
    p.add_argument("--no-latb", action="store_false", dest="latb")
    p.add_argument("--latb-alpha", default=1.2, type=float)
    p.add_argument("--latb-beta", default=1.0, type=float)
    p.add_argument("--latb-max-mult", default=2.0, type=float)
    p.add_argument("--pin-mem", action="store_true", default=True)
    p.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")

    # masking and losses
    p.add_argument("--mask-ratio-min", default=0.45, type=float)
    p.add_argument("--mask-ratio-max", default=0.65, type=float)
    p.add_argument("--contact-alpha", default=1.0, type=float, help="signal-aware patch weighting; 0 = uniform")
    p.add_argument("--recon-1d-weight", default=0.5, type=float, help="lambda_1D; 0 disables the descriptor loss")
    p.add_argument("--recon-1d-mask-ratio", default=0.15, type=float)
    p.add_argument("--recon-1d-span", default=4, type=int)
    p.add_argument("--moddrop-prob", default=0.3, type=float, help="probability of dropping the whole 1D track")
    p.add_argument("--moddrop-recon-weight", default=0.5, type=float)

    # model
    p.add_argument("--embed-dim", default=768, type=int)
    p.add_argument("--depth", default=16, type=int)
    p.add_argument("--mamba-ratio", default=3, type=int, help="Mamba blocks per attention block")
    p.add_argument("--window-size", default=16, type=int, help="attention window, in patches")
    p.add_argument("--num-heads", default=12, type=int)
    p.add_argument("--feat1d-cond-dim", default=256, type=int)
    p.add_argument("--feat1d-encoder-layers", default=2, type=int)

    # bookkeeping
    p.add_argument("--output_dir", default="")
    p.add_argument("--save-freq", default=10, type=int)
    p.add_argument("--val-every", default=1, type=int)
    p.add_argument("--no-val", action="store_true")
    p.add_argument("--early-stop-patience", default=10, type=int, help="0 disables")
    p.add_argument("--resume", default="")
    p.add_argument("--start_epoch", default=0, type=int)

    # distributed
    p.add_argument("--distributed", action="store_true", default=False)
    p.add_argument("--local-rank", default=0, type=int)
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--dist_url", default="env://")
    return p


def scan_split(data_path, split, patch_size, use_1d_feats, feat1d_root, feat1d_file, feat1d_dim,
               compute_stats=False, allow_empty=False):
    """Index the maps of one split and drop samples whose 1D features are missing or mis-shaped.

    Returns (ids, metadata, mean, std); mean/std are per-channel statistics (training split only).
    """
    files = sorted((Path(data_path) / split).glob("*.npy"))
    if not files and not allow_empty:
        raise RuntimeError(f"No .npy files found in {Path(data_path) / split}")

    ids, metadata = [], []
    skipped = []
    ch_sum = ch_sq = None
    n_pixels = 0
    for path in files:
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(f"Expected (H, W, C) array in {path}, got {arr.shape}")
        h, w, channels = arr.shape
        sid = path.stem

        if use_1d_feats:
            fpath = Path(feat1d_root) / sid / feat1d_file
            if not fpath.exists():
                skipped.append((sid, "missing 1D features"))
                continue
            feat = np.load(fpath)
            if feat.ndim != 2 or feat.shape[0] != h or feat.shape[1] != feat1d_dim:
                skipped.append((sid, f"1D feature shape {feat.shape}"))
                continue

        if compute_stats:
            flat = arr.astype(np.float64, copy=False).reshape(-1, channels)
            if ch_sum is None:
                ch_sum, ch_sq = np.zeros(channels), np.zeros(channels)
            ch_sum += flat.sum(axis=0)
            ch_sq += np.square(flat).sum(axis=0)
            n_pixels += flat.shape[0]

        aspect = w / float(max(h, 1))
        metadata.append({
            "index": len(metadata), "h": h, "w": w,
            "tokens": math.ceil(h / patch_size) * math.ceil(w / patch_size),
            "aspect_bin": 0 if aspect < 0.8 else (2 if aspect > 1.25 else 1),
        })
        ids.append(sid)

    if skipped:
        print(f"[{split}] skipped {len(skipped)} samples, e.g. {skipped[:5]}")
    if not metadata and not allow_empty:
        raise RuntimeError(f"No usable samples in split '{split}'")

    mean = std = None
    if compute_stats:
        m = ch_sum / n_pixels
        mean = m.tolist()
        std = np.sqrt(np.maximum(ch_sq / n_pixels - np.square(m), 1e-12)).tolist()
    return ids, metadata, mean, std


def run_on_rank0(args, fn, *fn_args, **fn_kwargs):
    """Run `fn` on rank 0 only and broadcast its result to the other ranks."""
    if not args.distributed:
        return fn(*fn_args, **fn_kwargs)
    payload = [None]
    if getattr(args, "rank", 0) == 0:
        try:
            payload[0] = {"ok": True, "value": fn(*fn_args, **fn_kwargs)}
        except Exception as exc:
            payload[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    torch.distributed.broadcast_object_list(payload, src=0)
    if not payload[0]["ok"]:
        raise RuntimeError(f"rank 0 failed: {payload[0]['error']}")
    return payload[0]["value"]


class TokenBucketBatchSampler(torch.utils.data.Sampler):
    """Batches maps of identical shape; leftovers are packed by token budget.

    `curriculum` caps the map size by epoch; `latb` raises the token budget for larger maps.
    """

    def __init__(self, metadata, batch_size, token_budget, drop_last=False, curriculum=None, latb=None,
                 distributed=False, num_replicas=1, rank=0, seed=0):
        self.metadata = metadata
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.curriculum = curriculum
        self.latb = latb
        self.distributed = distributed
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        median_tokens = int(np.median([m["tokens"] for m in metadata]))
        self.median_tokens = max(1, median_tokens)
        self.token_budget = token_budget if token_budget > 0 else max(batch_size, batch_size * median_tokens)
        self.max_tokens = self._curriculum_max_tokens(0) if curriculum is not None else None
        self._len = len(self._rank_batches(rng=None))

    def _curriculum_max_tokens(self, epoch):
        start, end = self.curriculum["start_tokens"], self.curriculum["end_tokens"]
        epochs = max(1, int(self.curriculum["epochs"]))
        if epochs <= 1:
            return int(round(end))
        t = min(1.0, epoch / float(epochs - 1))
        return int(round(start + t * (end - start)))

    def _budget_for(self, token_count):
        if self.latb is None or not self.latb["enabled"]:
            return self.token_budget
        max_budget = int(round(self.token_budget * self.latb["max_mult"]))
        scaled = self.token_budget * self.latb["alpha"] * (token_count / float(self.median_tokens)) ** self.latb["beta"]
        return max(1, int(round(max(self.token_budget, min(max_budget, scaled)))))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.curriculum is not None:
            self.max_tokens = self._curriculum_max_tokens(epoch)
            self._len = len(self._rank_batches(rng=None))

    def _full_batches_and_leftovers(self, rng):
        metadata = self.metadata
        if self.max_tokens is not None:
            metadata = [m for m in metadata if m["tokens"] <= self.max_tokens]
        by_shape = {}
        for m in metadata:
            by_shape.setdefault((m["h"], m["w"]), []).append(m["index"])

        full, leftovers = [], []
        for key in sorted(by_shape):
            group = by_shape[key]
            if rng is not None:
                rng.shuffle(group)
            n_full = len(group) // self.batch_size
            for i in range(n_full):
                full.append(group[i * self.batch_size:(i + 1) * self.batch_size])
            leftovers.extend(group[n_full * self.batch_size:])
        return full, leftovers

    def _pack_leftovers(self, leftovers, rng):
        if rng is not None:
            rng.shuffle(leftovers)
        leftovers.sort(key=lambda i: (self.metadata[i]["aspect_bin"], self.metadata[i]["tokens"]))

        packed, current = [], []
        current_tokens = current_max = 0
        for idx in leftovers:
            tokens = self.metadata[idx]["tokens"]
            current_max = tokens if not current else max(current_max, tokens)
            budget = self._budget_for(current_max)
            if current and (len(current) >= self.batch_size or current_tokens + tokens > budget):
                if len(current) == self.batch_size or not self.drop_last:
                    packed.append(current)
                current, current_tokens, current_max = [], 0, 0
            current.append(idx)
            current_tokens += tokens
            current_max = max(current_max, tokens)
        if current and (len(current) == self.batch_size or not self.drop_last):
            packed.append(current)
        return packed

    def _rank_batches(self, rng):
        full, leftovers = self._full_batches_and_leftovers(rng)
        batches = full + self._pack_leftovers(leftovers, rng)
        if rng is not None:
            rng.shuffle(batches)
        if not self.distributed:
            return batches
        usable = (len(batches) // self.num_replicas) * self.num_replicas
        return batches[:usable][self.rank::self.num_replicas]

    def __iter__(self):
        yield from self._rank_batches(np.random.default_rng(self.seed + self.epoch))

    def __len__(self):
        return self._len


def collate_batch(batch, patch_size):
    """Pad maps (and descriptors) to a common size; return the validity masks alongside."""
    if len(batch[0]) == 3:
        samples, targets, nt_feats = zip(*batch)
    else:
        samples, targets = zip(*batch)
        nt_feats = None

    align = lambda v: int(math.ceil(v / float(patch_size)) * patch_size)
    pad_h = align(max(s.shape[1] for s in samples))
    pad_w = align(max(s.shape[2] for s in samples))
    maps = samples[0].new_zeros((len(samples), samples[0].shape[0], pad_h, pad_w))
    valid_mask = torch.zeros((len(samples), pad_h, pad_w), dtype=torch.bool)
    for i, s in enumerate(samples):
        maps[i, :, :s.shape[1], :s.shape[2]] = s
        valid_mask[i, :s.shape[1], :s.shape[2]] = True

    if nt_feats is None:
        return maps, torch.stack(targets), valid_mask

    max_len = max(f.shape[0] for f in nt_feats)
    feats = nt_feats[0].new_zeros((len(nt_feats), max_len, nt_feats[0].shape[1]))
    nt_valid_mask = torch.zeros((len(nt_feats), max_len), dtype=torch.bool)
    for i, f in enumerate(nt_feats):
        feats[i, :f.shape[0]] = f
        nt_valid_mask[i, :f.shape[0]] = True
    return maps, torch.stack(targets), valid_mask, feats, nt_valid_mask


def save_training_curves(output_dir, epochs, train_loss, train_lr, val_loss):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, train_loss, color="tab:red", marker="o", markersize=3, label="train loss")
    if any(math.isfinite(v) for v in val_loss):
        ax1.plot(epochs, val_loss, color="tab:green", marker="s", markersize=3, label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_lr, color="tab:blue", label="lr")
    ax2.set_ylabel("learning rate")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def make_loader(dataset, metadata, args, sampler_kwargs):
    sampler = TokenBucketBatchSampler(
        metadata, args.batch_size, args.token_budget, distributed=args.distributed,
        num_replicas=utils.get_world_size(), rank=utils.get_rank(), **sampler_kwargs)
    if len(sampler) == 0:
        raise RuntimeError("Sampler produced zero batches; use a smaller --batch-size or fewer processes")
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=args.pin_mem,
        collate_fn=partial(collate_batch, patch_size=args.patch_size))
    return loader


def main(args):
    utils.init_distributed_mode(args)
    if not hasattr(args, "gpu"):
        args.gpu = 0
    for name in ("mask_ratio_min", "mask_ratio_max"):
        if not 0.0 < getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.mask_ratio_min > args.mask_ratio_max:
        raise ValueError("--mask-ratio-min must be <= --mask-ratio-max")
    if args.use_1d_feats and not args.feat1d_root:
        raise ValueError("--feat1d-root is required unless --no-use-1d-feats is set")

    args.in_chans = IN_CHANS
    main_process = args.local_rank == 0 and args.gpu == 0
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    scan_args = (args.patch_size, args.use_1d_feats, args.feat1d_root, args.feat1d_file, args.feat1d_dim)
    train_ids, train_meta, args.norm_mean, args.norm_std = run_on_rank0(
        args, scan_split, args.data_path, "train", *scan_args, compute_stats=True)
    if main_process:
        print(f"Train samples: {len(train_ids)}\nmean: {args.norm_mean}\nstd: {args.norm_std}")

    dataset_train = build_dataset(True, args, allowed_ids=train_ids)
    world_size = utils.get_world_size()

    curriculum = None
    if args.length_curriculum:
        tokens = np.array([m["tokens"] for m in train_meta])
        start_pct = float(np.clip(args.curriculum_start_percentile, 0.0, 100.0))
        end_pct = max(start_pct, float(np.clip(args.curriculum_end_percentile, 0.0, 100.0)))
        curriculum = {
            "start_tokens": int(round(np.percentile(tokens, start_pct))),
            "end_tokens": int(round(np.percentile(tokens, end_pct))),
            "epochs": max(1, args.curriculum_epochs),
        }
    latb = {"enabled": args.latb, "alpha": args.latb_alpha, "beta": args.latb_beta, "max_mult": args.latb_max_mult}
    data_loader_train = make_loader(
        dataset_train, train_meta, args, dict(curriculum=curriculum, latb=latb, seed=args.seed))

    data_loader_val, val_meta = None, []
    if not args.no_val:
        val_ids, val_meta, _, _ = run_on_rank0(
            args, scan_split, args.data_path, "val", *scan_args, allow_empty=True)
        if val_ids:
            data_loader_val = make_loader(
                build_dataset(False, args, allowed_ids=val_ids), val_meta, args, dict(seed=args.seed + 1000))
        else:
            print("No usable validation samples; validation disabled.")

    model = HybridNet(
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        embed_dim=args.embed_dim,
        depth=args.depth,
        mamba_ratio=args.mamba_ratio,
        window_size=args.window_size,
        window_num_heads=args.num_heads,
        use_1d_conditioning=args.use_1d_feats,
        nt_feat_dim=args.feat1d_dim,
        feat1d_cond_dim=args.feat1d_cond_dim,
        feat1d_encoder_layers=args.feat1d_encoder_layers,
        recon_1d_weight=args.recon_1d_weight,
        recon_1d_mask_ratio=args.recon_1d_mask_ratio,
        recon_1d_span=args.recon_1d_span,
        moddrop_prob=args.moddrop_prob,
        moddrop_recon_weight=args.moddrop_recon_weight,
        max_seq_len=max(max(m["h"], m["w"]) for m in train_meta + val_meta) + args.patch_size,
        contact_alpha=args.contact_alpha,
    ).to(device)

    model_ema = ModelEma(model, decay=args.model_ema_decay, device="", resume="") if args.model_ema else None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # --lr is the peak LR for a 256-sample batch; scale it up for larger effective batches.
    eff_batch_size = args.batch_size * args.accum_iter * world_size
    args.lr = args.lr * max(1.0, eff_batch_size / 256.0)
    args.min_lr = min(args.min_lr, args.lr * 0.01)
    args.warmup_lr = min(args.warmup_lr, args.lr * 0.001)

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if all(k in checkpoint for k in ("optimizer", "lr_scheduler", "epoch")):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
        lr_scheduler.step(args.start_epoch)

    def checkpoint_state(epoch, **extra):
        return {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "model_ema": get_state_dict(model_ema) if args.model_ema else None,
            "args": args,
            **extra,
        }

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    history = {"epochs": [], "train_loss": [], "val_loss": [], "train_lr": []}
    best_val_loss, epochs_since_improve = None, 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.mask_ratio_min == args.mask_ratio_max:
            args.mask_ratio = args.mask_ratio_min
        else:
            rng = np.random.default_rng(args.seed + epoch)
            args.mask_ratio = float(rng.uniform(args.mask_ratio_min, args.mask_ratio_max))
        epoch_mask_ratio = args.mask_ratio
        if main_process:
            print(f"Epoch {epoch} | mask_ratio={epoch_mask_ratio:.4f} | batches per rank={len(data_loader_train)}")
        data_loader_train.batch_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch,
                                      args.clip_grad, model_ema, args)

        val_stats = None
        run_val = data_loader_val is not None and ((epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs)
        if run_val:
            data_loader_val.batch_sampler.set_epoch(0)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(20240601)
            args.mask_ratio = 0.5
            val_model = model_ema.ema if model_ema is not None else model
            val_stats = validate_one_epoch(val_model, data_loader_val, device, epoch, args)
            args.mask_ratio = epoch_mask_ratio
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        lr_scheduler.step(epoch)
        if output_dir:
            state = checkpoint_state(epoch)
            utils.save_on_master(state, output_dir / "checkpoint.pth")
            if (epoch + 1) % args.save_freq == 0:
                utils.save_on_master(state, output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pth")

        history["epochs"].append(epoch)
        history["train_loss"].append(float(train_stats.get("loss", 0.0)))
        history["val_loss"].append(float(val_stats["loss"]) if val_stats is not None else float("nan"))
        history["train_lr"].append(float(train_stats.get("lr", 0.0)))

        if val_stats is not None and math.isfinite(history["val_loss"][-1]):
            current = history["val_loss"][-1]
            if best_val_loss is None or current < best_val_loss:
                best_val_loss, epochs_since_improve = current, 0
                if output_dir:
                    utils.save_on_master(checkpoint_state(epoch, best_val_loss=best_val_loss),
                                         output_dir / "checkpoint_best.pth")
            else:
                epochs_since_improve += 1
                if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
                    print(f"Early stop: no validation improvement for {epochs_since_improve} epochs "
                          f"(best {best_val_loss:.4f}).")
                    break

        if output_dir and utils.is_main_process():
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **({f"val_{k}": v for k, v in val_stats.items()} if val_stats is not None else {}),
                "epoch": epoch, "n_parameters": n_parameters, "mask_ratio_epoch": epoch_mask_ratio,
            }
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    print("Training time", datetime.timedelta(seconds=int(time.time() - start_time)))
    if output_dir and utils.is_main_process():
        save_training_curves(output_dir, history["epochs"], history["train_loss"],
                             history["train_lr"], history["val_loss"])
        with (output_dir / "loss_history.json").open("w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RNAtomy pretraining", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
