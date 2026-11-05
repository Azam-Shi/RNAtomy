"""Argument parser, encoder construction, optimizer, scheduler and checkpoint helpers shared by the fine-tuning scripts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

from models.rna_encoder import RNAEncoder, build_encoder_from_checkpoint


def get_base_args_parser() -> argparse.ArgumentParser:
    """Backbone and training arguments shared by every fine-tuning task."""
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument("--pretrain-ckpt", required=True, help="pretraining checkpoint")
    p.add_argument("--use-ema", action="store_true", default=True, help="load the EMA weights")
    p.add_argument("--no-use-ema", action="store_false", dest="use_ema")

    p.add_argument("--patch-size", default=8, type=int)
    p.add_argument("--in-chans", default=4, type=int)
    p.add_argument("--embed-dim", default=768, type=int)
    p.add_argument("--depth", default=16, type=int)
    p.add_argument("--mamba-ratio", default=3, type=int)
    p.add_argument("--window-size", default=16, type=int)
    p.add_argument("--window-num-heads", default=12, type=int)
    p.add_argument("--nt-feat-dim", default=23, type=int)
    p.add_argument("--use-1d-conditioning", action="store_true", default=True)
    p.add_argument("--feat1d-cond-dim", default=256, type=int, help="must match pretraining")
    p.add_argument("--feat1d-encoder-layers", default=2, type=int, help="must match pretraining")
    p.add_argument("--feat1d-mode", default="full", choices=["full", "none", "moddrop"],
                   help="full: always use the residue descriptors; none: pairwise-only path; "
                        "moddrop: drop the whole track per sample while training")
    p.add_argument("--feat1d-moddrop-prob", default=0.5, type=float)
    p.add_argument("--feat1d-eval-mode", default="full", choices=["full", "none"],
                   help="descriptor policy at evaluation when --feat1d-mode is moddrop")
    p.add_argument("--aux-recon-1d-weight", default=0.0, type=float,
                   help="weight of the descriptor-reconstruction loss added during fine-tuning (0 = off)")

    p.add_argument("--freeze-backbone", action="store_true", default=False)
    p.add_argument("--unfreeze-last-n-blocks", default=0, type=int,
                   help="with --freeze-backbone, keep the last N blocks trainable")

    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--backbone-lr-mult", default=0.1, type=float, help="backbone lr = lr * this")
    p.add_argument("--weight-decay", default=0.05, type=float)
    p.add_argument("--clip-grad", default=1.0, type=float)
    p.add_argument("--accum-iter", default=1, type=int)

    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--warmup-epochs", default=10, type=int)
    p.add_argument("--min-lr", default=1e-6, type=float)

    p.add_argument("--data-path", required=True, help="folder with train/ and val/ subfolders")
    p.add_argument("--dssr-root", default="", type=str)
    p.add_argument("--feat1d-root", default="", type=str)
    p.add_argument("--feat1d-file", default="feat1d_norm.npy", type=str)
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--num-workers", default=4, type=int)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--resume", default="", type=str, help="fine-tuning checkpoint to resume from")
    p.add_argument("--val-every", default=1, type=int)
    p.add_argument("--save-freq", default=10, type=int)
    p.add_argument("--early-stop-patience", default=20, type=int)

    return p


def build_encoder(args, device: str = "cpu") -> RNAEncoder:
    """Build the encoder from the CLI arguments and apply the freezing policy."""
    encoder = build_encoder_from_checkpoint(
        args.pretrain_ckpt,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        embed_dim=args.embed_dim,
        depth=args.depth,
        mamba_ratio=args.mamba_ratio,
        window_size=args.window_size,
        window_num_heads=args.window_num_heads,
        use_1d_conditioning=args.use_1d_conditioning,
        nt_feat_dim=args.nt_feat_dim,
        feat1d_cond_dim=getattr(args, "feat1d_cond_dim", 256),
        feat1d_encoder_layers=getattr(args, "feat1d_encoder_layers", 2),
        recon_1d_weight=(1.0 if getattr(args, "aux_recon_1d_weight", 0.0) > 0.0 else 0.0),
        feat1d_mode=getattr(args, "feat1d_mode", "full"),
        feat1d_moddrop_prob=getattr(args, "feat1d_moddrop_prob", 0.5),
        feat1d_eval_mode=getattr(args, "feat1d_eval_mode", "full"),
        use_ema=args.use_ema,
        device=device,
    )

    if args.freeze_backbone:
        encoder.freeze()
        if args.unfreeze_last_n_blocks > 0:
            encoder.unfreeze_last_n_blocks(args.unfreeze_last_n_blocks)
        print(f"Backbone frozen; trainable parameters: {encoder.trainable_parameter_count():,}")
    else:
        print("Backbone unfrozen (full fine-tuning)")
    if getattr(args, "aux_recon_1d_weight", 0.0) > 0.0 and not hasattr(encoder.backbone, "recon_1d_head"):
        raise SystemExit(
            "--aux-recon-1d-weight > 0 requires a checkpoint pretrained with --recon-1d-weight > 0"
        )
    return encoder


def aux_1d_recon_loss(encoder, x, nt_features, nt_valid_mask, weight, seq_lens=None):
    """Weighted descriptor-reconstruction loss, added to the task loss at every training step.

    Returns 0 when the weight is 0, the backbone is frozen or the checkpoint has no reconstruction head."""
    zero = x.new_zeros(())
    if weight is None or float(weight) <= 0.0 or nt_features is None:
        return zero
    bb = encoder.backbone
    if (not getattr(bb, "use_1d_conditioning", False)) or (not hasattr(bb, "recon_1d_head")):
        return zero
    if not any(p.requires_grad for p in bb.parameters()):
        return zero

    P = int(getattr(bb, "patch_size", encoder.patch_size))
    H, W = x.shape[-2], x.shape[-1]
    ph, pw = (-H) % P, (-W) % P
    x_pad = F.pad(x, (0, pw, 0, ph)) if (ph or pw) else x
    Hp, Wp = x_pad.shape[-2], x_pad.shape[-1]

    if seq_lens is not None:
        lens = [int(v) for v in seq_lens]
    else:
        lens = nt_valid_mask.sum(dim=1).tolist()
    ev = torch.zeros(x_pad.shape[0], Hp, Wp, dtype=torch.bool, device=x.device)
    for b, L in enumerate(lens):
        L = int(L)
        ev[b, :L, :L] = True

    loss = bb.aux_1d_recon_loss(x_pad, valid_mask=ev, nt_features=nt_features, nt_valid_mask=nt_valid_mask)
    return zero if loss is None else float(weight) * loss


def build_optimizer(
    encoder: RNAEncoder,
    head: torch.nn.Module,
    args,
) -> optim.AdamW:
    """AdamW with two groups: the head at `lr` and the trainable backbone parameters at `lr * backbone_lr_mult`."""
    backbone_params = [p for p in encoder.parameters() if p.requires_grad]
    head_params = list(head.parameters())

    param_groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": args.lr * args.backbone_lr_mult,
        })

    return optim.AdamW(param_groups, weight_decay=args.weight_decay)


def build_lr_scheduler(optimizer: optim.Optimizer, args) -> optim.lr_scheduler.LambdaLR:
    warmup = max(1, args.warmup_epochs)
    total = args.epochs
    min_lr_ratio = args.min_lr / args.lr

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        t = (epoch - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    output_dir: Path,
    encoder: RNAEncoder,
    head: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    filename: str = "checkpoint.pth",
) -> None:
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        output_dir / filename,
    )


def load_checkpoint(
    path: str,
    encoder: RNAEncoder,
    head: torch.nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """Returns the epoch to resume from (epoch + 1)."""
    ckpt = torch.load(path, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    print(f"[FINETUNE] Resumed from epoch {start_epoch - 1}  "
          f"metrics={ckpt.get('metrics', {})}")
    return start_epoch
