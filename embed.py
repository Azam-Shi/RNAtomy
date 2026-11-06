"""Compute RNAtomy embeddings from precomputed trRosettaRNA maps and, optionally, DSSR residue descriptors.

    python embed.py --ckpt rnatomy.pth --maps maps/ --feat1d-root feats/ --level pooled --out embeddings.npz

Levels: pooled -> [D] per RNA, nucleotide -> [L, D], pair -> [L, L, D]. Without --feat1d-root the encoder runs on
the pairwise map alone.
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from models.rna_encoder import build_encoder_from_checkpoint

NORM_MEAN = (28.472, 93.731, 89.640, -8.994)
NORM_STD = (6.018, 15.843, 22.327, 21.095)
LEVELS = ("pooled", "nucleotide", "pair")


def load_encoder(checkpoint: str, device: Optional[str] = None, **kwargs):
    """Load a pretrained encoder in eval mode. Keyword arguments are forwarded to build_encoder_from_checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return build_encoder_from_checkpoint(checkpoint, device=device, **kwargs).eval()


def _prepare(enc, pair_map, feats):
    p = enc.patch_size
    length = pair_map.shape[0]
    device = next(enc.parameters()).device
    mean, std = torch.tensor(NORM_MEAN), torch.tensor(NORM_STD)
    x = ((torch.as_tensor(pair_map, dtype=torch.float32) - mean) / std).permute(2, 0, 1).unsqueeze(0)
    pad = (-length) % p
    x = F.pad(x, (0, pad, 0, pad)).to(device)
    valid = torch.zeros(1, length + pad, length + pad, dtype=torch.bool, device=device)
    valid[:, :length, :length] = True
    if feats is None:
        return x, valid, None, None
    f = torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    return x, valid, f, torch.ones(1, length, dtype=torch.bool, device=device)


def _embed_window(enc, pair_map, feats, level):
    length = pair_map.shape[0]
    x, valid, f, fv = _prepare(enc, pair_map, feats)
    if level == "pooled":
        return enc.encode_pooled(x, valid, f, fv)[0]
    if level == "nucleotide":
        nt, _ = enc.encode_nucleotides(x, valid, f, fv, seq_lens=[length])
        return nt[0]
    tokens, _, h, w = enc.encode_tokens(x, valid, f, fv)
    g = math.ceil(length / enc.patch_size)
    grid = tokens.view(1, h, w, -1)[:, :g, :g].permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(length, length), mode="bilinear", align_corners=False)
    return grid[0].permute(1, 2, 0)


@torch.no_grad()
def embed(enc, pair_map, feats=None, level: str = "pooled", window: int = 1024) -> np.ndarray:
    """Embed one RNA.

    pair_map: [L, L, 4] trRosettaRNA maps (raw values). feats: [L, 23] DSSR descriptors or None.
    RNAs longer than `window` are processed in overlapping windows whose outputs are averaged
    (not available for level="pair").
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    pair_map = np.asarray(pair_map)
    if pair_map.ndim != 3 or pair_map.shape[0] != pair_map.shape[1] or pair_map.shape[2] != 4:
        raise ValueError(f"expected a [L, L, 4] map, got {pair_map.shape}")
    length = pair_map.shape[0]
    if feats is not None and len(feats) != length:
        raise ValueError(f"descriptors have {len(feats)} rows but the map has L = {length}")

    window = min(window, (enc.backbone.max_grid - 1) * enc.patch_size)
    if length <= window:
        return _embed_window(enc, pair_map, feats, level).float().cpu().numpy()
    if level == "pair":
        raise ValueError(f"pair embeddings need L <= {window}, got L = {length}")

    starts = list(range(0, length - window + 1, window // 2))
    if starts[-1] + window < length:
        starts.append(length - window)
    total, count = None, None
    for s in starts:
        e = s + window
        out = _embed_window(enc, pair_map[s:e, s:e], None if feats is None else feats[s:e], level).float().cpu()
        if level == "pooled":
            total = out if total is None else total + out
            count = 1 if count is None else count + 1
        else:
            if total is None:
                total, count = torch.zeros(length, out.shape[-1]), torch.zeros(length, 1)
            total[s:e] += out
            count[s:e] += 1
    return (total / count).numpy()


def _find_feats(root: Path, rna_id: str) -> Optional[Path]:
    for name in (rna_id, rna_id.lstrip("_")):
        path = root / name / "feat1d_norm.npy"
        if path.exists():
            return path
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="pretrained checkpoint (rnatomy.pth)")
    ap.add_argument("--maps", required=True, help="folder of <id>.npy trRosettaRNA maps, each [L, L, 4]")
    ap.add_argument("--feat1d-root", default="", help="folder with <id>/feat1d_norm.npy; omit for pairwise-only")
    ap.add_argument("--level", default="pooled", choices=LEVELS)
    ap.add_argument("--out", required=True, help="pooled: .npz file; nucleotide / pair: output folder")
    ap.add_argument("--device", default=None)
    ap.add_argument("--window", type=int, default=1024, help="window length for RNAs that exceed it")
    ap.add_argument("--fp16", action="store_true", help="save nucleotide / pair embeddings in half precision")
    args = ap.parse_args()

    enc = load_encoder(args.ckpt, device=args.device)
    ids = sorted(p.stem for p in Path(args.maps).glob("*.npy"))
    if not ids:
        raise SystemExit(f"no .npy files in {args.maps}")
    feat_root = Path(args.feat1d_root) if args.feat1d_root else None

    out_dir = None
    if args.level != "pooled":
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

    names, pooled = [], []
    for i, rna_id in enumerate(ids, 1):
        pair_map = np.load(Path(args.maps) / f"{rna_id}.npy")
        feats = None
        if feat_root is not None:
            path = _find_feats(feat_root, rna_id)
            if path is None:
                print(f"[{rna_id}] no descriptors found, using the pairwise map only")
            else:
                feats = np.load(path)
        emb = embed(enc, pair_map, feats, args.level, args.window)
        if args.level == "pooled":
            names.append(rna_id)
            pooled.append(emb)
        else:
            np.save(out_dir / f"{rna_id}.npy", emb.astype(np.float16) if args.fp16 else emb)
        if i % 100 == 0 or i == len(ids):
            print(f"{i}/{len(ids)}")

    if args.level == "pooled":
        np.savez(args.out, ids=np.array(names), embeddings=np.stack(pooled))


if __name__ == "__main__":
    main()
