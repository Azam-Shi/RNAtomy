"""RNA-protein binding dataset (PrismNet benchmark: 17 RBPs, HeLa, 101-nt windows).

One CSV per RBP with columns split (train | val | test), row_index, sequence, label and icshape (JSON list,
-1 = not measured). Maps and descriptors are keyed by row_index:
<npy_root>/<row_index>.npy and <feat1d_root>/<row_index>/feat1d_norm.npy.

Each sample is a dict: x, label, onehot, icshape, ic_measured, feat, mask, L, row_index.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEQ_LEN = 101
_NUC = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def _one_hot(seq: str, L: int) -> np.ndarray:
    oh = np.zeros((L, 4), dtype=np.float32)
    for i, c in enumerate(seq[:L].upper()):
        j = _NUC.get(c)
        if j is not None:
            oh[i, j] = 1.0
    return oh


def _load_map_4ch(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"map ndim={arr.ndim} at {path}")
    if arr.shape[0] in (4, 5):
        return arr[:4]
    if arr.shape[-1] in (4, 5):
        return np.transpose(arr, (2, 0, 1))[:4]
    raise ValueError(f"map has no channel axis in {{4,5}}: {arr.shape} at {path}")


def _map_len(path: Path) -> int:
    s = np.load(path, mmap_mode="r").shape
    if len(s) != 3:
        raise ValueError(f"map ndim={len(s)}")
    if s[0] in (4, 5):
        return min(int(s[1]), int(s[2]))
    if s[-1] in (4, 5):
        return min(int(s[0]), int(s[1]))
    raise ValueError(f"map has no channel axis: {s}")


class RNAProteinDataset(Dataset):
    """csv_path: per-RBP CSV. split: train | val | test. npy_root / feat1d_root: per-split feature folders.
    require_structure=False tolerates missing maps (sequence-only heads).
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        npy_root: str = "",
        feat1d_root: Optional[str] = None,
        dssr_root: Optional[str] = None,
        file_prefix: Optional[str] = None,
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
        max_len: int = SEQ_LEN,
        require_structure: bool = True,
        split_name: Optional[str] = None,
    ) -> None:
        self.split = split
        self.split_name = split_name or split
        self.max_len = int(max_len)
        self.require_structure = require_structure

        self.npy_root = Path(npy_root) if npy_root else None
        self.feat1d_root = Path(feat1d_root) if feat1d_root else self.npy_root
        self.dssr_root = Path(dssr_root) if dssr_root else None
        self.norm_mean = (np.array(norm_mean, dtype=np.float32).reshape(5, 1, 1)
                          if norm_mean is not None else None)
        self.norm_std = (np.array(norm_std, dtype=np.float32).reshape(5, 1, 1)
                         if norm_std is not None else None)

        df = pd.read_csv(csv_path)
        need = {"split", "row_index", "sequence", "label", "icshape"}
        if not need.issubset(df.columns):
            raise ValueError(f"[{self.split_name}] CSV missing columns {need - set(df.columns)}")
        df = df[df["split"].astype(str) == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"[{self.split_name}] no rows with split=='{split}' in {csv_path}")

        self._prefix = (file_prefix if file_prefix is not None
                        else (self._infer_prefix(self.npy_root) if self.npy_root else ""))

        self.records: List[Dict] = []
        self._paths: List[Dict[str, Optional[Path]]] = []
        miss = 0
        first_bad = None
        for rid, seq, lab, ic in zip(df["row_index"].astype(str), df["sequence"].astype(str),
                                     df["label"].astype(float), df["icshape"].astype(str)):
            npy = feat = dssr = None
            if self.npy_root is not None:
                npy = self._resolve_map(rid)
                feat = self._resolve_feat1d(rid)
                dssr = self._resolve_dssr(rid)
                if (npy is None or feat is None) and require_structure:
                    miss += 1
                    if first_bad is None:
                        first_bad = (f"no {'map' if npy is None else 'feat1d'} for '{rid}' under "
                                     f"{self.npy_root if npy is None else self.feat1d_root}")
                    continue
            self.records.append({"row_index": rid, "sequence": seq.upper().replace("T", "U"),
                                 "label": float(lab), "icshape_json": ic})
            self._paths.append({"npy": npy, "feat": feat, "dssr": dssr})

        self.used_count = len(self.records)
        self.ignored_count = miss
        labels = [r["label"] for r in self.records]
        self.n_pos = int(sum(labels))
        self.n_neg = self.used_count - self.n_pos
        print(f"[RNAProteinDataset:{self.split_name}] used={self.used_count} ignored={miss} "
              f"| pos={self.n_pos} neg={self.n_neg}"
              + (f" | map prefix='{self._prefix}'" if self.npy_root else " | sequence-only"))
        if self.used_count == 0:
            raise ValueError(
                f"[{self.split_name}] 0 usable rows. {first_bad}\n"
                f"  npy_root   : {self.npy_root} -> {self._listing(self.npy_root)}\n"
                f"  feat1d_root: {self.feat1d_root} -> {self._listing(self.feat1d_root)}"
            )

    @staticmethod
    def _listing(root: Optional[Path], k: int = 10) -> List[str]:
        if root is None:
            return []
        try:
            return sorted(p.name for p in root.iterdir())[:k]
        except Exception as exc:                                   # noqa: BLE001
            return [f"<cannot list: {exc}>"]

    @staticmethod
    def _infer_prefix(root: Optional[Path]) -> str:
        if root is None or not root.exists():
            return ""
        heads: Dict[str, int] = {}
        n = 0
        for p in sorted(root.iterdir()):
            stem = p.stem if p.suffix == ".npy" else (p.name if p.is_dir() else None)
            if stem is None:
                continue
            heads[stem.split("_", 1)[0] + "_" if "_" in stem else ""] = \
                heads.get(stem.split("_", 1)[0] + "_" if "_" in stem else "", 0) + 1
            n += 1
            if n >= 400:
                break
        if not heads:
            return ""
        best, c = max(heads.items(), key=lambda kv: kv[1])
        return best[:-1] if (best and c >= 0.8 * n) else ""

    def _keys(self, rid: str) -> List[str]:
        ks = [rid]
        m = re.search(r"(\d+)$", rid)
        if m:
            i = int(m.group(1))
            ks += [str(i), f"{i:06d}"]
            if self._prefix:
                ks += [f"{self._prefix}_{i}", f"{self._prefix}_{i:06d}"]
        out, seen = [], set()
        for k in ks:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def _resolve_map(self, rid: str) -> Optional[Path]:
        for k in self._keys(rid):
            for c in (self.npy_root / f"{k}.npy", self.npy_root / f"_{k}.npy",
                      self.npy_root / k / f"{k}.npy"):
                if c.exists():
                    return c
        g = sorted(self.npy_root.glob(f"*{rid}.npy"))
        return g[0] if len(g) == 1 else None

    def _resolve_feat1d(self, rid: str) -> Optional[Path]:
        for k in self._keys(rid):
            for c in (self.feat1d_root / k / "feat1d_norm.npy",
                      self.feat1d_root / k / "feat1d.npy",
                      self.feat1d_root / f"{k}_feat1d_norm.npy",
                      self.feat1d_root / f"{k}_1d.npy",
                      self.feat1d_root / f"{k}.npy"):
                if c.exists():
                    return c
        return None

    def _resolve_dssr(self, rid: str) -> Optional[Path]:
        if self.dssr_root is None:
            return None
        for k in self._keys(rid):
            for c in (self.dssr_root / k / "bp_mask.npy", self.dssr_root / k / "bp.npy",
                      self.dssr_root / f"{k}_bp.npy", self.dssr_root / f"{k}.npy"):
                if c.exists():
                    return c
        return None

    def __len__(self) -> int:
        return len(self.records)

    def pos_weight(self) -> float:
        return (self.n_neg / self.n_pos) if self.n_pos > 0 else 1.0

    def _effective_len(self, seq: str, ic_len: int, p: Dict) -> int:
        L = min(len(seq), ic_len, self.max_len)
        if p["npy"] is not None:
            L = min(L, _map_len(p["npy"]))
        if p["feat"] is not None:
            fs = np.load(p["feat"], mmap_mode="r").shape
            if len(fs) == 2 and int(fs[1]) >= 23:
                L = min(L, int(fs[0]))
        return int(L)

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]
        p = self._paths[idx]
        seq = r["sequence"]
        ic_full = np.asarray(json.loads(r["icshape_json"]), dtype=np.float32)

        L = self._effective_len(seq, len(ic_full), p)
        if L <= 0:
            raise ValueError(f"[{self.split_name}] non-positive length for {r['row_index']}")

        icshape = ic_full[:L].copy()
        ic_measured = (icshape > -0.5).astype(np.float32)
        icshape[icshape < -0.5] = 0.0

        if p["npy"] is not None:
            x4 = _load_map_4ch(p["npy"])[:, :L, :L]
        else:
            x4 = np.zeros((4, L, L), dtype=np.float32)
        if p["dssr"] is not None and p["dssr"].exists():
            bp = np.squeeze(np.load(p["dssr"]).astype(np.float32))
            if bp.ndim != 2:
                raise ValueError(f"DSSR must squeeze to 2D, got {bp.shape}")
            bp = bp[:L, :L]
        else:
            bp = np.zeros((L, L), dtype=np.float32)
        x = np.concatenate([x4, bp[None]], axis=0)
        if self.norm_mean is not None and self.norm_std is not None:
            x = (x - self.norm_mean) / (self.norm_std + 1e-8)

        if p["feat"] is not None:
            feat = np.load(p["feat"]).astype(np.float32)[:L, :23]
        else:
            feat = np.zeros((L, 23), dtype=np.float32)

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "label": torch.tensor(r["label"], dtype=torch.float32),
            "onehot": torch.from_numpy(_one_hot(seq, L)),
            "icshape": torch.from_numpy(icshape),
            "ic_measured": torch.from_numpy(ic_measured),
            "feat": torch.from_numpy(np.ascontiguousarray(feat)),
            "mask": torch.ones(L, dtype=torch.bool),
            "L": L,
            "row_index": r["row_index"],
        }


def rna_protein_collate_fn(batch: List[Dict]) -> Dict:
    B = len(batch)
    Lmax = max(b["L"] for b in batch)
    x = torch.zeros(B, 5, Lmax, Lmax)
    oh = torch.zeros(B, Lmax, 4)
    ic = torch.zeros(B, Lmax)
    icm = torch.zeros(B, Lmax)
    feat = torch.zeros(B, Lmax, 23)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    labels = torch.zeros(B)
    Ls, rids = [], []
    for i, s in enumerate(batch):
        L = s["L"]
        x[i, :, :L, :L] = s["x"]
        oh[i, :L] = s["onehot"]
        ic[i, :L] = s["icshape"]
        icm[i, :L] = s["ic_measured"]
        feat[i, :L] = s["feat"]
        mask[i, :L] = s["mask"]
        labels[i] = s["label"]
        Ls.append(L)
        rids.append(s["row_index"])
    return {"x": x, "onehot": oh, "icshape": ic, "ic_measured": icm, "feat": feat,
            "mask": mask, "labels": labels, "Ls": Ls, "row_index": rids}
