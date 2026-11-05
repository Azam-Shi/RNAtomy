"""Long-range base-pair dataset (bpRNA-1m TR0, RNA-km benchmark).

dataset_B.csv holds id, sequence and secondary_structure (dot-bracket, pseudoknots included); the label is the
base-pair matrix parsed from it. Inputs per RNA <id>: <npy_root>/<id>.npy (L x L x 4 map) and
<feat1d_root>/<id>/feat1d_norm.npy (L x 23). Long-range pairs are those with |i - j| >= 101.

Each sample: x [4, L, L], target [L, L], valid_mask, nt_features [L, 23], nt_valid_mask, L, meta.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_BRACKETS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
_BRACKETS += [(chr(65 + k), chr(97 + k)) for k in range(26)]
_OPEN = {o for o, _ in _BRACKETS}
_CLOSE_TO_OPEN = {c: o for o, c in _BRACKETS}


def dotbracket_to_pairs(ss: str) -> List[Tuple[int, int]]:
    """Dot-bracket string -> list of (i, j) 0-based base pairs (i < j). Pseudoknot-aware."""
    stacks: Dict[str, List[int]] = {o: [] for o in _OPEN}
    pairs: List[Tuple[int, int]] = []
    for i, c in enumerate(ss):
        if c in _OPEN:
            stacks[c].append(i)
        elif c in _CLOSE_TO_OPEN:
            st = stacks[_CLOSE_TO_OPEN[c]]
            if st:
                pairs.append((st.pop(), i))
    return pairs


def pairs_to_matrix(pairs: List[Tuple[int, int]], L: int) -> np.ndarray:
    m = np.zeros((L, L), dtype=np.float32)
    for i, j in pairs:
        if 0 <= i < L and 0 <= j < L:
            m[i, j] = m[j, i] = 1.0
    return m


def is_crossing(pairs: List[Tuple[int, int]]) -> np.ndarray:
    """Boolean array over `pairs`: True if the pair crosses another (pseudoknot pair)."""
    out = np.zeros(len(pairs), dtype=bool)
    for a in range(len(pairs)):
        i, j = pairs[a]
        for b in range(len(pairs)):
            if a == b:
                continue
            k, l = pairs[b]
            if (i < k < j < l) or (k < i < l < j):
                out[a] = True
                break
    return out


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


def load_split_ids(path: str) -> List[str]:
    ids: List[str] = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                line = line[1:]
            ids.append(line.split(",", 1)[0].split()[0])
    return ids


def read_dataset_b(csv_path: str) -> Dict[str, Dict[str, str]]:
    """id -> {sequence, secondary_structure}."""
    out: Dict[str, Dict[str, str]] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["id"]] = {"sequence": row["sequence"].upper().replace("T", "U"),
                              "secondary_structure": row["secondary_structure"]}
    return out


class LRBPDataset(Dataset):
    def __init__(
        self,
        split_ids: List[str],
        csv_path: str,
        npy_root: str,
        feat1d_root: str,
        dssr_root: Optional[str] = None,
        label_source: str = "dot_bracket",
        min_range: int = 101,
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
        feat1d_dim: int = 23,
        split_name: str = "dataset",
    ) -> None:
        if label_source not in ("dot_bracket", "bp_mask"):
            raise ValueError("label_source must be dot_bracket | bp_mask")
        self.label_source = label_source
        self.min_range = int(min_range)
        self.feat1d_dim = feat1d_dim
        self.split_name = split_name
        self.npy_root = Path(npy_root)
        self.feat1d_root = Path(feat1d_root)
        self.dssr_root = Path(dssr_root) if dssr_root else None
        self.norm_mean = (np.array(norm_mean, dtype=np.float32).reshape(4, 1, 1)
                          if norm_mean is not None else None)
        self.norm_std = (np.array(norm_std, dtype=np.float32).reshape(4, 1, 1)
                         if norm_std is not None else None)

        table = read_dataset_b(csv_path)
        self.records: List[Dict] = []
        self._paths: List[Dict[str, Optional[Path]]] = []
        miss = 0
        first_bad = None
        for rid in split_ids:
            if rid not in table:
                miss += 1
                first_bad = first_bad or f"'{rid}' not in {Path(csv_path).name}"
                continue
            npy = self._resolve(self.npy_root, rid, ".npy")
            feat = self._resolve_feat1d(rid)
            if npy is None or feat is None:
                miss += 1
                first_bad = first_bad or (f"no {'map' if npy is None else 'feat1d'} for '{rid}'")
                continue
            bp = None
            if label_source == "bp_mask":
                bp = self._resolve_bp_mask(rid)
                if bp is None:
                    miss += 1
                    first_bad = first_bad or f"no bp_mask for '{rid}'"
                    continue
            seq = table[rid]["sequence"]
            ss = table[rid]["secondary_structure"]
            self.records.append({"id": rid, "seq": seq, "ss": ss})
            self._paths.append({"npy": npy, "feat": feat, "bp": bp})

        self.used_count = len(self.records)
        self.ignored_count = miss
        self.n_lr: List[int] = []
        self.has_pk: List[int] = []
        for r in self.records:
            pairs = dotbracket_to_pairs(r["ss"])
            n_lr = sum(1 for i, j in pairs if abs(i - j) >= self.min_range)
            self.n_lr.append(n_lr)
            self.has_pk.append(int(is_crossing(pairs).any()) if pairs else 0)

        print(f"[LRBPDataset:{split_name}] used={self.used_count} ignored={miss} "
              f"| RNAs with >=1 long-range pair: {sum(1 for n in self.n_lr if n > 0)} "
              f"| pseudoknot RNAs: {sum(self.has_pk)}")
        if self.used_count == 0:
            raise ValueError(
                f"[{split_name}] 0 usable RNAs. {first_bad}\n"
                f"  npy_root={self.npy_root} -> {self._listing(self.npy_root)}\n"
                f"  feat1d_root={self.feat1d_root} -> {self._listing(self.feat1d_root)}"
            )

    @staticmethod
    def _listing(root: Path, k: int = 10) -> List[str]:
        try:
            return sorted(p.name for p in root.iterdir())[:k]
        except Exception as exc:                                   # noqa: BLE001
            return [f"<cannot list: {exc}>"]

    def _resolve(self, root: Path, rid: str, suffix: str) -> Optional[Path]:
        for c in (root / f"{rid}{suffix}", root / f"_{rid}{suffix}",
                  root / f"_={rid}{suffix}", root / rid / f"{rid}{suffix}"):
            if c.exists():
                return c
        g = sorted(root.glob(f"*{rid}{suffix}"))
        return g[0] if len(g) == 1 else None

    def _resolve_feat1d(self, rid: str) -> Optional[Path]:
        for c in (self.feat1d_root / rid / "feat1d_norm.npy",
                  self.feat1d_root / rid / "feat1d.npy",
                  self.feat1d_root / f"{rid}_feat1d_norm.npy",
                  self.feat1d_root / f"{rid}.npy"):
            if c.exists():
                return c
        return None

    def _resolve_bp_mask(self, rid: str) -> Optional[Path]:
        if self.dssr_root is None:
            return None
        for c in (self.dssr_root / rid / "bp_mask.npy", self.dssr_root / rid / "bp.npy",
                  self.dssr_root / f"{rid}_bp.npy"):
            if c.exists():
                return c
        return None

    def __len__(self) -> int:
        return len(self.records)

    def _effective_len(self, seq: str, npy: Path, feat: Path) -> int:
        fs = np.load(feat, mmap_mode="r").shape
        L = min(len(seq), _map_len(npy))
        if len(fs) == 2 and int(fs[1]) >= 23:
            L = min(L, int(fs[0]))
        return int(L)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        p = self._paths[idx]
        L = self._effective_len(r["seq"], p["npy"], p["feat"])
        if L <= 0:
            raise ValueError(f"[{self.split_name}] non-positive L for {r['id']}")

        x4 = _load_map_4ch(p["npy"])[:, :L, :L]
        x = torch.from_numpy(np.ascontiguousarray(x4))
        if self.norm_mean is not None and self.norm_std is not None:
            x = (x - torch.from_numpy(self.norm_mean)) / (torch.from_numpy(self.norm_std) + 1e-8)

        feat = np.load(p["feat"]).astype(np.float32)[:L, :23]
        nt_features = torch.from_numpy(np.ascontiguousarray(feat))
        nt_valid_mask = torch.ones(L, dtype=torch.bool)

        if self.label_source == "bp_mask":
            bp = np.squeeze(np.load(p["bp"]).astype(np.float32))[:L, :L]
            target = torch.from_numpy(np.ascontiguousarray(bp))
            target = torch.maximum(target, target.transpose(0, 1))
        else:
            pairs = dotbracket_to_pairs(r["ss"])
            target = torch.from_numpy(pairs_to_matrix(pairs, L))

        valid_mask = torch.ones(L, L, dtype=torch.bool)
        meta = {"id": r["id"], "n_lr": int(self.n_lr[idx]), "has_pk": int(self.has_pk[idx]),
                "seq": r["seq"][:L], "ss": r["ss"][:L]}
        return x, target, valid_mask, nt_features, nt_valid_mask, L, meta


def lrbp_collate_fn(batch):
    xs, tgts, vms, nfs, nvs, Ls, metas = zip(*batch)
    return (torch.stack(xs, 0), torch.stack(tgts, 0), torch.stack(vms, 0),
            torch.stack(nfs, 0), torch.stack(nvs, 0), list(Ls), list(metas))
