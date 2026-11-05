"""IRES identification dataset: 174-nt sequences labelled IRES / non-IRES.

Records come from a CSV (id, sequence, label; column names are auto-detected) or a FASTA whose header
encodes the label. Per RNA <id>: <npy_root>/<id>.npy (trRosettaRNA map) and
<feat1d_root>/<id>/feat1d_norm.npy (L x 23). Each sample is a dict: x, onehot, feat, mask, label, L, id.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from finetune.rna_protein_dataset import RNAProteinDataset, _load_map_4ch, _map_len, _one_hot

_ID_COLS = ("id", "name", "seq_id", "seqid", "rna_id", "row_index", "index", "idx", "header")
_SEQ_COLS = ("sequence", "seq", "rna", "rna_seq", "sequences")
_LAB_COLS = ("label", "labels", "y", "class", "target", "is_ires", "ires")
_POS_WORDS = ("ires", "pos", "true", "yes")
_NEG_WORDS = ("non", "neg", "false")


def _parse_label(v) -> int:
    """0/1 from a number or a word ('IRES', 'non-IRES', 'pos', ...)."""
    if isinstance(v, (int, float, np.integer, np.floating)) and float(v) in (0.0, 1.0):
        return int(v)
    s = str(v).strip().lower()
    if s in ("0", "1"):
        return int(s)
    if any(w in s for w in _NEG_WORDS):
        return 0
    if any(w in s for w in _POS_WORDS):
        return 1
    raise ValueError(f"cannot interpret label {v!r} as 0/1")


def _pick(cols: Sequence[str], wanted: Sequence[str]) -> Optional[str]:
    low = {c.lower().strip(): c for c in cols}
    for w in wanted:
        if w in low:
            return low[w]
    return None


def load_ires_records(csv_path: str = "", fasta_path: str = "") -> List[Dict]:
    """Return [{id, seq, label}] from a CSV or a FASTA (exactly one must be given)."""
    if bool(csv_path) == bool(fasta_path):
        raise ValueError("pass exactly one of the CSV or FASTA for each split")
    recs: List[Dict] = []
    if csv_path:
        df = pd.read_csv(csv_path)
        sc = _pick(df.columns, _SEQ_COLS)
        lc = _pick(df.columns, _LAB_COLS)
        ic = _pick(df.columns, _ID_COLS)
        if sc is None or lc is None:
            raise ValueError(f"{csv_path}: need sequence+label columns; found {list(df.columns)}")
        for i, row in df.reset_index(drop=True).iterrows():
            rid = str(row[ic]) if ic is not None else str(i)
            recs.append({"id": rid, "seq": str(row[sc]).strip().upper().replace("T", "U"),
                         "label": _parse_label(row[lc])})
    else:
        hdr, seq = None, []

        def flush():
            if hdr is None:
                return
            m = (re.search(r"label[=:_]?\s*([01])\b", hdr, re.I)
                 or re.search(r"[|,;_\s]([01])\s*$", hdr))
            lab = int(m.group(1)) if m else _parse_label(hdr)
            rid = re.split(r"[|\s,;]", hdr.strip(), maxsplit=1)[0]
            recs.append({"id": rid, "seq": "".join(seq).upper().replace("T", "U"),
                         "label": lab, "header": hdr})

        for line in open(fasta_path):
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                hdr, seq = line[1:], []
            elif line.strip():
                seq.append(line.strip())
        flush()
    if not recs:
        raise ValueError(f"no records read from {csv_path or fasta_path}")
    return recs


def load_ires_label_files(pos_fasta: str, neg_fasta: str) -> List[Dict]:
    """Records from one positive FASTA + one negative FASTA (label = which file).
    Headers are just the ids (e.g. `>51`); ids must be unique across both files."""
    recs: List[Dict] = []
    for path, lab in ((pos_fasta, 1), (neg_fasta, 0)):
        rid, seq = None, []
        for line in list(open(path)) + [">"]:
            line = line.strip()
            if line.startswith(">"):
                if rid is not None:
                    recs.append({"id": rid, "seq": "".join(seq).upper().replace("T", "U"),
                                 "label": lab})
                rid, seq = line[1:].strip(), []
            elif line:
                seq.append(line)
    ids = [r["id"] for r in recs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate ids across {pos_fasta} / {neg_fasta}: map files "
                         f"<id>.npy would be ambiguous")
    if not recs:
        raise ValueError(f"no records in {pos_fasta} / {neg_fasta}")
    return recs


def stratified_split(records: List[Dict], frac: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """(train, val) with `frac` of each class held out; deterministic in `seed`."""
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for lab in (0, 1):
        grp = [r for r in records if r["label"] == lab]
        order = rng.permutation(len(grp))
        n_val = int(round(frac * len(grp)))
        va += [grp[i] for i in order[:n_val]]
        tr += [grp[i] for i in order[n_val:]]
    return tr, va


def resample_train(records: List[Dict], mode: str, seed: int,
                   over_factor: int = 3, under_ratio: float = 1.0) -> List[Dict]:
    """
    Paper protocol for the imbalanced training set:
      under : all positives + a random (seeded, fixed) negative subset of
              size under_ratio * n_pos            -> one ensemble member each seed
      over  : every positive repeated `over_factor` times (paper: x3)
      none  : unchanged
    """
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    if mode == "under":
        rng = np.random.RandomState(seed)
        k = min(len(neg), int(round(under_ratio * len(pos))))
        neg = [neg[i] for i in rng.choice(len(neg), size=k, replace=False)]
        return pos + neg
    if mode == "over":
        return pos * int(over_factor) + neg
    return records


class IRESDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        npy_root: str = "",
        feat1d_root: Optional[str] = None,
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
        max_len: int = 0,
        require_structure: bool = True,
        split_name: str = "",
    ) -> None:
        self.split_name = split_name
        self.max_len = int(max_len)
        self.npy_root = Path(npy_root) if npy_root else None
        self.feat1d_root = Path(feat1d_root) if feat1d_root else self.npy_root
        self.norm_mean = (np.array(norm_mean[:4], dtype=np.float32).reshape(4, 1, 1)
                          if norm_mean is not None else None)
        self.norm_std = (np.array(norm_std[:4], dtype=np.float32).reshape(4, 1, 1)
                         if norm_std is not None else None)

        self._prefix_cache: Dict[str, str] = {}
        self.records: List[Dict] = []
        self._paths: List[Dict[str, Optional[Path]]] = []
        cache: Dict[str, Tuple[Optional[Path], Optional[Path]]] = {}
        miss, first_bad = 0, None
        for r in records:
            if r["id"] not in cache:
                npy = feat = None
                if self.npy_root is not None:
                    npy = self._resolve(self.npy_root, r, "map")
                    feat = self._resolve(self.feat1d_root, r, "feat")
                cache[r["id"]] = (npy, feat)
            npy, feat = cache[r["id"]]
            if self.npy_root is not None and require_structure and (npy is None or feat is None):
                miss += 1
                first_bad = first_bad or (f"no {'map' if npy is None else 'feat1d'} for id "
                                          f"'{r['id']}'")
                continue
            self.records.append(r)
            self._paths.append({"npy": npy, "feat": feat})

        self.n_pos = sum(r["label"] for r in self.records)
        self.n_neg = len(self.records) - self.n_pos
        print(f"[IRESDataset:{split_name}] used={len(self.records)} ignored={miss} "
              f"| pos={self.n_pos} neg={self.n_neg}")
        if not self.records:
            raise ValueError(
                f"[{split_name}] 0 usable records. {first_bad}\n"
                f"  npy_root   : {self.npy_root} -> {self._listing(self.npy_root)}\n"
                f"  feat1d_root: {self.feat1d_root} -> {self._listing(self.feat1d_root)}\n"
                f"  first ids  : {[r['id'] for r in records[:5]]}")

    @staticmethod
    def _listing(root: Optional[Path], k: int = 8) -> List[str]:
        try:
            return sorted(p.name for p in root.iterdir())[:k] if root else []
        except Exception as exc:                                   # noqa: BLE001
            return [f"<cannot list: {exc}>"]

    def _keys(self, rec: Dict, prefix: str = "") -> List[str]:
        raw = [rec["id"], rec.get("header", "")]
        ks: List[str] = []
        for s in raw:
            if not s:
                continue
            ks += [s, re.sub(r"[^\w.\-]", "_", s), re.split(r"[|\s,;]", s.strip())[0]]
        m = re.search(r"(\d+)$", rec["id"])
        if m:
            i = int(m.group(1))
            ks += [str(i), f"{i:06d}"]
            if prefix:
                ks += [f"{prefix}_{i}", f"{prefix}_{i:06d}"]
        out, seen = [], set()
        for k in ks:
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def _resolve(self, root: Path, rec: Dict, kind: str) -> Optional[Path]:
        prefix = self._prefix_cache.setdefault(str(root), RNAProteinDataset._infer_prefix(root))
        for k in self._keys(rec, prefix):
            cands = ((root / f"{k}.npy", root / k / f"{k}.npy") if kind == "map" else
                     (root / k / "feat1d_norm.npy", root / k / "feat1d.npy",
                      root / f"{k}_feat1d_norm.npy", root / f"{k}.npy"))
            for c in cands:
                if c.exists():
                    return c
        g = sorted(root.glob(f"*_{rec['id']}.npy")) if kind == "map" else \
            sorted(root.glob(f"*_{rec['id']}/feat1d_norm.npy"))
        return g[0] if len(g) == 1 else None

    def __len__(self) -> int:
        return len(self.records)

    def pos_weight(self) -> float:
        return (self.n_neg / self.n_pos) if self.n_pos > 0 else 1.0

    def _length(self, seq: str, p: Dict) -> int:
        L = len(seq)
        if p["npy"] is not None:
            L = min(L, _map_len(p["npy"]))
        if p["feat"] is not None:
            fs = np.load(p["feat"], mmap_mode="r").shape
            if len(fs) == 2 and int(fs[1]) >= 23:
                L = min(L, int(fs[0]))
        return min(L, self.max_len) if self.max_len > 0 else L

    def __getitem__(self, idx: int) -> Dict:
        r, p = self.records[idx], self._paths[idx]
        L = self._length(r["seq"], p)
        if L <= 0:
            raise ValueError(f"[{self.split_name}] non-positive length for {r['id']}")
        if p["npy"] is not None:
            x = _load_map_4ch(p["npy"])[:, :L, :L]
            if self.norm_mean is not None and self.norm_std is not None:
                x = (x - self.norm_mean) / (self.norm_std + 1e-8)
        else:
            x = np.zeros((4, L, L), dtype=np.float32)
        if p["feat"] is not None:
            feat = np.load(p["feat"]).astype(np.float32)[:L, :23]
        else:
            feat = np.zeros((L, 23), dtype=np.float32)
        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "label": torch.tensor(float(r["label"])),
            "onehot": torch.from_numpy(_one_hot(r["seq"], L)),
            "feat": torch.from_numpy(np.ascontiguousarray(feat)),
            "L": L, "id": r["id"],
        }


def ires_collate_fn(batch: List[Dict]) -> Dict:
    B, Lm = len(batch), max(b["L"] for b in batch)
    x = torch.zeros(B, 4, Lm, Lm)
    oh = torch.zeros(B, Lm, 4)
    feat = torch.zeros(B, Lm, 23)
    mask = torch.zeros(B, Lm, dtype=torch.bool)
    labels = torch.zeros(B)
    for i, s in enumerate(batch):
        L = s["L"]
        x[i, :, :L, :L] = s["x"]
        oh[i, :L] = s["onehot"]
        feat[i, :L] = s["feat"]
        mask[i, :L] = True
        labels[i] = s["label"]
    return {"x": x, "onehot": oh, "feat": feat, "mask": mask, "labels": labels,
            "Ls": [s["L"] for s in batch], "ids": [s["id"] for s in batch]}
