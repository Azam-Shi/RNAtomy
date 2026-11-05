"""RNA-RNA interaction dataset (DeepMirTar: miRNA x mRNA target-site pairs, split as in RNAErnie).

Each split has <split>.csv (a_name, a_seq, b_name, b_seq, label) and <split>.fasta (one `>name,seq` line per
unique RNA). Maps and descriptors are keyed by the RNA's FASTA index or name:
<feat_root>/<key>.npy and <feat1d_root>/<key>/feat1d_norm.npy.

Each sample is a dict with the maps, descriptors, one-hot sequences and masks of both RNAs, the
engineered seed features, the label and both lengths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MAX_LEN_A = 26
MAX_LEN_B = 60

_NUC = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
_CMP = {"A": "U", "U": "A", "G": "C", "C": "G", "T": "A", "N": "N"}
_WC = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
_GU = {("G", "U"), ("U", "G")}

SEED_FEAT_DIM = 11


def _to_rna(seq: str) -> str:
    return str(seq).upper().replace("T", "U")


def _deN(name: str) -> str:
    """Reverse the split writer's 'strip every N' corruption for name matching."""
    return str(name).replace("N", "").replace("n", "")


def _one_hot(seq: str, L: int) -> np.ndarray:
    oh = np.zeros((L, 4), dtype=np.float32)
    for i, c in enumerate(seq[:L]):
        j = _NUC.get(c)
        if j is not None:
            oh[i, j] = 1.0
    return oh


def _revcomp(s: str) -> str:
    return "".join(_CMP.get(c, "N") for c in reversed(s))


def _local_complementary_score(a: str, b: str, allow_gu: bool) -> float:
    """Smith-Waterman local alignment of miRNA `a` (5'->3') vs reverse of `b`,
    scoring an antiparallel complementary duplex (WC +1, G.U +0.5, mm -1, gap -2)."""
    br = b[::-1]
    n, m = len(a), len(br)
    if n == 0 or m == 0:
        return 0.0
    prev = [0.0] * (m + 1)
    best = 0.0
    for i in range(1, n + 1):
        cur = [0.0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            pair = (ai, br[j - 1])
            s = 1.0 if pair in _WC else (0.5 if (allow_gu and pair in _GU) else -1.0)
            v = max(0.0, prev[j - 1] + s, prev[j] - 2.0, cur[j - 1] - 2.0)
            cur[j] = v
            if v > best:
                best = v
        prev = cur
    return best


def _gc_frac(s: str) -> float:
    n = max(len(s), 1)
    return (s.count("G") + s.count("C")) / n


def compute_seed_features(a_seq: str, b_seq: str) -> np.ndarray:
    a, b = _to_rna(a_seq), _to_rna(b_seq)
    seed_67 = _revcomp(a[1:7])
    seed_28 = _revcomp(a[1:8])
    off_38 = _revcomp(a[2:8])
    has_6mer = 1.0 if (len(seed_67) == 6 and seed_67 in b) else 0.0
    has_7mer = 1.0 if (len(seed_28) == 7 and seed_28 in b) else 0.0
    has_off6 = 1.0 if (len(off_38) == 6 and off_38 in b) else 0.0
    sw_wc = _local_complementary_score(a, b, False)
    sw_gu = _local_complementary_score(a, b, True)
    seed = a[1:8]
    seed_wc = _local_complementary_score(seed, b, False)
    seed_gu = _local_complementary_score(seed, b, True)
    na = max(len(a), 1)
    return np.array([
        has_6mer, has_7mer, has_off6,
        sw_wc / na, sw_gu / na, seed_wc / 7.0, seed_gu / 7.0,
        _gc_frac(a), _gc_frac(b),
        len(a) / float(MAX_LEN_A), len(b) / float(MAX_LEN_B),
    ], dtype=np.float32)


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


def parse_pair_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """`<split>.fasta` -> ordered [(name, seq)]; every line is `>name,seq`."""
    out: List[Tuple[str, str]] = []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                line = line[1:]
            if "," not in line:
                raise ValueError(f"unexpected fasta line (no comma): {line[:60]!r}")
            name, seq = line.split(",", 1)
            out.append((name.strip(), _to_rna(seq.strip())))
    return out


class RRIDataset(Dataset):
    """csv_path / fasta_path: split files. npy_root: folder with the 2D maps. feat1d_root: descriptor folder
    (defaults to npy_root). norm_mean / norm_std: per-channel statistics.
    """

    def __init__(
        self,
        csv_path: str,
        fasta_path: str,
        npy_root: str,
        feat1d_root: Optional[str] = None,
        dssr_root: Optional[str] = None,
        file_prefix: Optional[str] = None,
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
        max_len_a: int = MAX_LEN_A,
        max_len_b: int = MAX_LEN_B,
        split_name: str = "dataset",
    ) -> None:
        self.split_name = split_name
        self.max_len_a, self.max_len_b = int(max_len_a), int(max_len_b)
        self.npy_root = Path(npy_root)
        self.feat1d_root = Path(feat1d_root) if feat1d_root else Path(npy_root)
        self.dssr_root = Path(dssr_root) if dssr_root else None
        self.norm_mean = (np.array(norm_mean, dtype=np.float32).reshape(5, 1, 1)
                          if norm_mean is not None else None)
        self.norm_std = (np.array(norm_std, dtype=np.float32).reshape(5, 1, 1)
                         if norm_std is not None else None)

        entries = parse_pair_fasta(fasta_path)
        self._deN_index: Dict[str, Tuple[str, int, str]] = {}
        deN_collisions = 0
        for idx, (nm, sq) in enumerate(entries):
            k = _deN(nm)
            if k in self._deN_index:
                deN_collisions += 1
            self._deN_index[k] = (nm, idx, sq)
        if deN_collisions:
            print(f"[RRIDataset:{split_name}] WARNING: {deN_collisions} fasta names "
                  f"collide after N-stripping; some rows may map to the wrong RNA.")

        self._prefix = file_prefix if file_prefix is not None else self._infer_prefix(self.npy_root)
        print(f"[RRIDataset:{split_name}] map dir='{self.npy_root}' prefix='{self._prefix}'  "
              f"feat1d dir='{self.feat1d_root}'")

        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        a_name_col = cols.get("a_name") or cols.get("a_ame")
        b_name_col = cols.get("b_name") or cols.get("b_ame")
        if not (a_name_col and b_name_col and "a_seq" in df.columns
                and "b_seq" in df.columns and "label" in df.columns):
            raise ValueError(f"[{split_name}] unexpected CSV columns {list(df.columns)}")

        a_names = df[a_name_col].astype(str).tolist()
        b_names = df[b_name_col].astype(str).tolist()
        a_seqs = df["a_seq"].astype(str).tolist()
        b_seqs = df["b_seq"].astype(str).tolist()
        labels = df["label"].astype(float).tolist()

        self.records: List[Dict] = []
        self._paths: List[Dict[str, Optional[Path]]] = []
        miss_name = miss_file = 0
        n_repr_mismatch_a = n_repr_mismatch_b = 0
        first_bad = None

        for i in range(len(df)):
            ea = self._deN_index.get(_deN(a_names[i]))
            eb = self._deN_index.get(_deN(b_names[i]))
            if ea is None or eb is None:
                miss_name += 1
                if first_bad is None:
                    first_bad = f"name not in fasta: {a_names[i] if ea is None else b_names[i]!r}"
                continue
            pa = self._resolve(ea[0], ea[1])
            pb = self._resolve(eb[0], eb[1])
            if pa is None or pb is None:
                miss_file += 1
                if first_bad is None:
                    which = ea if pa is None else eb
                    first_bad = (f"no feature file for '{which[0]}' (idx {which[1]}); tried "
                                 + ", ".join(str(c.name) for c in self._map_candidates(which[0], which[1])[:6]))
                continue

            a_seq = _to_rna(a_seqs[i])
            b_seq = _to_rna(b_seqs[i])
            if a_seq != ea[2]:
                n_repr_mismatch_a += 1
            if b_seq != eb[2]:
                n_repr_mismatch_b += 1

            self.records.append(dict(a_seq=a_seq, b_seq=b_seq, label=labels[i], row_id=i,
                                     a_name=ea[0], b_name=eb[0]))
            self._paths.append({**{f"a_{k}": v for k, v in pa.items()},
                                **{f"b_{k}": v for k, v in pb.items()}})

        self.used_count = len(self.records)
        self.ignored_count = miss_name + miss_file
        print(f"[RRIDataset:{split_name}] used={self.used_count} "
              f"ignored={self.ignored_count} (name_not_in_fasta={miss_name}, no_feature_file={miss_file})")
        if self.used_count:
            print(f"[RRIDataset:{split_name}] rows whose sequence != fasta representative: "
                  f"miRNA {n_repr_mismatch_a}/{self.used_count}, mRNA {n_repr_mismatch_b}/{self.used_count} "
                  f"(these use a name-level structure with the row's own sequence)")
        if self.used_count == 0:
            raise ValueError(
                f"[{split_name}] 0 usable pairs. {first_bad}\n"
                f"  map dir   : {self.npy_root}  -> {self._listing(self.npy_root)}\n"
                f"  feat1d dir: {self.feat1d_root} -> {self._listing(self.feat1d_root)}\n"
                f"  pass --file-prefix or fix --*-npy-root / --*-feat1d-root."
            )

        self._seed_feats = [compute_seed_features(r["a_seq"], r["b_seq"]) for r in self.records]

    @staticmethod
    def _listing(root: Path, k: int = 12) -> List[str]:
        try:
            return sorted(p.name for p in root.iterdir())[:k]
        except Exception as exc:                                   # noqa: BLE001
            return [f"<cannot list: {exc}>"]

    @staticmethod
    def _infer_prefix(root: Path) -> str:
        """
        Detect a constant '<prefix>_' shared by (almost) every map file, whether
        the key is an integer index (train_0.npy) or an RNA name
        (task5_hsa-miR-1.npy). Returns '' when there is no shared prefix.
        """
        if not root.exists():
            return ""
        stems: List[str] = []
        for p in sorted(root.iterdir()):
            if p.suffix == ".npy":
                stems.append(p.stem)
            elif p.is_dir():
                stems.append(p.name)
            if len(stems) >= 500:
                break
        if not stems:
            return ""
        heads: Dict[str, int] = {}
        for s in stems:
            heads[s.split("_", 1)[0] + "_" if "_" in s else ""] = \
                heads.get(s.split("_", 1)[0] + "_" if "_" in s else "", 0) + 1
        best, n = max(heads.items(), key=lambda kv: kv[1])
        if best and n >= 0.9 * len(stems):
            return best[:-1]
        idx_heads: Dict[str, int] = {}
        for s in stems:
            if "_" in s and s.rsplit("_", 1)[1].isdigit():
                idx_heads[s.rsplit("_", 1)[0]] = idx_heads.get(s.rsplit("_", 1)[0], 0) + 1
        if idx_heads:
            b, c = max(idx_heads.items(), key=lambda kv: kv[1])
            if c >= 0.9 * len(stems):
                return b
        return ""

    def _keys(self, name: str, idx: int) -> List[str]:
        ks = [str(idx), f"{idx:06d}", f"{idx:05d}", f"{idx:04d}", name]
        seen = set()
        return [k for k in ks if not (k in seen or seen.add(k))]

    def _map_candidates(self, name: str, idx: int) -> List[Path]:
        out: List[Path] = []
        for k in self._keys(name, idx):
            if self._prefix:
                out.append(self.npy_root / f"{self._prefix}_{k}.npy")
            out += [self.npy_root / f"{k}.npy", self.npy_root / f"_{k}.npy",
                    self.npy_root / f"_={k}.npy", self.npy_root / k / f"{k}.npy"]
        return out

    def _feat1d_candidates(self, name: str, idx: int) -> List[Path]:
        out: List[Path] = []
        for k in self._keys(name, idx):
            folders = ([f"{self._prefix}_{k}"] if self._prefix else []) + [k, f"_{k}"]
            for fo in folders:
                out += [self.feat1d_root / fo / "feat1d_norm.npy",
                        self.feat1d_root / fo / "feat1d.npy"]
            out += [self.feat1d_root / f"{k}_feat1d_norm.npy",
                    self.feat1d_root / f"{k}_1d.npy",
                    self.feat1d_root / f"{k}.npy"]
        return out

    def _glob_unique(self, root: Path, key: str, suffix: str) -> Optional[Path]:
        if not root.is_dir() or len(key) < 4:
            return None
        g = sorted(root.glob(f"*{key}{suffix}"))
        return g[0] if len(g) == 1 else None

    def _resolve(self, name: str, idx: int) -> Optional[Dict[str, Optional[Path]]]:
        npy = next((c for c in self._map_candidates(name, idx) if c.exists()), None)
        if npy is None:
            npy = self._glob_unique(self.npy_root, name, ".npy")
        if npy is None:
            return None
        feat = next((c for c in self._feat1d_candidates(name, idx) if c.exists()), None)
        if feat is None:
            d = self.feat1d_root / name
            if d.is_dir():
                g = sorted(d.glob("*.npy"))
                feat = g[0] if len(g) == 1 else None
        if feat is None:
            feat = self._glob_unique(self.feat1d_root, name, "_1d.npy") \
                or self._glob_unique(self.feat1d_root, name, ".npy")
        if feat is None:
            return None
        dssr = None
        if self.dssr_root is not None:
            for k in self._keys(name, idx):
                for c in (self.dssr_root / k / "bp_mask.npy",
                          self.dssr_root / k / "bp.npy",
                          self.dssr_root / f"{k}_bp.npy",
                          self.dssr_root / f"{k}.npy"):
                    if c.exists():
                        dssr = c
                        break
                if dssr:
                    break
        return {"npy": npy, "feat": feat, "dssr": dssr}

    def _effective_len(self, seq: str, npy_path: Path, feat_path: Path, cap: int) -> int:
        fs = np.load(feat_path, mmap_mode="r").shape
        if len(fs) != 2 or int(fs[1]) < 23:
            raise ValueError(f"feat1d shape {fs}")
        return int(min(len(seq), _map_len(npy_path), int(fs[0]), cap))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]
        p = self._paths[idx]
        a_seq, b_seq = r["a_seq"], r["b_seq"]

        La = self._effective_len(a_seq, p["a_npy"], p["a_feat"], self.max_len_a)
        Lb = self._effective_len(b_seq, p["b_npy"], p["b_feat"], self.max_len_b)
        if La <= 0 or Lb <= 0:
            raise ValueError(f"[{self.split_name}] non-positive length row {r['row_id']}")

        x_a = self._build_x(p["a_npy"], p["a_dssr"], La)
        x_b = self._build_x(p["b_npy"], p["b_dssr"], Lb)
        feat_a = np.load(p["a_feat"]).astype(np.float32)[:La, :23]
        feat_b = np.load(p["b_feat"]).astype(np.float32)[:Lb, :23]

        return {
            "x_a": torch.from_numpy(np.ascontiguousarray(x_a)),
            "x_b": torch.from_numpy(np.ascontiguousarray(x_b)),
            "feat_a": torch.from_numpy(np.ascontiguousarray(feat_a)),
            "feat_b": torch.from_numpy(np.ascontiguousarray(feat_b)),
            "onehot_a": torch.from_numpy(_one_hot(a_seq, La)),
            "onehot_b": torch.from_numpy(_one_hot(b_seq, Lb)),
            "mask_a": torch.ones(La, dtype=torch.bool),
            "mask_b": torch.ones(Lb, dtype=torch.bool),
            "seed_feats": torch.from_numpy(self._seed_feats[idx]),
            "label": torch.tensor(float(r["label"]), dtype=torch.float32),
            "La": La, "Lb": Lb, "row_id": int(r["row_id"]),
        }

    def _build_x(self, npy_path: Path, dssr_path: Optional[Path], L: int) -> np.ndarray:
        x4 = _load_map_4ch(npy_path)[:, :L, :L]
        if dssr_path is not None and dssr_path.exists():
            bp = np.squeeze(np.load(dssr_path).astype(np.float32))
            if bp.ndim != 2:
                raise ValueError(f"DSSR must squeeze to 2D, got {bp.shape}")
            bp = bp[:L, :L]
        else:
            bp = np.zeros((L, L), dtype=np.float32)
        x = np.concatenate([x4, bp[None]], axis=0)
        if self.norm_mean is not None and self.norm_std is not None:
            x = (x - self.norm_mean) / (self.norm_std + 1e-8)
        return x

    def pos_fraction(self) -> float:
        if not self.records:
            return 0.5
        return float(sum(r["label"] for r in self.records) / len(self.records))


def rri_collate_fn(batch: List[Dict]) -> Dict:
    B = len(batch)
    La_max = max(b["La"] for b in batch)
    Lb_max = max(b["Lb"] for b in batch)
    x_a = torch.zeros(B, 5, La_max, La_max)
    x_b = torch.zeros(B, 5, Lb_max, Lb_max)
    feat_a = torch.zeros(B, La_max, 23)
    feat_b = torch.zeros(B, Lb_max, 23)
    oh_a = torch.zeros(B, La_max, 4)
    oh_b = torch.zeros(B, Lb_max, 4)
    ma = torch.zeros(B, La_max, dtype=torch.bool)
    mb = torch.zeros(B, Lb_max, dtype=torch.bool)
    seed = torch.zeros(B, batch[0]["seed_feats"].shape[0])
    labels = torch.zeros(B)
    La_list, Lb_list, row_ids = [], [], []
    for i, s in enumerate(batch):
        la, lb = s["La"], s["Lb"]
        x_a[i, :, :la, :la] = s["x_a"]
        x_b[i, :, :lb, :lb] = s["x_b"]
        feat_a[i, :la] = s["feat_a"]
        feat_b[i, :lb] = s["feat_b"]
        oh_a[i, :la] = s["onehot_a"]
        oh_b[i, :lb] = s["onehot_b"]
        ma[i, :la] = s["mask_a"]
        mb[i, :lb] = s["mask_b"]
        seed[i] = s["seed_feats"]
        labels[i] = s["label"]
        La_list.append(la); Lb_list.append(lb); row_ids.append(s["row_id"])
    return {
        "x_a": x_a, "x_b": x_b, "feat_a": feat_a, "feat_b": feat_b,
        "onehot_a": oh_a, "onehot_b": oh_b, "mask_a": ma, "mask_b": mb,
        "seed_feats": seed, "labels": labels,
        "La": La_list, "Lb": Lb_list, "row_ids": row_ids,
    }
