"""5'UTR mean ribosome load dataset (Optimus 5-Prime polysome profiling; random and human UTR test sets).

Maps and descriptors are named after the row position of the sequence in its split CSV:
<npy_root>/<prefix>_<row>.npy and <feat1d_root>/<prefix>_<row>/feat1d_norm.npy (the prefix is detected
automatically). Besides the structural tensors, each sample carries the one-hot sequence and six engineered
upstream-AUG features [has_uAUG, uAUG count, in-frame uAUG count, first uAUG position, GC fraction, length],
the standardized target and the read count used as a loss weight.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Sampler

MAX_LEN = 100

_NUC = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def _one_hot(seq: str, L: int) -> np.ndarray:
    oh = np.zeros((L, 4), dtype=np.float32)
    for i, c in enumerate(seq[:L].upper()):
        j = _NUC.get(c)
        if j is not None:
            oh[i, j] = 1.0
    return oh


def _uorf_aux(seq: str, L: int) -> np.ndarray:
    """
    Engineered features for the dominant MRL determinants. The main ORF starts
    immediately 3' of the UTR (position L), so a uAUG at position i is in the
    ORF reading frame iff (L - i) % 3 == 0.
    """
    s = seq[:L].upper().replace("U", "T")
    n = max(len(s), 1)
    aug = [i for i in range(len(s) - 2) if s[i:i + 3] == "ATG"]
    n_inframe = sum(1 for i in aug if (n - i) % 3 == 0)
    first_frac = (aug[0] / max(n - 1, 1)) if aug else 1.0
    gc = (s.count("G") + s.count("C")) / n
    len_norm = (n - 25) / 75.0
    return np.array(
        [
            1.0 if aug else 0.0,
            min(len(aug), 5) / 5.0,
            min(n_inframe, 3) / 3.0,
            first_frac,
            gc,
            len_norm,
        ],
        dtype=np.float32,
    )


AUX_DIM = 6


class UTRMRLDataset(Dataset):
    """csv_path: split CSV. npy_root / feat1d_root: feature folders. scaler: fitted target scaler (fit on this split
    when None). min_reads: drop rows below this read count. id_column: column kept only as a label in the outputs.
    """

    def __init__(
        self,
        csv_path: str,
        npy_root: str,
        feat1d_root: str,
        dssr_root: Optional[str] = None,
        scaler: Optional[StandardScaler] = None,
        min_reads: int = 0,
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
        split_name: str = "dataset",
        id_column: str = "Unnamed: 0",
        seq_column: str = "utr",
        target_column: str = "rl",
        read_column: str = "total_reads",
        file_prefix: Optional[str] = None,
    ) -> None:
        df = pd.read_csv(csv_path)
        self.split_name = split_name

        if seq_column not in df.columns:
            raise ValueError(f"[{split_name}] CSV has no '{seq_column}' column; got {list(df.columns)}")
        if target_column not in df.columns:
            raise ValueError(f"[{split_name}] CSV has no '{target_column}' column")

        has_reads = read_column in df.columns
        if min_reads > 0 and has_reads:
            before = len(df)
            df = df[df[read_column] >= min_reads].reset_index(drop=True)
            print(f"[UTRMRLDataset:{split_name}] min_reads={min_reads}: kept {len(df)}/{before} rows")

        if bool(id_column) and id_column in df.columns:
            label_vals = df[id_column].astype(int).tolist()
        else:
            label_vals = list(range(len(df)))

        self.npy_root = Path(npy_root)
        self.feat1d_root = Path(feat1d_root)
        self.dssr_root = Path(dssr_root) if dssr_root else None

        self._npy_prefix = file_prefix if file_prefix is not None else self._infer_prefix(self.npy_root, want_dirs=False)
        self._folder_prefix = file_prefix if file_prefix is not None else self._infer_prefix(self.feat1d_root, want_dirs=True)
        print(f"[UTRMRLDataset:{split_name}] npy prefix='{self._npy_prefix}'  "
              f"feat1d folder prefix='{self._folder_prefix}'")

        if self.dssr_root is not None and not self._dssr_root_has_any_masks():
            print(f"[UTRMRLDataset:{split_name}] DSSR root has no masks; using zero DSSR channel.")
            self.dssr_root = None

        self.sample_ids: List[int] = []
        self.seq_lens: List[int] = []
        self.sequences: List[str] = []
        self.file_paths: List[Dict[str, Optional[Path]]] = []
        self.read_weights: List[float] = []
        kept_targets: List[float] = []

        miss_npy = miss_feat1d = miss_dssr = bad_shape = 0
        first_miss_example: Optional[str] = None
        seq_vals = df[seq_column].astype(str).tolist()
        tgt_vals = df[target_column].astype(float).tolist()
        read_vals = df[read_column].astype(float).tolist() if has_reads else [1.0] * len(df)

        for pos in range(len(df)):
            npy_path = self._resolve_npy_path(pos)
            if npy_path is None:
                miss_npy += 1
                if first_miss_example is None:
                    first_miss_example = "npy row %d tried: %s" % (
                        pos, ", ".join(str(c) for c in self._npy_candidates(pos)[:6]))
                continue
            feat1d_path = self._resolve_feat1d_path(pos)
            if feat1d_path is None:
                miss_feat1d += 1
                if first_miss_example is None:
                    first_miss_example = "feat1d row %d tried: %s" % (
                        pos, ", ".join(str(c) for c in self._feat1d_candidates(pos)[:6]))
                continue
            dssr_path = self._resolve_dssr_path(pos)
            if self.dssr_root is not None and dssr_path is None:
                miss_dssr += 1
                continue

            seq = seq_vals[pos]
            try:
                L = self._consensus_length(seq, npy_path, feat1d_path, dssr_path)
            except Exception:
                bad_shape += 1
                continue
            if L <= 0:
                bad_shape += 1
                continue

            self.sample_ids.append(int(label_vals[pos]))
            self.seq_lens.append(L)
            self.sequences.append(seq)
            self.file_paths.append({"npy": npy_path, "feat1d": feat1d_path, "dssr": dssr_path})
            kept_targets.append(tgt_vals[pos])
            self.read_weights.append(read_vals[pos])

        self.ignored_count = miss_npy + miss_feat1d + miss_dssr + bad_shape
        self.used_count = len(self.sequences)
        print(f"[UTRMRLDataset:{split_name}] used={self.used_count} ignored={self.ignored_count} "
              f"(npy={miss_npy}, feat1d={miss_feat1d}, dssr={miss_dssr}, shape={bad_shape})")
        if self.used_count == 0:
            npy_sample = self._sample_listing(self.npy_root)
            feat_sample = self._sample_listing(self.feat1d_root)
            raise ValueError(
                f"No usable samples in split '{split_name}'.\n"
                f"  npy_root   = {self.npy_root}\n"
                f"  feat1d_root= {self.feat1d_root}\n"
                f"  npy_root contains   : {npy_sample}\n"
                f"  feat1d_root contains: {feat_sample}\n"
                f"  {first_miss_example}\n"
                f"  -> pass --file-prefix if the prefix was mis-detected, or fix the roots."
            )

        t = np.asarray(kept_targets, dtype=np.float32).reshape(-1, 1)
        if scaler is None:
            scaler = StandardScaler().fit(t)
        self.scaler = scaler
        self.rl_norm = scaler.transform(t).astype(np.float32).squeeze(1)

        self.norm_mean = (
            np.array(norm_mean, dtype=np.float32).reshape(-1, 1, 1) if norm_mean is not None else None
        )
        self.norm_std = (
            np.array(norm_std, dtype=np.float32).reshape(-1, 1, 1) if norm_std is not None else None
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def get_seq_len(self, idx: int) -> int:
        return self.seq_lens[idx]

    @staticmethod
    def _sample_listing(root: Path, k: int = 12) -> List[str]:
        try:
            return sorted(p.name for p in root.iterdir())[:k]
        except Exception as exc:                                   # noqa: BLE001
            return [f"<cannot list: {exc}>"]

    @staticmethod
    def _infer_prefix(root: Path, want_dirs: bool) -> str:
        """
        Detect the common '<prefix>_<int>' stem shared by feature files/folders.
        Returns '' when names are bare integers (or nothing can be inferred).
        """
        if not root.exists():
            return ""
        counts: Dict[str, int] = {}
        seen = 0
        for p in sorted(root.iterdir()):
            if want_dirs and not p.is_dir():
                continue
            if not want_dirs and p.suffix != ".npy":
                continue
            stem = p.name if want_dirs else p.stem
            stem = stem[:-3] if (not want_dirs and stem.endswith("_bp")) else stem
            if "_" in stem:
                head, tail = stem.rsplit("_", 1)
                if tail.isdigit():
                    counts[head] = counts.get(head, 0) + 1
            elif stem.isdigit():
                counts[""] = counts.get("", 0) + 1
            seen += 1
            if seen >= 400:
                break
        if not counts:
            return ""
        return max(counts, key=counts.get)

    def _id_tokens(self, row_pos: int) -> List[str]:
        return [str(row_pos), f"{row_pos:06d}", f"{row_pos:05d}", f"{row_pos:04d}"]

    def _npy_candidates(self, row_pos: int) -> List[Path]:
        toks = self._id_tokens(row_pos)
        names: List[str] = []
        for t in toks:
            if self._npy_prefix:
                names += [f"{self._npy_prefix}_{t}.npy"]
            names += [f"{t}.npy", f"_{t}.npy", f"_={t}.npy"]
        seen = set()
        return [self.npy_root / n for n in names if not (n in seen or seen.add(n))]

    def _feat1d_candidates(self, row_pos: int) -> List[Path]:
        toks = self._id_tokens(row_pos)
        out: List[Path] = []
        for t in toks:
            folders = ([f"{self._folder_prefix}_{t}"] if self._folder_prefix else []) + [t, f"_{t}", f"_={t}"]
            for fo in folders:
                out += [
                    self.feat1d_root / fo / "feat1d_norm.npy",
                    self.feat1d_root / fo / "feat1d.npy",
                ]
            if self._folder_prefix:
                out.append(self.feat1d_root / f"{self._folder_prefix}_{t}_feat1d_norm.npy")
            out.append(self.feat1d_root / f"{t}_feat1d_norm.npy")
        return out

    def _dssr_candidates(self, row_pos: int) -> List[Path]:
        toks = self._id_tokens(row_pos)
        out: List[Path] = []
        for t in toks:
            stems = ([f"{self._folder_prefix}_{t}"] if self._folder_prefix else []) + [t, f"_{t}", f"_={t}"]
            for st in stems:
                out += [
                    self.dssr_root / f"{st}_bp.npy",
                    self.dssr_root / f"{st}.npy",
                    self.dssr_root / st / "bp_mask.npy",
                    self.dssr_root / st / "bp.npy",
                ]
        return out

    def _resolve_npy_path(self, row_pos: int) -> Optional[Path]:
        for p in self._npy_candidates(row_pos):
            if p.exists():
                return p
        return None

    def _resolve_feat1d_path(self, row_pos: int) -> Optional[Path]:
        for p in self._feat1d_candidates(row_pos):
            if p.exists():
                return p
        return None

    def _resolve_dssr_path(self, row_pos: int) -> Optional[Path]:
        if self.dssr_root is None:
            return None
        for p in self._dssr_candidates(row_pos):
            if p.exists():
                return p
        return None

    def _dssr_root_has_any_masks(self) -> bool:
        if self.dssr_root is None or not self.dssr_root.exists():
            return False
        if next(self.dssr_root.glob("*_bp.npy"), None) is not None:
            return True
        try:
            for d in self.dssr_root.iterdir():
                if d.is_dir() and ((d / "bp_mask.npy").exists() or (d / "bp.npy").exists()):
                    return True
        except Exception:                                          # noqa: BLE001
            return False
        return False

    @staticmethod
    def _map_len(path: Path) -> int:
        s = np.load(path, mmap_mode="r").shape
        if len(s) != 3:
            raise ValueError(f"map ndim={len(s)}")
        if s[0] in (4, 5):
            return min(int(s[1]), int(s[2]))
        if s[-1] in (4, 5):
            return min(int(s[0]), int(s[1]))
        raise ValueError(f"map has no channel axis: {s}")

    @staticmethod
    def _feat1d_len(path: Path) -> int:
        s = np.load(path, mmap_mode="r").shape
        if len(s) != 2 or int(s[1]) < 23:
            raise ValueError(f"feat1d shape {s}")
        return int(s[0])

    @staticmethod
    def _dssr_len(path: Path) -> int:
        s = np.load(path, mmap_mode="r").shape
        if len(s) == 2:
            return min(int(s[0]), int(s[1]))
        if len(s) == 3 and (int(s[0]) == 1 or int(s[-1]) == 1):
            a = [d for d in s if d != 1]
            return min(int(a[0]), int(a[1]))
        raise ValueError(f"dssr shape {s}")

    def _consensus_length(self, seq, npy_path, feat1d_path, dssr_path) -> int:
        L = min(len(seq), self._map_len(npy_path), self._feat1d_len(feat1d_path), MAX_LEN)
        if dssr_path is not None:
            L = min(L, self._dssr_len(dssr_path))
        return int(L)

    @staticmethod
    def _load_map(path: Path) -> np.ndarray:
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 3:
            raise ValueError(f"map ndim={arr.ndim}")
        if arr.shape[0] in (4, 5):
            return arr[:4]
        if arr.shape[-1] in (4, 5):
            return np.transpose(arr, (2, 0, 1))[:4]
        raise ValueError(f"map has no channel axis: {arr.shape}")

    def __getitem__(self, idx: int) -> Tuple:
        sid = self.sample_ids[idx]
        p = self.file_paths[idx]
        L = self.seq_lens[idx]
        seq = self.sequences[idx]

        x4 = self._load_map(p["npy"])[:, :L, :L]
        if self.dssr_root is not None and p["dssr"] is not None:
            bp = np.load(p["dssr"]).astype(np.float32)
            bp = np.squeeze(bp)
            if bp.ndim != 2:
                raise ValueError(f"DSSR array must squeeze to 2D, got {bp.shape}")
            bp = bp[:L, :L]
        else:
            bp = np.zeros((L, L), dtype=np.float32)
        x = np.concatenate([x4, bp[None]], axis=0)
        if self.norm_mean is not None and self.norm_std is not None:
            x = (x - self.norm_mean) / (self.norm_std + 1e-8)

        feat1d = np.load(p["feat1d"]).astype(np.float32)
        if feat1d.ndim != 2 or feat1d.shape[1] < 23:
            raise ValueError(f"feat1d shape {feat1d.shape}")
        nt_features = feat1d[:L, :23]

        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.tensor(self.rl_norm[idx], dtype=torch.float32),
            torch.from_numpy(np.ascontiguousarray(nt_features)),
            torch.ones(L, dtype=torch.bool),
            torch.from_numpy(_one_hot(seq, L)),
            torch.from_numpy(_uorf_aux(seq, L)),
            torch.tensor(self.read_weights[idx], dtype=torch.float32),
            L,
            sid,
        )


class UTRSameLengthBatchSampler(Sampler[List[int]]):
    """Groups indices by exact sequence length and yields same-length batches."""

    def __init__(self, dataset: UTRMRLDataset, batch_size: int,
                 shuffle: bool = True, drop_last: bool = False, seed: int = 0) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        buckets: Dict[int, List[int]] = {}
        for idx, L in enumerate(dataset.seq_lens):
            buckets.setdefault(int(L), []).append(idx)
        self.length_buckets = buckets

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        all_batches: List[List[int]] = []
        for indices in self.length_buckets.values():
            local = indices.copy()
            if self.shuffle:
                rng.shuffle(local)
            n = len(local)
            if self.drop_last:
                n = (n // self.batch_size) * self.batch_size
            for s in range(0, n, self.batch_size):
                b = local[s:s + self.batch_size]
                if len(b) == self.batch_size or (b and not self.drop_last):
                    all_batches.append(b)
        if self.shuffle:
            rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for indices in self.length_buckets.values():
            n = len(indices)
            total += n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
        return total

    def bucket_counts(self) -> Dict[int, int]:
        return {L: len(v) for L, v in self.length_buckets.items()}
