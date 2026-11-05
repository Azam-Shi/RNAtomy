"""RNA 3D contact-map dataset (RNAcontact benchmark; contact = C1'-C1' distance < 8 A).

Per RNA <id>: <npy_root>/<id>.npy (L x L x 4 trRosettaRNA map), <feat1d_root>/<id>/feat1d_norm.npy
(L x 23 descriptors) and <contact_root>/<id>.csv (L x L label). Samples are returned unpadded, so
batches must contain RNAs of equal length.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def load_contact_matrix(path: Path) -> np.ndarray:
    """Load an L x L contact matrix from a CSV with or without a header row."""
    with path.open("r", newline="") as f:
        first = next(csv.reader(f), None)
    if first is None:
        raise ValueError(f"empty contact CSV: {path}")
    has_header = not all(_is_number(v) for v in first if v != "")
    arr = np.loadtxt(
        path, delimiter=",", dtype=np.float32, skiprows=1 if has_header else 0
    )
    if arr.ndim != 2:
        raise ValueError(f"contact CSV {path} parsed to {arr.shape}, expected 2D")
    return arr


class ContactMapDataset(Dataset):
    """split_ids: RNA ids of the split. norm_mean / norm_std: per-channel pretraining statistics.
    use_dssr adds the DSSR base-pair mask as an extra input channel (off by default).
    """

    def __init__(
        self,
        split_ids: List[str],
        npy_root: str,
        dssr_root: str,
        feat1d_root: str,
        contact_root: Optional[str] = None,
        norm_mean: Optional[List[float]] = None,
        norm_std:  Optional[List[float]] = None,
        feat1d_file: str = "feat1d_norm.npy",
        feat1d_dim: int = 23,
        npy_file: str = "",
        npy_prefix: str = "",
        use_dssr: bool = False,
        dssr_feature: str = "bp_mask",
        dssr_required: bool = True,
        symmetrize_target: bool = True,
        use_feat1d: bool = True,
    ):
        self.npy_root     = Path(npy_root)
        self.dssr_root    = Path(dssr_root)
        self.feat1d_root  = Path(feat1d_root)
        self.contact_root = Path(contact_root) if contact_root else None
        self.inference_only = self.contact_root is None
        self.feat1d_file  = feat1d_file
        self.use_feat1d   = use_feat1d
        self.npy_file     = npy_file
        self.npy_prefix   = npy_prefix
        self.feat1d_dim   = feat1d_dim
        self.use_dssr     = use_dssr
        self.dssr_feature = dssr_feature
        self.dssr_required = dssr_required
        self.symmetrize_target = symmetrize_target

        if norm_mean is not None and norm_std is not None:
            self.norm_mean = torch.tensor(norm_mean, dtype=torch.float32).view(-1, 1, 1)
            self.norm_std  = torch.tensor(norm_std,  dtype=torch.float32).view(-1, 1, 1)
        else:
            self.norm_mean = None
            self.norm_std  = None

        self.samples: List[str] = []
        self.seq_lens: List[int] = []
        missing = []
        inconsistent = []
        for rna_id in split_ids:
            ok = True
            if self._resolve_npy_path(rna_id) is None:
                missing.append(f"npy:{rna_id}")
                ok = False
            if self.use_dssr:
                if not (self.dssr_root / rna_id / f"{self.dssr_feature}.npy").exists():
                    missing.append(f"dssr:{rna_id}")
                    if self.dssr_required:
                        ok = False
            if self.use_feat1d and not (self.feat1d_root / rna_id / feat1d_file).exists():
                missing.append(f"feat1d:{rna_id}")
                ok = False
            if not self.inference_only and not (self.contact_root / f"{rna_id}.csv").exists():
                missing.append(f"contact:{rna_id}")
                ok = False
            if ok:
                try:
                    seq_len = self._infer_seq_len(rna_id)
                except Exception as exc:
                    missing.append(f"shape:{rna_id} ({exc})")
                    continue

                inconsistency = self._check_inconsistent_lengths(rna_id, seq_len)
                if inconsistency is not None:
                    inconsistent.append(inconsistency)
                    continue

                self.samples.append(rna_id)
                self.seq_lens.append(seq_len)

        if missing:
            print(f"[ContactMapDataset] Skipped {len(missing)} missing files "
                  f"(first 10): {missing[:10]}")
        if inconsistent:
            print(f"[ContactMapDataset] Filtered {len(inconsistent)} inconsistent IDs "
                  f"(first 10): {inconsistent[:10]}")
        print(f"[ContactMapDataset] {len(self.samples)} usable samples loaded.")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_npy_path(self, rna_id: str) -> Optional[Path]:
        """Locate the map of one RNA: <root>/<id>.npy, <root>/_<id>.npy, <root>/<id>/<id>.npy or a unique match."""
        candidates = [
            self.npy_root / f"{rna_id}.npy",
            self.npy_root / f"_{rna_id}.npy",
            self.npy_root / f"_={rna_id}.npy",
        ]
        if self.npy_prefix:
            candidates.append(self.npy_root / f"{self.npy_prefix}{rna_id}.npy")
        if self.npy_file:
            candidates.append(self.npy_root / rna_id / self.npy_file)
        candidates.append(self.npy_root / rna_id / f"{rna_id}.npy")
        for p in candidates:
            if p.exists():
                return p
        flat = sorted(self.npy_root.glob(f"*{rna_id}.npy"))
        if len(flat) == 1:
            return flat[0]
        sub = self.npy_root / rna_id
        if sub.is_dir():
            npys = sorted(sub.glob("*.npy"))
            if len(npys) == 1:
                return npys[0]
        return None

    def _infer_seq_len(self, rna_id: str) -> int:
        """Infer the consensus RNA length used by __getitem__ without full file loads."""
        npy_shape = np.load(self._resolve_npy_path(rna_id), mmap_mode="r").shape
        if len(npy_shape) == 3:
            dims = sorted(int(d) for d in npy_shape)
            L = int(dims[1])
        else:
            L = min(int(npy_shape[0]), int(npy_shape[1]))

        if self.use_dssr:
            dssr_path = self.dssr_root / rna_id / f"{self.dssr_feature}.npy"
            if dssr_path.exists():
                dssr_shape = np.load(dssr_path, mmap_mode="r").shape
                L = min(L, int(dssr_shape[0]), int(dssr_shape[1]))

        if L <= 0:
            raise ValueError("inferred non-positive sequence length")
        return L

    def _contact_csv_shape(self, rna_id: str) -> tuple[int, int]:
        """Return (rows, cols) of the contact-map CSV without loading a dense matrix."""
        csv_path = self.contact_root / f"{rna_id}.csv"
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            first = next(reader, None)
            if first is None:
                return 0, 0
            n_cols = len(first)
            first_is_data = all(_is_number(v) for v in first if v != "")
            n_rows = (1 if first_is_data else 0) + sum(1 for row in reader if row)
        return n_rows, n_cols

    def _check_inconsistent_lengths(self, rna_id: str, inferred_len: int) -> Optional[str]:
        """Return a reason string when a sample is inconsistent, otherwise None."""
        if self.use_feat1d:
            feat1d_shape = np.load(
                self.feat1d_root / rna_id / self.feat1d_file,
                mmap_mode="r",
            ).shape
            if len(feat1d_shape) != 2:
                return (
                    f"{rna_id}: invalid feat1d shape={feat1d_shape} "
                    f"(expected [L, {self.feat1d_dim}])"
                )
            if int(feat1d_shape[0]) < inferred_len:
                return (
                    f"{rna_id}: feat1d_len={int(feat1d_shape[0])} "
                    f"< inferred_L={inferred_len}"
                )

        if not self.inference_only:
            contact_rows, contact_cols = self._contact_csv_shape(rna_id)
            if contact_rows < inferred_len or contact_cols < inferred_len:
                return (
                    f"{rna_id}: contact_shape=({contact_rows},{contact_cols}) "
                    f"< inferred_L={inferred_len}"
                )

        return None

    def get_seq_len(self, idx: int) -> int:
        """Return the precomputed sequence length for a sample index."""
        return self.seq_lens[idx]

    def __getitem__(self, idx: int):
        rna_id = self.samples[idx]
        L = self.seq_lens[idx]

        arr = np.load(self._resolve_npy_path(rna_id)).astype(np.float32)
        if arr.ndim != 3:
            raise ValueError(f"expected 3D trRosetta map for {rna_id}, got {arr.shape}")
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[-1]:
            arr = np.moveaxis(arr, 0, -1)
        arr = arr[:L, :L, :]

        if self.use_dssr:
            dssr_path = self.dssr_root / rna_id / f"{self.dssr_feature}.npy"
            if dssr_path.exists():
                dssr = np.load(dssr_path).astype(np.float32)
                dssr = dssr[:L, :L]
            else:
                dssr = np.zeros((L, L), dtype=np.float32)
            x = np.concatenate([arr, dssr[..., None]], axis=-1)
        else:
            x = arr

        x = torch.tensor(x).permute(2, 0, 1)
        if self.norm_mean is not None:
            x = (x - self.norm_mean) / self.norm_std

        if self.use_feat1d:
            feat1d = np.load(
                self.feat1d_root / rna_id / self.feat1d_file
            ).astype(np.float32)
            if feat1d.shape[0] < L:
                raise ValueError(
                    f"feat1d shorter than inferred L for {rna_id}: "
                    f"feat1d={feat1d.shape[0]} < L={L}"
                )
            feat1d = feat1d[:L]
        else:
            feat1d = np.zeros((L, self.feat1d_dim), dtype=np.float32)
        nt_features  = torch.tensor(feat1d)
        nt_valid_mask = torch.ones(L, dtype=torch.bool)

        if self.inference_only:
            target = torch.zeros(L, L, dtype=torch.float32)
        else:
            contact_arr = load_contact_matrix(self.contact_root / f"{rna_id}.csv")
            if contact_arr.shape[0] < L or contact_arr.shape[1] < L:
                raise ValueError(
                    f"contact map smaller than inferred L for {rna_id}: "
                    f"contact={contact_arr.shape}, L={L}"
                )
            contact_arr = np.ascontiguousarray(contact_arr[:L, :L])
            target = torch.from_numpy(contact_arr).float()
            if self.symmetrize_target:
                target = torch.maximum(target, target.transpose(0, 1))

        valid_mask = torch.ones(L, L, dtype=torch.bool)

        return x, target, valid_mask, nt_features, nt_valid_mask, L


def contact_map_collate_fn(batch):
    """
    Custom collate for same-length batches.
    The same-length batch sampler guarantees stackable shapes.
    """
    xs, targets, valid_masks, nt_feats, nt_valids, seq_lens = zip(*batch)
    return (
        torch.stack(xs, 0),
        torch.stack(targets, 0),
        torch.stack(valid_masks, 0),
        torch.stack(nt_feats, 0),
        torch.stack(nt_valids, 0),
        list(seq_lens),
    )


class SameLengthBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups indices by sequence length and yields
    same-length mini-batches.

    This avoids cross-sample padding for variable-L contact maps.
    """

    def __init__(
        self,
        dataset: ContactMapDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        buckets: Dict[int, List[int]] = {}
        for idx, L in enumerate(dataset.seq_lens):
            buckets.setdefault(int(L), []).append(idx)
        self.length_buckets = buckets

    def __iter__(self) -> Iterator[List[int]]:
        all_batches: List[List[int]] = []

        for indices in self.length_buckets.values():
            local_indices = indices.copy()
            if self.shuffle:
                random.shuffle(local_indices)

            n = len(local_indices)
            if self.drop_last:
                n = (n // self.batch_size) * self.batch_size

            for start in range(0, n, self.batch_size):
                batch = local_indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.length_buckets.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total

    def bucket_counts(self) -> Dict[int, int]:
        """Returns {sequence_length: number_of_samples} for logging/debug."""
        return {L: len(indices) for L, indices in self.length_buckets.items()}


def load_split_ids(split_file: str) -> List[str]:
    """
    Load RNA IDs from a split file.

    Supported formats:
      1) one ID per line
      2) FASTA-like headers: ">rna_id" or ">rna_id,SEQUENCE"
      3) CSV-style lines: "rna_id,SEQUENCE"
    """
    ids: List[str] = []
    saw_fasta_header = False

    with open(split_file) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith(">"):
                saw_fasta_header = True
                header = line[1:].strip()
                if not header:
                    continue
                rna_id = header.split(",", 1)[0].split()[0]
                if rna_id:
                    ids.append(rna_id)
                continue

            if saw_fasta_header:
                continue

            rna_id = line.split(",", 1)[0].split()[0]
            if rna_id:
                ids.append(rna_id)

    return ids
