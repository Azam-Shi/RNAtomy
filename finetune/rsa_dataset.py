"""Solvent-accessibility dataset (RNAsnap2 benchmark: TR95 / VL24 / TS44 chains).

Each chain takes three lines in the data file: >chain_id, the sequence, and space-separated ASA values in A^2
(or "---" to skip the chain). The target is RSA = ASA / max ASA with max ASA = 400 (A, G) and 350 (C, U).

Per chain: <npy_root>/<chain_id>.npy (trRosettaRNA map) and <feat1d_root>/<chain_id>/feat1d_norm.npy (L x 23).
Long chains are split into overlapping windows; loss_mask marks the positions each window is trained on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_MAX_ASA: dict = {"A": 400.0, "G": 400.0, "U": 350.0, "C": 350.0}


def _compute_rsa(seq: str, asa_values: List[float]) -> np.ndarray:
    """Continuous per-nucleotide RSA = ASA / max_ASA (no thresholding)."""
    rsa = np.zeros(len(seq), dtype=np.float32)
    for i, (nt, asa) in enumerate(zip(seq, asa_values)):
        rsa[i] = asa / _MAX_ASA.get(nt.upper(), 400.0)
    return rsa


def _max_asa_vector(seq: str) -> np.ndarray:
    """Per-nucleotide max ASA (Å²) -- used to recover ASA from a predicted RSA."""
    return np.array([_MAX_ASA.get(nt.upper(), 400.0) for nt in seq], dtype=np.float32)


def _parse_data_file(path: str) -> List[Tuple[str, str, List[float]]]:
    """
    Parse a TR95/VL24/TS45 data file.

    Returns a list of (chain_id, sequence, asa_values).
    Chains whose ASA line is "---" are silently skipped.
    """
    records: List[Tuple[str, str, List[float]]] = []
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        chain_id = lines[i][1:].strip()
        seq      = lines[i + 1].strip()
        asa_line = lines[i + 2].strip()
        i += 3

        if asa_line == "---":
            continue

        asa_values = [float(v) for v in asa_line.split()]
        if len(asa_values) != len(seq):
            continue

        records.append((chain_id, seq, asa_values))

    return records


class RSADataset(Dataset):
    """data_path: split file. npy_root / feat1d_root: feature folders. norm_mean / norm_std: per-channel statistics."""

    def __init__(
        self,
        data_path: str,
        npy_root: str,
        feat1d_root: str,
        dssr_root: Optional[str] = None,
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
        feat1d_file: str = "feat1d_norm.npy",
        split_name: str = "split",
        max_seq_len: Optional[int] = None,
        chunk_overlap: int = 256,
    ) -> None:
        if max_seq_len is not None and max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0 when provided")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if max_seq_len is not None and chunk_overlap >= max_seq_len:
            raise ValueError("chunk_overlap must be smaller than max_seq_len")

        raw_records = _parse_data_file(data_path)
        self.npy_root    = Path(npy_root)
        self.feat1d_root = Path(feat1d_root)
        self.dssr_root   = Path(dssr_root) if dssr_root else None
        self.feat1d_file = feat1d_file
        self.split_name  = split_name
        self.max_seq_len = max_seq_len
        self.chunk_overlap = chunk_overlap

        self.total_records = len(raw_records)
        self.records: List[Tuple[str, str, List[float]]] = []
        self.ignored_records: List[Tuple[str, str]] = []
        self.ignore_reason_counts: Dict[str, int] = {}

        for rec in raw_records:
            chain_id = rec[0]
            reason = self._incomplete_reason(rec)
            if reason is None:
                self.records.append(rec)
            else:
                self.ignored_records.append((chain_id, reason))
                self.ignore_reason_counts[reason] = self.ignore_reason_counts.get(reason, 0) + 1

        self.ignored_count = len(self.ignored_records)
        self.usable_chain_count = len(self.records)

        self.samples: List[Tuple[int, int, int, int, int]] = []
        self.chunked_records: List[Tuple[str, int, int]] = []
        for record_idx, (chain_id, seq, _asa_values) in enumerate(self.records):
            windows = self._make_windows(len(seq))
            if len(windows) > 1:
                self.chunked_records.append((chain_id, len(seq), len(windows)))
            for start, end, target_start, target_end in windows:
                self.samples.append(
                    (record_idx, start, end, target_start, target_end)
                )

        self.chunked_chain_count = len(self.chunked_records)
        self.chunk_count = sum(item[2] for item in self.chunked_records)

        self.norm_mean = (
            np.array(norm_mean, dtype=np.float32).reshape(5, 1, 1)
            if norm_mean is not None else None
        )
        self.norm_std = (
            np.array(norm_std, dtype=np.float32).reshape(5, 1, 1)
            if norm_std is not None else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _make_windows(self, length: int) -> List[Tuple[int, int, int, int]]:
        """Create context windows and a disjoint target region for each window."""
        if self.max_seq_len is None or length <= self.max_seq_len:
            return [(0, length, 0, length)]

        stride = self.max_seq_len - self.chunk_overlap
        last_start = length - self.max_seq_len
        starts = list(range(0, last_start + 1, stride))
        if starts[-1] != last_start:
            starts.append(last_start)

        context_windows = [
            (start, min(start + self.max_seq_len, length))
            for start in starts
        ]
        target_boundaries = [0]
        for (_prev_start, prev_end), (next_start, _next_end) in zip(
            context_windows, context_windows[1:]
        ):
            target_boundaries.append((prev_end + next_start) // 2)
        target_boundaries.append(length)

        return [
            (start, end, target_boundaries[i], target_boundaries[i + 1])
            for i, (start, end) in enumerate(context_windows)
        ]

    def _resolve_feat1d_path(
        self,
        chain_id: str,
        *,
        raise_if_missing: bool = True,
    ) -> Optional[Path]:
        """Path of a chain's descriptors: <root>/<id>/<file>, falling back to <root>/<id>_1d.npy."""
        candidates = [
            self.feat1d_root / chain_id / self.feat1d_file,
            self.feat1d_root / f"{chain_id}_1d.npy",
        ]
        for path in candidates:
            if path.exists():
                return path

        if not raise_if_missing:
            return None

        raise FileNotFoundError(
            "Missing 1D feature file for chain "
            f"{chain_id}. Tried: {candidates[0]} and {candidates[1]}"
        )

    def _incomplete_reason(self, record: Tuple[str, str, List[float]]) -> Optional[str]:
        chain_id, seq, _asa_values = record
        L = len(seq)

        npy_path = self.npy_root / f"{chain_id}.npy"
        if not npy_path.exists():
            return "missing_npy"

        try:
            npy_shape = np.load(npy_path, mmap_mode="r").shape
        except Exception:
            return "unreadable_npy"

        if len(npy_shape) != 3:
            return "invalid_npy_shape"

        if int(npy_shape[0]) == 4:
            h, w = int(npy_shape[1]), int(npy_shape[2])
        elif int(npy_shape[2]) == 4:
            h, w = int(npy_shape[0]), int(npy_shape[1])
        else:
            return "invalid_npy_shape"

        if h < L or w < L:
            return "npy_too_small"
        if h > L or w > L:
            return "npy_too_large"

        feat1d_path = self._resolve_feat1d_path(chain_id, raise_if_missing=False)
        if feat1d_path is None:
            return "missing_feat1d"

        try:
            feat_shape = np.load(feat1d_path, mmap_mode="r").shape
        except Exception:
            return "unreadable_feat1d"

        if len(feat_shape) != 2:
            return "invalid_feat1d_shape"
        if int(feat_shape[0]) < L:
            return "feat1d_too_short"
        if int(feat_shape[0]) > L:
            return "feat1d_too_long"
        if int(feat_shape[1]) != 23:
            return "feat1d_bad_dim"

        if self.dssr_root is not None:
            dssr_path = self.dssr_root / f"{chain_id}_bp.npy"
            if not dssr_path.exists():
                return "missing_dssr"
            try:
                dssr_shape = np.load(dssr_path, mmap_mode="r").shape
            except Exception:
                return "unreadable_dssr"
            if len(dssr_shape) != 2:
                return "invalid_dssr_shape"
            if int(dssr_shape[0]) < L or int(dssr_shape[1]) < L:
                return "dssr_too_small"
            if int(dssr_shape[0]) > L or int(dssr_shape[1]) > L:
                return "dssr_too_large"

        return None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        record_idx, start, end, target_start, target_end = self.samples[idx]
        chain_id, full_seq, full_asa_values = self.records[record_idx]
        seq = full_seq[start:end]
        asa_values = full_asa_values[start:end]
        L = end - start

        rsa = _compute_rsa(seq, asa_values)
        max_asa = _max_asa_vector(seq)

        x4_full = np.load(self.npy_root / f"{chain_id}.npy", mmap_mode="r")
        if x4_full.shape[0] == 4:
            x4 = np.asarray(
                x4_full[:, start:end, start:end], dtype=np.float32
            )
        else:
            x4 = np.asarray(
                x4_full[start:end, start:end, :], dtype=np.float32
            ).transpose(2, 0, 1)

        if self.dssr_root is not None:
            bp_full = np.load(
                self.dssr_root / f"{chain_id}_bp.npy", mmap_mode="r"
            )
            bp = np.asarray(
                bp_full[start:end, start:end], dtype=np.float32
            )
            x = np.concatenate([x4, bp[None]], axis=0)
        else:
            x = np.concatenate(
                [x4, np.zeros((1, L, L), dtype=np.float32)], axis=0
            )

        nt_features_full = np.load(
            self._resolve_feat1d_path(chain_id), mmap_mode="r"
        )
        nt_features = np.asarray(
            nt_features_full[start:end], dtype=np.float32
        )

        if self.norm_mean is not None and self.norm_std is not None:
            x = (x - self.norm_mean) / (self.norm_std + 1e-8)

        nt_valid_mask = np.ones(L, dtype=bool)
        loss_mask = np.zeros(L, dtype=bool)
        loss_mask[target_start - start:target_end - start] = True
        metadata = {
            "chain_id": chain_id,
            "start": start,
            "end": end,
            "chain_len": len(full_seq),
        }

        return (
            torch.tensor(x,             dtype=torch.float32),
            torch.tensor(rsa,           dtype=torch.float32),
            torch.tensor(nt_features,   dtype=torch.float32),
            torch.tensor(nt_valid_mask, dtype=torch.bool),
            torch.tensor(loss_mask,     dtype=torch.bool),
            torch.tensor(max_asa,       dtype=torch.float32),
            L,
            metadata,
        )


    def mean_rsa(self) -> float:
        total = ssum = 0
        for chain_id, seq, asa_values in self.records:
            r = _compute_rsa(seq, asa_values)
            ssum += r.sum()
            total += len(r)
        return float(ssum / total) if total > 0 else float("nan")


def rsa_collate_fn(
    batch: List[Tuple[torch.Tensor, ...]],
) -> Tuple[torch.Tensor, ...]:
    """Pad samples to the longest chain of the batch (x, rsa, nt_features, masks, max_asa, lengths, metadata)."""
    xs, rsa_vals, nt_feats, nt_valids, loss_masks, max_asas, seq_lens, metadata = zip(*batch)
    B     = len(xs)
    L_max = max(seq_lens)

    x_pad        = torch.zeros(B, 5, L_max, L_max, dtype=torch.float32)
    rsa_pad      = torch.zeros(B, L_max,           dtype=torch.float32)
    nt_feat_pad  = torch.zeros(B, L_max, 23,       dtype=torch.float32)
    nt_valid_pad = torch.zeros(B, L_max,           dtype=torch.bool)
    loss_mask_pad = torch.zeros(B, L_max,           dtype=torch.bool)
    max_asa_pad  = torch.zeros(B, L_max,           dtype=torch.float32)

    for i, (x, r, nf, nv, loss_mask, ma, L) in enumerate(
        zip(xs, rsa_vals, nt_feats, nt_valids, loss_masks, max_asas, seq_lens)
    ):
        x_pad[i, :, :L, :L]  = x
        rsa_pad[i, :L]        = r
        nt_feat_pad[i, :L, :] = nf
        nt_valid_pad[i, :L]   = nv
        loss_mask_pad[i, :L]  = loss_mask
        max_asa_pad[i, :L]    = ma

    return (
        x_pad,
        rsa_pad,
        nt_feat_pad,
        nt_valid_pad,
        loss_mask_pad,
        max_asa_pad,
        list(seq_lens),
        list(metadata),
    )
