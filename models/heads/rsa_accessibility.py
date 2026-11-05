"""Solvent-accessibility head: per-nucleotide regression of RSA = ASA / max ASA.

Input: encode_nucleotides(..., include_diagonal=True) [B, L, 2D] concatenated with the descriptors [B, L, 23],
followed by LayerNorm, a zero-initialised residual depthwise Conv1d (kernel 5), Linear -> GELU -> Dropout -> Linear.
Loss: masked MSE on window-owned, non-padding positions.

Metrics (RNAsol protocol): mean per-chain PCC / SCC, pooled PCC / SCC (diagnostic), mae_A2 (MAE in A^2 on the
recovered ASA scale) and mae_rsa.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads.base import FinetuneHead

try:
    from scipy.stats import pearsonr, spearmanr
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    if _HAVE_SCIPY:
        return float(pearsonr(a, b)[0])
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    if _HAVE_SCIPY:
        r = spearmanr(a, b).correlation
        return float(r) if r == r else float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return _pearson(ra, rb)


class RSAAccessibilityHead(FinetuneHead):
    """embed_dim: width of the nucleotide embedding. hidden_dim / dropout: MLP width and dropout."""

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        feat_dim: int = 23,
        include_diagonal: bool = True,
        use_local_conv: bool = True,
        conv_kernel: int = 5,
    ) -> None:
        super().__init__()

        self.feat_dim = int(feat_dim)
        self.include_diagonal = bool(include_diagonal)
        self.use_local_conv = bool(use_local_conv)

        enc_dim = embed_dim * (2 if self.include_diagonal else 1)
        in_dim = enc_dim + max(self.feat_dim, 0)
        self.in_dim = in_dim

        self.in_norm = nn.LayerNorm(in_dim)

        if self.use_local_conv:
            pad = conv_kernel // 2
            self.local_conv = nn.Conv1d(
                in_dim, in_dim, kernel_size=conv_kernel, padding=pad, groups=in_dim
            )
            nn.init.zeros_(self.local_conv.weight)
            nn.init.zeros_(self.local_conv.bias)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        nt_emb: torch.Tensor,
        nt_features: Optional[torch.Tensor] = None,
        nt_valid_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        h = nt_emb
        if self.feat_dim > 0 and nt_features is not None:
            L = h.shape[1]
            f = nt_features
            if f.shape[1] > L:
                f = f[:, :L, :]
            elif f.shape[1] < L:
                pad = f.new_zeros(f.shape[0], L - f.shape[1], f.shape[2])
                f = torch.cat([f, pad], dim=1)
            h = torch.cat([h, f], dim=-1)

        h = self.in_norm(h)
        if self.use_local_conv:
            h = h + self.local_conv(h.transpose(1, 2)).transpose(1, 2)
        return self.mlp(h).squeeze(-1)


    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        nt_valid_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if nt_valid_mask is not None:
            pred_valid = predictions[nt_valid_mask]
            tgt_valid = targets[nt_valid_mask]
        else:
            pred_valid = predictions.reshape(-1)
            tgt_valid = targets.reshape(-1)
        return F.mse_loss(pred_valid, tgt_valid)


    @staticmethod
    def compute_metrics(
        predictions: Union[Sequence[np.ndarray], np.ndarray],
        targets: Union[Sequence[np.ndarray], np.ndarray],
        max_asa: Optional[Sequence[np.ndarray]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """predictions / targets: one 1D array per chain (or one flat array, which gives pooled metrics only).
        max_asa: matching per-nucleotide max ASA (A^2); needed for mae_A2.
        """
        def _to_list(x):
            if isinstance(x, (list, tuple)):
                return [np.asarray(v, dtype=np.float64).reshape(-1) for v in x]
            arr = np.asarray(x, dtype=np.float64)
            return [arr.reshape(-1)] if arr.ndim <= 1 else [arr[i].reshape(-1) for i in range(arr.shape[0])]

        pred_chains = _to_list(predictions)
        tgt_chains = _to_list(targets)
        asa_chains = _to_list(max_asa) if max_asa is not None else None

        pccs, sccs, chain_maes = [], [], []
        all_pred, all_tgt = [], []
        all_asa_pred, all_asa_tgt = [], []

        for i, (p, t) in enumerate(zip(pred_chains, tgt_chains)):
            if len(p) != len(t):
                raise ValueError(f"chain {i}: prediction/target length mismatch {len(p)} != {len(t)}")
            pcc = _pearson(p, t)
            scc = _spearman(p, t)
            if pcc == pcc:
                pccs.append(pcc)
            if scc == scc:
                sccs.append(scc)
            all_pred.append(p)
            all_tgt.append(t)
            if asa_chains is not None:
                ma = asa_chains[i]
                asa_pred = p * ma
                asa_tgt = t * ma
                chain_maes.append(float(np.mean(np.abs(asa_pred - asa_tgt))))
                all_asa_pred.append(asa_pred)
                all_asa_tgt.append(asa_tgt)

        pooled_pred = np.concatenate(all_pred) if all_pred else np.array([])
        pooled_tgt = np.concatenate(all_tgt) if all_tgt else np.array([])

        metrics: Dict[str, float] = {
            "mean_pcc": float(np.mean(pccs)) if pccs else float("nan"),
            "mean_scc": float(np.mean(sccs)) if sccs else float("nan"),
            "pooled_pcc": _pearson(pooled_pred, pooled_tgt) if len(pooled_pred) else float("nan"),
            "pooled_scc": _spearman(pooled_pred, pooled_tgt) if len(pooled_pred) else float("nan"),
            "mae_rsa": float(np.mean(np.abs(pooled_pred - pooled_tgt))) if len(pooled_pred) else float("nan"),
            "n_chains": float(len(pred_chains)),
            "n_chains_scored": float(len(pccs)),
        }
        if asa_chains is not None:
            pooled_asa_pred = np.concatenate(all_asa_pred)
            pooled_asa_tgt = np.concatenate(all_asa_tgt)
            metrics["mae_A2"] = float(np.mean(np.abs(pooled_asa_pred - pooled_asa_tgt)))
            metrics["mean_chain_mae_A2"] = float(np.mean(chain_maes)) if chain_maes else float("nan")
        else:
            metrics["mae_A2"] = float("nan")
            metrics["mean_chain_mae_A2"] = float("nan")
        return metrics
