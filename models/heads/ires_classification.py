"""IRES identification head (sequence-level binary classification).

head_type="cnn": per-nucleotide tokens [one-hot | encoder embedding | descriptors] go through a residual 1D CNN
with squeeze-excitation, attention + max pooling and an MLP. head_type="linear": pooled embedding plus flattened
one-hot sequence into a linear classifier. use_seq / use_struct / use_feat1d switch inputs off.

Loss: BCE with logits (pos_weight). Metrics: AUROC, AUPRC, F1, precision, recall, accuracy, MCC.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads.base import FinetuneHead
from models.heads.rna_protein_binding import _ResBlock1d, _SE1d


def binary_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """The seven metrics of structRFM's IRES radar plot + the confusion counts."""
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                 matthews_corrcoef, precision_score, recall_score,
                                 roc_auc_score)
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    y = labels.astype(np.int32)
    pred = (prob >= threshold).astype(np.int32)
    out = {"threshold": float(threshold),
           "accuracy": float(accuracy_score(y, pred)),
           "tp": int(((pred == 1) & (y == 1)).sum()), "fp": int(((pred == 1) & (y == 0)).sum()),
           "tn": int(((pred == 0) & (y == 0)).sum()), "fn": int(((pred == 0) & (y == 1)).sum())}
    out["precision"] = float(precision_score(y, pred, zero_division=0))
    out["recall"] = float(recall_score(y, pred, zero_division=0))
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0
    if len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, prob))
        out["auprc"] = float(average_precision_score(y, prob))
    else:
        out["auroc"] = out["auprc"] = float("nan")
    return out


def youden_threshold(logits: np.ndarray, labels: np.ndarray) -> float:
    """Probability threshold maximising TPR-FPR (prevalence-independent), tuned on VAL only."""
    from sklearn.metrics import roc_curve
    y = labels.astype(np.int32)
    if len(np.unique(y)) < 2:
        return 0.5
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    fpr, tpr, thr = roc_curve(y, prob)
    j = tpr - fpr
    if float(j.max()) <= 0.0:
        return 0.5
    t = float(thr[int(np.argmax(j))])
    return min(max(t, 1e-6), 1.0 - 1e-6)


class IRESHead(FinetuneHead):
    def __init__(
        self,
        struct_dim: int,
        feat1d_dim: int = 23,
        head_type: str = "cnn",
        conv_dim: int = 128,
        n_blocks: int = 3,
        kernel_size: int = 9,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        max_len: int = 174,
        use_seq: bool = True,
        use_struct: bool = True,
        use_feat1d: bool = False,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        if head_type not in ("linear", "cnn"):
            raise ValueError(f"head_type must be linear|cnn, got {head_type}")
        self.head_type = head_type
        self.max_len = int(max_len)
        self.use_seq, self.use_struct, self.use_feat1d = use_seq, use_struct, use_feat1d
        self.feat1d_dim = feat1d_dim

        if head_type == "linear":
            in_dim = (struct_dim if use_struct else 0) + (self.max_len * 4 if use_seq else 0) \
                     + (feat1d_dim if use_feat1d else 0)
            if in_dim == 0:
                raise ValueError("IRESHead: all input branches disabled")
            self.in_dim = in_dim
            self.norm = nn.LayerNorm(in_dim)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(in_dim, 1)
            return

        in_dim = (4 if use_seq else 0) + (struct_dim if use_struct else 0) \
                 + (feat1d_dim if use_feat1d else 0)
        if in_dim == 0:
            raise ValueError("IRESHead: all input branches disabled")
        self.in_dim = in_dim
        self.input_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, conv_dim)
        self.stem = nn.Conv1d(conv_dim, conv_dim, kernel_size, padding=kernel_size // 2)
        self.blocks = nn.ModuleList(_ResBlock1d(conv_dim, kernel_size, dropout) for _ in range(n_blocks))
        self.se = _SE1d(conv_dim) if use_se else None
        self.post_norm = nn.GroupNorm(8, conv_dim)
        self.attn_q = nn.Linear(conv_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * conv_dim),
            nn.Linear(2 * conv_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, nt_emb, pooled, onehot, feat1d, mask, **kwargs) -> torch.Tensor:
        return self._forward_linear(pooled, onehot, feat1d, mask) if self.head_type == "linear" \
            else self._forward_cnn(nt_emb, onehot, feat1d, mask)

    def _forward_linear(self, pooled, onehot, feat1d, mask) -> torch.Tensor:
        B = onehot.shape[0]
        parts = []
        if self.use_struct:
            if pooled is None:
                raise ValueError("use_struct=True but pooled embedding is None")
            parts.append(pooled.float())
        if self.use_seq:
            oh = onehot.float() * mask.unsqueeze(-1).float()
            oh = oh[:, :self.max_len]
            if oh.shape[1] < self.max_len:
                oh = F.pad(oh, (0, 0, 0, self.max_len - oh.shape[1]))
            parts.append(oh.reshape(B, -1))
        if self.use_feat1d:
            m = mask.unsqueeze(-1).float()
            parts.append((feat1d[..., :self.feat1d_dim].float() * m).sum(1) / m.sum(1).clamp(min=1.0))
        x = torch.cat(parts, dim=-1)
        return self.fc(self.drop(self.norm(x))).squeeze(-1)

    def _forward_cnn(self, nt_emb, onehot, feat1d, mask) -> torch.Tensor:
        B, L, _ = onehot.shape
        parts = []
        if self.use_seq:
            parts.append(onehot.float())
        if self.use_struct:
            if nt_emb is None:
                raise ValueError("use_struct=True but nt_emb is None")
            parts.append(nt_emb[:, :L, :].float())
        if self.use_feat1d:
            parts.append(feat1d[:, :L, :self.feat1d_dim].float())
        x = torch.cat(parts, dim=-1)
        mc = mask.unsqueeze(1).float()
        x = self.input_norm(x) * mask.unsqueeze(-1).float()
        x = self.proj(x).transpose(1, 2)
        x = self.stem(x) * mc
        for blk in self.blocks:
            x = blk(x, mc)
        if self.se is not None:
            x = self.se(x, mask)
        x = F.gelu(self.post_norm(x)) * mc
        h = x.transpose(1, 2)
        s = self.attn_q(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        attn_pool = (torch.softmax(s, dim=1).unsqueeze(-1) * h).sum(1)
        max_pool = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        return self.classifier(torch.cat([attn_pool, max_pool], dim=-1)).squeeze(-1)

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                     pos_weight: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            predictions, targets.float(),
            pos_weight=pos_weight.to(predictions.device) if pos_weight is not None else None)

    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                        threshold: float = 0.5, **kwargs) -> Dict[str, float]:
        return binary_metrics(predictions.detach().float().cpu().numpy(),
                              targets.detach().cpu().numpy(), threshold)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
