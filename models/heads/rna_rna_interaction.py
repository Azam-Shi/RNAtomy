"""RNA-RNA interaction heads, both reading the two RNAs position by position so that inter-molecular
complementarity can be represented.

RRIJointSeqHead: per-nucleotide tokens of both RNAs plus a base embedding are joined as [A | SEP | B] and read by
Conv1d -> BiLSTM -> MLP.
RRIInteractionMapHead: builds an interaction tensor M[i, j] from both token sets and Watson-Crick / G-U
complementarity channels, applies a small 2D CNN and pools.

Both also take engineered seed-complementarity features. Loss: BCE with logits.
Metrics: AUROC, accuracy, F1, recall, precision.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.heads.base import FinetuneHead


class _RRIHeadBase(FinetuneHead):
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                     **kwargs) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(predictions, targets.float())

    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                        **kwargs) -> Dict[str, float]:
        scores = predictions.detach().cpu().numpy().astype(np.float64)
        labels = targets.detach().cpu().numpy().astype(np.int32)
        preds = (scores >= 0.0).astype(np.int32)
        try:
            auroc = float(roc_auc_score(labels, scores))
        except ValueError:
            auroc = float("nan")
        return {
            "auroc": auroc,
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _complementarity_channels(onehot_a: torch.Tensor, onehot_b: torch.Tensor):
    """
    onehot_* : [B, L, 4] with column order A, C, G, U (matches _NUC in the dataset).
    Returns WC [B, La, Lb] and GU [B, La, Lb] as soft {0,1} pairing indicators.
    """
    aA, aC, aG, aU = onehot_a[..., 0], onehot_a[..., 1], onehot_a[..., 2], onehot_a[..., 3]
    bA, bC, bG, bU = onehot_b[..., 0], onehot_b[..., 1], onehot_b[..., 2], onehot_b[..., 3]
    wc = (torch.einsum("bi,bj->bij", aA, bU) + torch.einsum("bi,bj->bij", aU, bA)
          + torch.einsum("bi,bj->bij", aG, bC) + torch.einsum("bi,bj->bij", aC, bG))
    gu = (torch.einsum("bi,bj->bij", aG, bU) + torch.einsum("bi,bj->bij", aU, bG))
    return wc, gu


class RRIJointSeqHead(_RRIHeadBase):
    def __init__(
        self,
        embed_dim: int,
        struct_dim: Optional[int] = None,
        base_dim: int = 32,
        proj_dim: int = 128,
        conv_dim: int = 256,
        lstm_hidden: int = 128,
        seed_feat_dim: int = 11,
        dropout: float = 0.1,
        use_seed_feats: bool = True,
        use_struct: bool = True,
    ) -> None:
        super().__init__()
        struct_dim = struct_dim or embed_dim
        self.use_seed_feats = use_seed_feats
        self.use_struct = use_struct

        self.base_proj = nn.Linear(4, base_dim)
        tok_dim = base_dim
        if use_struct:
            self.struct_norm = nn.LayerNorm(struct_dim)
            self.struct_proj = nn.Linear(struct_dim, proj_dim)
            tok_dim += proj_dim
        self.sep = nn.Parameter(torch.zeros(tok_dim))
        nn.init.normal_(self.sep, std=0.02)

        self.conv = nn.Sequential(
            nn.Conv1d(tok_dim, conv_dim, kernel_size=11, padding=5),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(conv_dim, lstm_hidden, batch_first=True, bidirectional=True)

        pooled = 4 * lstm_hidden
        fuse_in = pooled
        if use_seed_feats:
            self.seed_mlp = nn.Sequential(nn.Linear(seed_feat_dim, 32), nn.GELU())
            fuse_in += 32
        self.classifier = nn.Sequential(
            nn.LayerNorm(fuse_in),
            nn.Linear(fuse_in, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _tok(self, z: Optional[torch.Tensor], onehot: torch.Tensor) -> torch.Tensor:
        parts = [self.base_proj(onehot.float())]
        if self.use_struct:
            parts.append(self.struct_proj(self.struct_norm(z.float())))
        return torch.cat(parts, dim=-1)

    def forward(self, nt_a, nt_b, onehot_a, onehot_b, mask_a, mask_b,
                seed_feats=None, **kwargs) -> torch.Tensor:
        B = onehot_a.shape[0]
        ta = self._tok(nt_a, onehot_a)
        tb = self._tok(nt_b, onehot_b)
        sep = self.sep.view(1, 1, -1).expand(B, 1, -1)
        seq = torch.cat([ta, sep, tb], dim=1)
        m = torch.cat([mask_a, torch.ones(B, 1, dtype=torch.bool, device=ta.device), mask_b], dim=1)

        seq = seq * m.unsqueeze(-1)
        h = self.conv(seq.transpose(1, 2))
        h = self.pool(h)
        mp = F.max_pool1d(m.float().unsqueeze(1), kernel_size=2).squeeze(1) > 0.5
        h = h.transpose(1, 2)

        h, _ = self.lstm(h)
        mf = mp.unsqueeze(-1).float()
        mean_p = (h * mf).sum(1) / mf.sum(1).clamp(min=1.0)
        max_p = h.masked_fill(~mp.unsqueeze(-1), float("-inf")).max(1).values
        pooled = torch.cat([mean_p, max_p], dim=-1)

        if self.use_seed_feats:
            pooled = torch.cat([pooled, self.seed_mlp(seed_feats.float())], dim=-1)
        return self.classifier(pooled).squeeze(-1)


class _ResBlock2d(nn.Module):
    def __init__(self, dim: int, k: int = 3) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, k, padding=k // 2)
        self.c2 = nn.Conv2d(dim, dim, k, padding=k // 2)
        self.n1 = nn.GroupNorm(8, dim)
        self.n2 = nn.GroupNorm(8, dim)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(x + h)


class RRIInteractionMapHead(_RRIHeadBase):
    def __init__(
        self,
        embed_dim: int,
        struct_dim: Optional[int] = None,
        base_dim: int = 16,
        pair_rank: int = 16,
        cnn_dim: int = 48,
        n_res_blocks: int = 2,
        seed_feat_dim: int = 11,
        dropout: float = 0.1,
        use_seed_feats: bool = True,
        use_struct: bool = True,
    ) -> None:
        super().__init__()
        struct_dim = struct_dim or embed_dim
        self.use_seed_feats = use_seed_feats
        self.use_struct = use_struct

        self.base_proj = nn.Linear(4, base_dim)
        tok_dim = base_dim
        if use_struct:
            self.struct_norm = nn.LayerNorm(struct_dim)
            self.struct_proj = nn.Linear(struct_dim, 64)
            tok_dim += 64
        self.to_rank_a = nn.Linear(tok_dim, pair_rank)
        self.to_rank_b = nn.Linear(tok_dim, pair_rank)

        in_ch = 2 * pair_rank + 2
        self.stem = nn.Conv2d(in_ch, cnn_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(_ResBlock2d(cnn_dim, 3 if i % 2 == 0 else 5)
                                    for i in range(n_res_blocks))
        self.head_conv = nn.Conv2d(cnn_dim, 32, kernel_size=3, padding=1)

        fuse_in = 64
        if use_seed_feats:
            self.seed_mlp = nn.Sequential(nn.Linear(seed_feat_dim, 32), nn.GELU())
            fuse_in += 32
        self.classifier = nn.Sequential(
            nn.LayerNorm(fuse_in),
            nn.Linear(fuse_in, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _tok(self, z, onehot):
        parts = [self.base_proj(onehot.float())]
        if self.use_struct:
            parts.append(self.struct_proj(self.struct_norm(z.float())))
        return torch.cat(parts, dim=-1)

    def forward(self, nt_a, nt_b, onehot_a, onehot_b, mask_a, mask_b,
                seed_feats=None, **kwargs) -> torch.Tensor:
        ra = self.to_rank_a(self._tok(nt_a, onehot_a))
        rb = self.to_rank_b(self._tok(nt_b, onehot_b))

        mult = ra.unsqueeze(2) * rb.unsqueeze(1)
        diff = -(ra.unsqueeze(2) - rb.unsqueeze(1)).abs()
        wc, gu = _complementarity_channels(onehot_a, onehot_b)
        M = torch.cat([mult, diff, wc.unsqueeze(-1), gu.unsqueeze(-1)], dim=-1)
        M = M.permute(0, 3, 1, 2)

        pair_mask = (mask_a.unsqueeze(2) & mask_b.unsqueeze(1)).unsqueeze(1).float()
        h = self.stem(M) * pair_mask
        for blk in self.blocks:
            h = blk(h) * pair_mask
        h = self.head_conv(h)

        pm = pair_mask.bool().expand_as(h)
        mean_p = (h * pair_mask).sum(dim=(2, 3)) / pair_mask.sum(dim=(2, 3)).clamp(min=1.0)
        max_p = h.masked_fill(~pm, float("-inf")).amax(dim=(2, 3))
        pooled = torch.cat([mean_p, max_p], dim=-1)

        if self.use_seed_feats:
            pooled = torch.cat([pooled, self.seed_mlp(seed_feats.float())], dim=-1)
        return self.classifier(pooled).squeeze(-1)


class RNARNAInteractionHead(_RRIHeadBase):
    """Siamese pooled-embedding NLI fusion. Kept as a weak-baseline ablation;
    it cannot represent inter-molecular complementarity -- prefer the joint or
    map heads."""

    def __init__(self, embed_dim: int = 768, hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        dims = [4 * embed_dim] + list(hidden_dims)
        layers: List[nn.Module] = [nn.LayerNorm(4 * embed_dim)]
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, **kwargs) -> torch.Tensor:
        combined = torch.cat([emb_a, emb_b, emb_a * emb_b, torch.abs(emb_a - emb_b)], dim=-1)
        return self.mlp(combined).squeeze(-1)
