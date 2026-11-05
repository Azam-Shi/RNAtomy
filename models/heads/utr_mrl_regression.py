"""5'UTR mean-ribosome-load regression head.

Per-nucleotide tokens [one-hot (4) | position features (2) | encoder embedding (D) | descriptors (23)] go through
a dilated residual 1D CNN, masked attention / mean / max pooling and a fusion MLP that also receives six engineered
upstream-AUG features. The embedding and descriptor inputs can be switched off.

forward(nt_emb, onehot, feat1d, aux, valid_mask) -> prediction [B]
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.stats import pearsonr, spearmanr
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if _HAVE_SCIPY:
        return float(pearsonr(a, b)[0])
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if _HAVE_SCIPY:
        return float(spearmanr(a, b).correlation)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return _pearson(ra, rb)


class _DilatedResidualBlock(nn.Module):
    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = h * pad_mask.unsqueeze(-1)
        h = h.transpose(1, 2)
        h = F.gelu(self.conv1(h))
        h = self.drop(h)
        h = self.conv2(h)
        h = h.transpose(1, 2)
        return x + h * pad_mask.unsqueeze(-1)


class _AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        scores = self.query(x).squeeze(-1)
        scores = scores.masked_fill(~pad_mask, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (w * x).sum(dim=1)


class UTRMrlHead(nn.Module):
    POS_DIM = 2
    ONEHOT_DIM = 4

    def __init__(
        self,
        embed_dim: int,
        feat1d_dim: int = 23,
        aux_dim: int = 6,
        conv_dim: int = 256,
        n_conv_blocks: int = 4,
        kernel_size: int = 5,
        dilations: Optional[List[int]] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_struct_emb: bool = True,
        use_feat1d: bool = False,
        use_aux: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.feat1d_dim = feat1d_dim
        self.aux_dim = aux_dim
        self.use_struct_emb = use_struct_emb
        self.use_feat1d = use_feat1d
        self.use_aux = use_aux
        self.dilations = dilations or [1, 2, 4, 8]
        hidden_dims = hidden_dims or [256, 128]

        in_dim = self.ONEHOT_DIM + self.POS_DIM
        if use_struct_emb:
            in_dim += embed_dim
        if use_feat1d:
            in_dim += feat1d_dim
        self.in_dim = in_dim

        self.input_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, conv_dim)
        self.blocks = nn.ModuleList(
            _DilatedResidualBlock(conv_dim, kernel_size, self.dilations[i % len(self.dilations)], dropout)
            for i in range(n_conv_blocks)
        )
        self.post_norm = nn.LayerNorm(conv_dim)
        self.attn_pool = _AttentionPool(conv_dim)

        pooled_dim = conv_dim * 3
        fuse_in = pooled_dim
        if use_aux:
            self.aux_mlp = nn.Sequential(
                nn.Linear(aux_dim, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 32), nn.GELU()
            )
            fuse_in += 32

        layers: List[nn.Module] = []
        d = fuse_in
        for hd in hidden_dims:
            layers += [nn.Linear(d, hd), nn.GELU(), nn.Dropout(dropout)]
            d = hd
        layers.append(nn.Linear(d, 1))
        self.regressor = nn.Sequential(*layers)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def on_train_epoch_start(self, epoch: int) -> None:
        pass

    def on_val_epoch_end(self, metrics: Dict[str, float]) -> None:
        pass

    @staticmethod
    def _positional_features(pad_mask: torch.Tensor) -> torch.Tensor:
        B, L = pad_mask.shape
        lengths = pad_mask.sum(dim=1).clamp(min=1).float()
        idx = torch.arange(L, device=pad_mask.device).float().unsqueeze(0).expand(B, L)
        frac = idx / (lengths.unsqueeze(1) - 1).clamp(min=1)
        frac = frac.clamp(0.0, 1.0)
        return torch.stack([frac, 1.0 - frac], dim=-1) * pad_mask.unsqueeze(-1)

    def forward(
        self,
        nt_emb: Optional[torch.Tensor],
        onehot: torch.Tensor,
        feat1d: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = onehot.shape
        if valid_mask is None:
            valid_mask = torch.ones(B, L, dtype=torch.bool, device=onehot.device)
        pad_mask = valid_mask.bool()

        parts = [onehot.float(), self._positional_features(pad_mask)]
        if self.use_struct_emb:
            if nt_emb is None:
                raise ValueError("use_struct_emb=True but nt_emb is None")
            parts.append(nt_emb[:, :L, :].float())
        if self.use_feat1d:
            if feat1d is None:
                raise ValueError("use_feat1d=True but feat1d is None")
            parts.append(feat1d[:, :L, :self.feat1d_dim].float())

        x = torch.cat(parts, dim=-1)
        x = self.input_norm(x)
        x = self.proj(x) * pad_mask.unsqueeze(-1)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        x = self.post_norm(x) * pad_mask.unsqueeze(-1)

        m = pad_mask.unsqueeze(-1).float()
        mean_pool = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        max_pool = x.masked_fill(~pad_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        attn = self.attn_pool(x, pad_mask)
        pooled = torch.cat([attn, mean_pool, max_pool], dim=-1)

        if self.use_aux:
            if aux is None:
                raise ValueError("use_aux=True but aux is None")
            pooled = torch.cat([pooled, self.aux_mlp(aux.float())], dim=-1)

        return self.regressor(pooled).squeeze(-1)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if sample_weight is None:
            return F.mse_loss(pred, target)
        w = sample_weight / sample_weight.mean().clamp(min=1e-8)
        return (w * (pred - target) ** 2).mean()

    @staticmethod
    def compute_metrics(
        preds: np.ndarray,
        targets: np.ndarray,
        scaler=None,
    ) -> Dict[str, float]:
        preds = np.asarray(preds, dtype=np.float64).ravel()
        targets = np.asarray(targets, dtype=np.float64).ravel()

        ss_res = float(((targets - preds) ** 2).sum())
        ss_tot = float(((targets - targets.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r = _pearson(preds, targets)

        out = {
            "r2": r2,
            "pearson_r": r,
            "pearson_r2": r * r,
            "spearman_r": _spearman(preds, targets),
            "rmse": float(np.sqrt(((preds - targets) ** 2).mean())),
            "mae": float(np.abs(preds - targets).mean()),
        }
        if scaler is not None:
            p_raw = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
            t_raw = scaler.inverse_transform(targets.reshape(-1, 1)).ravel()
            out["rmse_raw"] = float(np.sqrt(((p_raw - t_raw) ** 2).mean()))
            out["mae_raw"] = float(np.abs(p_raw - t_raw).mean())
        return out
