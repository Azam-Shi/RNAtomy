"""Contact-map head: a 2D convolutional network on the encoder's per-pair embeddings.

input_repr="pair" projects encode_pairs() output with a 1x1 convolution; "nt" builds outer features
[z_i, z_j, z_i * z_j] from encode_nucleotides(); "both" uses both. A stack of Conv2d-LayerNorm-GELU blocks and a
1x1 output convolution produce logits, which are symmetrised as (logits + logits^T) / 2.

Loss: sigmoid focal loss on the upper triangle. Metric: long-range (|i-j| >= 24) Top-L/k precision.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads.base import FinetuneHead


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of a [B, C, H, W] tensor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResBlock2D(nn.Module):
    """
    Pre-activation 2D residual block (He et al. 2016, v2) with channel-LayerNorm
    (batch-size-1 / variable-L safe) and an optional dilation:

        x + Conv3x3_d( Drop( GELU( LN( Conv3x3_d( GELU( LN(x) ) ) ) ) ) )

    The last conv is zero-initialised so every block starts as the identity.
    """

    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(dilation)
        self.norm1 = ChannelLayerNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=d, dilation=d, bias=False)
        self.norm2 = ChannelLayerNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=d, dilation=d)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(self.drop(F.gelu(self.norm2(h))))
        return x + h


class ContactMapHead(FinetuneHead):
    """embed_dim: encoder width. proj_dim / hidden_dim: channel widths. n_layers: convolution depth.
    focal_alpha / focal_gamma: focal-loss weight of the positive class and focusing parameter.
    min_seq_sep: pairs closer than this are excluded from the loss. long_range_sep: pairs at least this far
    apart are scored.
    """

    def __init__(
        self,
        embed_dim:      int   = 768,
        proj_dim:       int   = 64,
        hidden_dim:     int   = 128,
        n_layers:       int   = 3,
        conv_kernel:    int   = 1,
        input_repr:     str   = "nt",
        head_arch:      str   = "mlp",
        resnet_blocks:  int   = 8,
        resnet_dilations: tuple = (1, 2, 4, 8),
        resnet_dropout: float = 0.1,
        focal_alpha:    float = 0.5,
        focal_gamma:    float = 2.0,
        pos_weight:     float = 300.0,
        min_seq_sep:    int   = 6,
        long_range_sep: int   = 24,
    ) -> None:
        super().__init__()
        if input_repr not in ("nt", "pair", "both"):
            raise ValueError(f"input_repr must be nt|pair|both, got {input_repr}")
        if head_arch not in ("mlp", "resnet"):
            raise ValueError(f"head_arch must be mlp|resnet, got {head_arch}")
        self.proj_dim       = proj_dim
        self.input_repr     = input_repr
        self.head_arch      = head_arch
        self.min_seq_sep    = min_seq_sep
        self.long_range_sep = long_range_sep
        self.focal_alpha    = focal_alpha
        self.focal_gamma    = focal_gamma

        if input_repr in ("nt", "both"):
            self.proj = nn.Linear(embed_dim, proj_dim)
        if input_repr in ("pair", "both"):
            self.pair_proj = nn.Conv2d(embed_dim, proj_dim, kernel_size=1)

        in_ch = {"nt": 3 * proj_dim, "pair": proj_dim, "both": 4 * proj_dim}[input_repr]

        if head_arch == "resnet":
            dil = [int(d) for d in resnet_dilations]
            body: List[nn.Module] = [nn.Conv2d(in_ch, hidden_dim, kernel_size=1)]
            body += [ResBlock2D(hidden_dim, dil[i % len(dil)], resnet_dropout)
                     for i in range(int(resnet_blocks))]
            body += [ChannelLayerNorm(hidden_dim), nn.GELU(),
                     nn.Conv2d(hidden_dim, 1, kernel_size=1)]
            self.mlp = nn.Sequential(*body)
        else:
            pad = conv_kernel // 2
            layers: List[nn.Module] = []
            curr = in_ch
            for _ in range(n_layers - 1):
                layers += [
                    nn.Conv2d(curr, hidden_dim, kernel_size=conv_kernel, padding=pad, bias=False),
                    ChannelLayerNorm(hidden_dim),
                    nn.GELU(),
                ]
                curr = hidden_dim
            layers.append(nn.Conv2d(curr, 1, kernel_size=1))
            self.mlp = nn.Sequential(*layers)

        self.register_buffer("pos_weight", torch.tensor([pos_weight]))


    def forward(
        self,
        nt_emb:   Optional[torch.Tensor] = None,
        pair_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Returns symmetric logits [B, L, L]."""
        if self.input_repr == "nt":
            if nt_emb is None:
                raise ValueError("input_repr='nt' requires nt_emb [B, L, D]")
            feats = self._outer_features(nt_emb)
        elif self.input_repr == "pair":
            if pair_emb is None:
                raise ValueError("input_repr='pair' requires pair_emb [B, L, L, D]")
            feats = self.pair_proj(pair_emb.permute(0, 3, 1, 2))
        else:
            if pair_emb is None:
                raise ValueError("input_repr='both' requires pair_emb [B, L, L, D]")
            if nt_emb is None:
                nt_emb = pair_emb.mean(dim=2)
            feats = torch.cat(
                [self.pair_proj(pair_emb.permute(0, 3, 1, 2)),
                 self._outer_features(nt_emb)],
                dim=1,
            )

        logits_raw = self.mlp(feats).squeeze(1)
        return (logits_raw + logits_raw.transpose(-2, -1)) * 0.5

    def _outer_features(self, nt_emb: torch.Tensor) -> torch.Tensor:
        """cat(z_i, z_j, z_i ⊙ z_j) → [B, 3p, L, L]  (expand() is a view; no copy until cat)."""
        B, L, _ = nt_emb.shape
        z = self.proj(nt_emb)
        z_i = z.unsqueeze(2).expand(B, L, L, self.proj_dim)
        z_j = z.unsqueeze(1).expand(B, L, L, self.proj_dim)
        pair = torch.cat([z_i, z_j, z_i * z_j], dim=-1)
        return pair.permute(0, 3, 1, 2).contiguous()


    @staticmethod
    def _focal_loss(
        logits:      torch.Tensor,
        targets:     torch.Tensor,
        alpha:       float,
        gamma:       float,
    ) -> torch.Tensor:
        """Element-wise sigmoid focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)."""
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        p    = torch.sigmoid(logits)
        p_t  = p * targets + (1.0 - p) * (1.0 - targets)

        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)

        modulation = (1.0 - p_t) ** gamma

        return alpha_t * modulation * bce

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets:     torch.Tensor,
        seq_len:     Optional[int] = None,
        pair_valid_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Focal loss over unique upper-triangle pairs only (i < j), optionally masked.

        The min_seq_sep constraint (|i-j| >= min_seq_sep) is always applied.
        pair_valid_mask can add further exclusions (e.g. padding positions).
        Because supervision is on unique pairs only, the valid mask is
        symmetrized before masking.
        """
        _, L, _ = predictions.shape

        idx = torch.arange(L, device=predictions.device)
        sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        seq_mask = (sep >= self.min_seq_sep)
        tri_mask = idx.unsqueeze(1) < idx.unsqueeze(0)
        base_mask = (seq_mask & tri_mask).unsqueeze(0)

        if pair_valid_mask is not None:
            valid_mask = pair_valid_mask.to(dtype=torch.bool,
                                            device=predictions.device)
            if valid_mask.shape != predictions.shape:
                raise ValueError(
                    f"pair_valid_mask shape {tuple(valid_mask.shape)} "
                    f"does not match predictions shape {tuple(predictions.shape)}"
                )
            valid_mask = valid_mask & valid_mask.transpose(-2, -1)
            pair_mask = base_mask & valid_mask
        else:
            pair_mask = base_mask.expand_as(predictions)

        pair_mask_f = pair_mask.to(dtype=predictions.dtype)

        per_elem = self._focal_loss(
            predictions,
            targets.to(dtype=predictions.dtype),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )

        denom = pair_mask_f.sum().clamp(min=1.0)
        loss   = (per_elem * pair_mask_f).sum() / denom
        return loss


    @staticmethod
    def long_range_top_l_stats(
        logits: torch.Tensor,
        target: torch.Tensor,
        k:       int = 5,
        min_sep: int = 24,
    ):
        """
        Long-Range Top-L/k: among strict-upper-triangle pairs with
        |i-j| >= min_sep, take the top L/k predictions by probability.

        Returns (precision, n_hits, n_top).
        precision is NaN and n_top is 0 when there are no long-range pairs.
        Use n_hits / n_top to build a pooled (micro) precision over a set of RNAs.
        """
        L = logits.shape[0]
        probs = torch.sigmoid(logits.detach().float())

        idx = torch.arange(L, device=logits.device)
        sep = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        mask = (sep >= min_sep) & (idx.unsqueeze(1) < idx.unsqueeze(0))

        n_valid = int(mask.sum().item())
        if n_valid == 0:
            return float("nan"), 0, 0

        valid_probs   = probs[mask]
        valid_targets = target[mask].float()

        n_top = max(1, min(L // k, n_valid))
        _, top_idx = valid_probs.topk(n_top)
        n_hits = int(valid_targets[top_idx].sum().item())
        return n_hits / n_top, n_hits, n_top

    @staticmethod
    def long_range_top_l_precision(
        logits: torch.Tensor,
        target: torch.Tensor,
        k:       int = 5,
        min_sep: int = 24,
    ) -> float:
        """Long-Range Top-L/k precision (per-RNA). NaN when no long-range pairs."""
        return ContactMapHead.long_range_top_l_stats(
            logits, target, k=k, min_sep=min_sep
        )[0]

    def compute_metrics_per_sample(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        if logits.dim() == 3:
            logits = logits.squeeze(0)
        if target.dim() == 3:
            target = target.squeeze(0)
        return {
            f"top_l{k}_prec": self.long_range_top_l_precision(
                logits, target, k=k, min_sep=self.long_range_sep
            )
            for k in (10, 5, 2, 1)
        }

    def compute_metrics(
        self,
        predictions: Union[torch.Tensor, List[torch.Tensor]],
        targets:     Union[torch.Tensor, List[torch.Tensor]],
        **kwargs,
    ) -> Dict[str, float]:
        """
        FinetuneHead interface.  For variable-L sequences pass lists;
        for fixed-L batches pass stacked tensors [N, L, L].
        """
        if isinstance(predictions, list):
            sample_metrics = [
                self.compute_metrics_per_sample(p, t)
                for p, t in zip(predictions, targets)
            ]
        else:
            sample_metrics = [
                self.compute_metrics_per_sample(predictions[i], targets[i])
                for i in range(predictions.shape[0])
            ]

        out: Dict[str, float] = {}
        for k in sample_metrics[0].keys():
            vals = [m[k] for m in sample_metrics if not math.isnan(m[k])]
            out[k] = float("nan") if not vals else sum(vals) / len(vals)
        return out

    def on_train_epoch_start(self, epoch: int) -> None:
        pass

    def on_val_epoch_end(self, metrics: Dict[str, float]) -> None:
        pass
