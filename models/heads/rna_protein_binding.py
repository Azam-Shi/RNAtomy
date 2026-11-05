"""RNA-protein binding heads.

RNAProteinBindingHead: per-nucleotide tokens [one-hot | icSHAPE | measured flag | encoder embedding | descriptors]
go through a residual 1D CNN, attention + max pooling and an MLP.
PrismNetHead: re-implementation of PrismNet's 2D-convolution network on [one-hot | icSHAPE] (mode pu | seq | str).
EmbeddingPrismNetHead: the same network on [one-hot | encoder embedding].
RNAEmbeddingMLPHead: masked mean of the nucleotide embeddings followed by an MLP.

Loss: BCE with logits (optional pos_weight). Metrics: AUROC, AUPR, accuracy.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads.base import FinetuneHead


class _RBPHeadBase(FinetuneHead):
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                     pos_weight: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            predictions, targets.float(),
            pos_weight=pos_weight.to(predictions.device) if pos_weight is not None else None,
        )

    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                        **kwargs) -> Dict[str, float]:
        scores = torch.sigmoid(predictions.detach().float()).cpu().numpy()
        labels = targets.detach().cpu().numpy().astype(np.int32)
        out = {"auroc": float("nan"), "aupr": float("nan"), "aupr_trapz": float("nan"),
               "accuracy": float(((scores >= 0.5).astype(np.int32) == labels).mean())}
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
            out["auroc"] = float(roc_auc_score(labels, scores))
            out["aupr"] = float(average_precision_score(labels, scores))
            precision, recall, _ = precision_recall_curve(labels, scores)
            out["aupr_trapz"] = float(auc(recall, precision))
        except Exception:
            pass
        return out

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _SE1d(nn.Module):
    def __init__(self, ch: int, reduction: int = 8) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, max(ch // reduction, 4)), nn.ReLU(inplace=True),
                                nn.Linear(max(ch // reduction, 4), ch), nn.Sigmoid())

    def forward(self, x, mask):
        m = mask.unsqueeze(1).float()
        pooled = (x * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        return x * self.fc(pooled).unsqueeze(-1)


class _ResBlock1d(nn.Module):
    def __init__(self, dim: int, k: int, dropout: float) -> None:
        super().__init__()
        p = k // 2
        self.n1 = nn.GroupNorm(8, dim)
        self.c1 = nn.Conv1d(dim, dim, k, padding=p)
        self.n2 = nn.GroupNorm(8, dim)
        self.c2 = nn.Conv1d(dim, dim, k, padding=p)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, m):
        h = self.c1(F.gelu(self.n1(x))) * m
        h = self.drop(h)
        h = self.c2(F.gelu(self.n2(h))) * m
        return x + h


class RNAProteinBindingHead(_RBPHeadBase):
    def __init__(
        self,
        embed_dim: int,
        struct_dim: Optional[int] = None,
        feat1d_dim: int = 23,
        conv_dim: int = 128,
        n_blocks: int = 3,
        kernel_size: int = 11,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_seq: bool = True,
        use_icshape: bool = True,
        use_struct: bool = True,
        use_feat1d: bool = False,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        struct_dim = struct_dim or embed_dim
        self.use_seq, self.use_icshape = use_seq, use_icshape
        self.use_struct, self.use_feat1d, self.use_se = use_struct, use_feat1d, use_se
        self.feat1d_dim = feat1d_dim

        in_dim = 0
        in_dim += 4 if use_seq else 0
        in_dim += 2 if use_icshape else 0
        in_dim += struct_dim if use_struct else 0
        in_dim += feat1d_dim if use_feat1d else 0
        if in_dim == 0:
            raise ValueError("RNAProteinBindingHead: all input branches disabled")
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

    def forward(self, nt_emb, onehot, icshape, ic_measured, feat1d, mask, **kwargs) -> torch.Tensor:
        B, L, _ = onehot.shape
        parts = []
        if self.use_seq:
            parts.append(onehot.float())
        if self.use_icshape:
            parts.append(torch.stack([icshape.float(), ic_measured.float()], dim=-1))
        if self.use_struct:
            if nt_emb is None:
                raise ValueError("use_struct=True but nt_emb is None")
            parts.append(nt_emb[:, :L, :].float())
        if self.use_feat1d:
            parts.append(feat1d[:, :L, :self.feat1d_dim].float())
        x = torch.cat(parts, dim=-1)

        m1 = mask.unsqueeze(-1).float()
        x = self.input_norm(x) * m1
        x = self.proj(x).transpose(1, 2)
        mc = mask.unsqueeze(1).float()
        x = self.stem(x) * mc
        for blk in self.blocks:
            x = blk(x, mc)
        if self.se is not None:
            x = self.se(x, mask)
        x = F.gelu(self.post_norm(x)) * mc
        h = x.transpose(1, 2)

        scores = self.attn_q(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        attn_pool = (w * h).sum(dim=1)
        max_pool = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        return self.classifier(torch.cat([attn_pool, max_pool], dim=-1)).squeeze(-1)


class _PN_Conv2d(nn.Module):
    def __init__(self, cin, cout, k, same_padding=True, bn=True, relu=True):
        super().__init__()
        p = ((k[0] - 1) // 2, (k[1] - 1) // 2) if same_padding else (0, 0)
        self.conv = nn.Conv2d(cin, cout, k, 1, padding=p)
        self.bn = nn.BatchNorm2d(cout) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class _PN_SEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        return self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)


class _PN_ResidualBlock2D(nn.Module):
    def __init__(self, planes, kernel_size=(11, 5), padding=(5, 2)):
        super().__init__()
        self.c1 = nn.Conv2d(planes, planes, 1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes, planes * 2, kernel_size, padding=padding, bias=False)
        self.b2 = nn.BatchNorm2d(planes * 2)
        self.c3 = nn.Conv2d(planes * 2, planes * 4, 1, bias=False)
        self.b3 = nn.BatchNorm2d(planes * 4)
        self.downsample = nn.Sequential(nn.Conv2d(planes, planes * 4, 1, bias=False),
                                        nn.BatchNorm2d(planes * 4))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.b1(self.c1(x)))
        out = self.relu(self.b2(self.c2(out)))
        out = self.b3(self.c3(out))
        out = out + self.downsample(x)
        return self.relu(out)


class _PN_ResidualBlock1D(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.c1 = nn.Conv1d(planes, planes, 1, bias=False)
        self.b1 = nn.BatchNorm1d(planes)
        self.c2 = nn.Conv1d(planes, planes * 2, 11, padding=5, bias=False)
        self.b2 = nn.BatchNorm1d(planes * 2)
        self.c3 = nn.Conv1d(planes * 2, planes * 8, 1, bias=False)
        self.b3 = nn.BatchNorm1d(planes * 8)
        self.downsample = nn.Sequential(nn.Conv1d(planes, planes * 8, 1, bias=False),
                                        nn.BatchNorm1d(planes * 8))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.b1(self.c1(x)))
        out = self.relu(self.b2(self.c2(out)))
        out = self.b3(self.c3(out))
        out = out + self.downsample(x)
        return self.relu(out)


class PrismNetHead(_RBPHeadBase):
    """
    Faithful re-implementation. Input built from the one-hot sequence and the
    icSHAPE channel; missing icSHAPE positions carry -1 (as in the original data).
    """

    def __init__(self, mode: str = "pu", base_channel: int = 8) -> None:
        super().__init__()
        if mode not in ("pu", "seq", "str"):
            raise ValueError("mode must be pu | seq | str")
        self.mode = mode
        h_k, h_p = (5, 2)
        if mode == "seq":
            self.n_features, h_k, h_p = 4, 3, 1
        elif mode == "str":
            self.n_features, h_k, h_p = 1, 1, 0
        else:
            self.n_features = 5
        self.conv = _PN_Conv2d(1, base_channel, (11, h_k), same_padding=True, bn=True)
        self.se = _PN_SEBlock(base_channel)
        self.res2d = _PN_ResidualBlock2D(base_channel, (11, h_k), (5, h_p))
        self.avgpool = nn.AvgPool2d((1, self.n_features))
        self.res1d = _PN_ResidualBlock1D(base_channel * 4)
        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channel * 4 * 8, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, nt_emb, onehot, icshape, ic_measured, feat1d, mask, **kwargs) -> torch.Tensor:
        raw_ic = torch.where(ic_measured.bool(), icshape, torch.full_like(icshape, -1.0))
        feats = torch.cat([onehot.float(), raw_ic.unsqueeze(-1)], dim=-1)
        feats = feats * mask.unsqueeze(-1).float()
        x = feats.unsqueeze(1)
        if self.mode == "seq":
            x = x[:, :, :, :4]
        elif self.mode == "str":
            x = x[:, :, :, 4:]
        return self._forward_image(x)

    def _forward_image(self, x):
        """Shared PrismNet convolution/SE/residual/pooling stack."""
        x = self.conv(x)
        x = F.dropout(x, 0.1, training=self.training)
        z = self.se(x)
        x = self.res2d(x * z)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.res1d(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.gpool(x).view(x.shape[0], -1)
        return self.fc(x).squeeze(-1)


class EmbeddingPrismNetHead(PrismNetHead):
    """PrismNet stack on [one-hot sequence | nucleotide embedding] windows of fixed length (no icSHAPE input).
    Whole-RNA [B, D] vectors are rejected.
    """

    def __init__(self, embed_dim: int, base_channel: int = 8,
                 expected_length: int = 101):
        if embed_dim < 1 or base_channel < 2 or expected_length < 1:
            raise ValueError("embed_dim/length must be positive; base_channel >= 2")
        super().__init__(mode="pu", base_channel=base_channel)
        self.embed_dim = embed_dim
        self.expected_length = expected_length
        self.n_features = embed_dim + 4
        self.avgpool = nn.AvgPool2d((1, self.n_features))
        self.use_struct = True

    def forward(self, nt_emb, onehot, icshape, ic_measured, feat1d, mask,
                **kwargs) -> torch.Tensor:
        if nt_emb is None or nt_emb.ndim != 3:
            raise ValueError("embedding_prismnet requires nucleotide embeddings "
                             "[B,L,D], not a whole-RNA [B,D] vector")
        if onehot.ndim != 3 or onehot.shape[-1] != 4:
            raise ValueError("onehot must have shape [B,L,4]")
        B, L, _ = onehot.shape
        if (nt_emb.shape[0] != B or nt_emb.shape[1] < L
                or nt_emb.shape[2] != self.embed_dim):
            raise ValueError(f"Expected embeddings [B={B},L>={L},D={self.embed_dim}], "
                             f"got {tuple(nt_emb.shape)}")
        if L != self.expected_length or mask.shape != (B, L) or not bool(mask.bool().all()):
            raise ValueError("embedding_prismnet expects fixed-length, fully valid "
                             f"{self.expected_length}-nt windows (no padded positions)")
        feats = torch.cat([onehot.float(), nt_emb[:, :L, :].float()], dim=-1)
        if not bool(torch.isfinite(feats).all()):
            raise ValueError("Non-finite sequence/embedding features")
        return self._forward_image(feats.unsqueeze(1))


class RNAEmbeddingMLPHead(_RBPHeadBase):
    """Mean-pooled RNA representation followed by a two-layer MLP.

    The pooled representation is not a native encoder CLS/global token.
    The mask controls valid sequence positions; other feature arguments are
    accepted for runner compatibility but never used by this head.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        if embed_dim < 1 or hidden_dim < 1:
            raise ValueError("Embedding and hidden dimensions must be positive")
        self.embed_dim = embed_dim
        self.use_struct = True
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def pool_rna(self, nt_emb, mask):
        if nt_emb is None or nt_emb.ndim != 3 or mask.ndim != 2:
            raise ValueError("MLP route expects nucleotide embeddings [B,L,D] and mask [B,L]")
        B, L = mask.shape
        if nt_emb.shape[0] != B or nt_emb.shape[1] < L or nt_emb.shape[2] != self.embed_dim:
            raise ValueError(f"Expected nucleotide embeddings [B={B},L>={L},D={self.embed_dim}]")
        valid = mask.bool()
        if not bool(valid.any(dim=1).all()):
            raise ValueError("Cannot mean-pool an RNA with no valid nucleotides")
        values = nt_emb[:, :L, :].float().masked_fill(~valid.unsqueeze(-1), 0.0)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Non-finite embeddings at valid RNA positions")
        return values.sum(dim=1) / valid.sum(dim=1, keepdim=True).to(values.dtype)

    def forward(self, nt_emb, onehot, icshape, ic_measured, feat1d, mask,
                **kwargs):
        return self.classifier(self.pool_rna(nt_emb, mask)).squeeze(-1)
