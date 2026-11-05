"""RNAEncoder: wraps a pretrained HybridNet and exposes the embeddings used by the downstream tasks.

  encode_pooled()       [B, D]        whole-RNA embedding
  encode_tokens()       [B, N, D]     patch-token grid
  encode_pairs()        [B, L, L, D]  per-pair embedding at nucleotide resolution
  encode_nucleotides()  [B, L, D]     per-nucleotide embedding
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hybridnet import HybridNet


def build_encoder_from_checkpoint(
    checkpoint_path: str,
    *,
    patch_size: int = 8,
    in_chans: int = 4,
    embed_dim: int = 768,
    depth: int = 16,
    mamba_ratio: int = 3,
    window_size: int = 16,
    window_num_heads: int = 12,
    use_1d_conditioning: bool = True,
    nt_feat_dim: int = 23,
    feat1d_cond_dim: int = 256,
    feat1d_encoder_layers: int = 2,
    recon_1d_weight: float = 0.0,
    feat1d_mode: str = "full",
    feat1d_moddrop_prob: float = 0.5,
    feat1d_eval_mode: str = "full",
    contact_alpha: float = 0.0,
    use_ema: bool = True,
    device: str = "cpu",
) -> "RNAEncoder":
    """Build an RNAEncoder from a checkpoint written by pretrain.py (EMA weights when available)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state = None
    if isinstance(ckpt, dict) and use_ema and ckpt.get("model_ema") is not None:
        raw = ckpt["model_ema"]
        if isinstance(raw, dict):
            state = raw
        elif hasattr(raw, "state_dict"):
            state = raw.state_dict()
    if state is None:
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if hasattr(state, "state_dict"):
            state = state.state_dict()
    if not isinstance(state, dict):
        raise ValueError("checkpoint state_dict is not a dict-like object")

    row_pos_embed = state.get("row_pos_embed")
    col_pos_embed = state.get("col_pos_embed")
    if not torch.is_tensor(row_pos_embed) or not torch.is_tensor(col_pos_embed):
        raise ValueError(
            "checkpoint is missing row_pos_embed or col_pos_embed; "
            "cannot infer the pretrained positional-grid capacity"
        )
    if row_pos_embed.ndim != 3 or row_pos_embed.shape != col_pos_embed.shape:
        raise ValueError(
            "checkpoint row/column positional embeddings must have matching "
            f"3D shapes, got {tuple(row_pos_embed.shape)} and "
            f"{tuple(col_pos_embed.shape)}"
        )
    if int(row_pos_embed.shape[2]) != embed_dim:
        raise ValueError(
            "checkpoint positional embedding dimension does not match the "
            f"requested embed_dim: {row_pos_embed.shape[2]} != {embed_dim}"
        )

    max_grid = int(row_pos_embed.shape[1])
    if max_grid <= 1:
        raise ValueError(f"invalid checkpoint positional-grid size: {max_grid}")
    max_seq_len = (max_grid - 1) * patch_size

    backbone = HybridNet(
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        mamba_ratio=mamba_ratio,
        window_size=window_size,
        window_num_heads=window_num_heads,
        use_1d_conditioning=use_1d_conditioning,
        nt_feat_dim=nt_feat_dim,
        feat1d_cond_dim=feat1d_cond_dim,
        feat1d_encoder_layers=feat1d_encoder_layers,
        recon_1d_weight=float(recon_1d_weight),
        moddrop_prob=0.0,
        max_seq_len=max_seq_len,
        contact_alpha=contact_alpha,
    )

    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing:
        print(f"[RNAEncoder] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[RNAEncoder] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    if feat1d_mode != "full" or (feat1d_mode == "moddrop" and feat1d_eval_mode == "none"):
        tok = state.get("no_1d_token")
        tok_norm = float(tok.norm()) if hasattr(tok, "norm") else 0.0
        if tok_norm < 1e-3:
            print(
                f"[RNAEncoder] WARNING: feat1d_mode='{feat1d_mode}' uses the 2D-only path, but this "
                f"checkpoint's no_1d_token norm is {tok_norm:.4f} -- it was very likely pretrained "
                f"WITHOUT modality dropout (--moddrop-prob 0). 2D-only inference will be "
                f"out-of-distribution. Use a checkpoint pretrained with modality dropout."
            )

    return RNAEncoder(
        backbone, patch_size=patch_size,
        feat1d_mode=feat1d_mode, feat1d_moddrop_prob=feat1d_moddrop_prob,
        feat1d_eval_mode=feat1d_eval_mode,
    ).to(device)


class RNAEncoder(nn.Module):
    def __init__(self, backbone: HybridNet, patch_size: int = 16,
                 feat1d_mode: str = "full", feat1d_moddrop_prob: float = 0.5,
                 feat1d_eval_mode: str = "full"):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.embed_dim: int = backbone.patch_embed.out_channels
        # feat1d_mode: "full" always uses the residue descriptors, "none" runs the pairwise-only path,
        # "moddrop" drops the whole track per sample while training (feat1d_eval_mode applies at eval).
        if feat1d_mode not in ("full", "none", "moddrop"):
            raise ValueError(f"feat1d_mode must be full|none|moddrop, got {feat1d_mode}")
        if feat1d_eval_mode not in ("full", "none"):
            raise ValueError(f"feat1d_eval_mode must be full|none, got {feat1d_eval_mode}")
        self.feat1d_mode = feat1d_mode
        self.feat1d_moddrop_prob = float(feat1d_moddrop_prob)
        self.feat1d_eval_mode = feat1d_eval_mode

    def _resolve_feat1d(self, x, nt_features, nt_valid_mask):
        if not getattr(self.backbone, "use_1d_conditioning", False):
            return nt_features, nt_valid_mask, None
        mode = self.feat1d_mode
        if mode == "moddrop":
            mode = "moddrop" if self.training else self.feat1d_eval_mode
        if mode == "none":
            return None, None, None
        if mode == "moddrop" and nt_features is not None:
            keep = torch.rand(x.shape[0], device=x.device) >= self.feat1d_moddrop_prob
            return nt_features, nt_valid_mask, keep
        return nt_features, nt_valid_mask, None

    def freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze the last n blocks and the output LayerNorm."""
        self.freeze()
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        if n > 0:
            for blk in list(self.backbone.blocks)[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train(self, mode: bool = True):
        """Frozen blocks stay in eval mode when the wrapper is switched to train mode."""
        super().train(mode)
        if mode:
            for blk in self.backbone.blocks:
                if all(not p.requires_grad for p in blk.parameters()):
                    blk.eval()
            for name in ("patch_embed", "feat1d_encoder", "film_row", "film_col"):
                m = getattr(self.backbone, name, None)
                if m is not None and not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self

    def encode_pooled(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        nt_features: Optional[torch.Tensor] = None,
        nt_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        nt_features, nt_valid_mask, mod_keep = self._resolve_feat1d(x, nt_features, nt_valid_mask)
        return self.backbone.forward_features(
            x,
            valid_mask=valid_mask,
            nt_features=nt_features,
            nt_valid_mask=nt_valid_mask,
            mod_keep=mod_keep,
            return_tokens=False,
        )

    def encode_tokens(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        nt_features: Optional[torch.Tensor] = None,
        nt_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Returns (tokens [B, h*w, D], patch_valid_mask [B, h*w], h_patches, w_patches)."""
        h_patches = x.shape[2] // self.patch_size
        w_patches = x.shape[3] // self.patch_size
        nt_features, nt_valid_mask, mod_keep = self._resolve_feat1d(x, nt_features, nt_valid_mask)
        tokens, patch_valid_mask = self.backbone.forward_features(
            x,
            valid_mask=valid_mask,
            nt_features=nt_features,
            nt_valid_mask=nt_valid_mask,
            mod_keep=mod_keep,
            return_tokens=True,
        )
        return tokens, patch_valid_mask, h_patches, w_patches

    def encode_pairs(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        nt_features: Optional[torch.Tensor] = None,
        nt_valid_mask: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        seq_lens: Optional[list] = None,
    ) -> torch.Tensor:
        """Patch tokens reshaped to a grid and bilinearly upsampled to [B, L, L, D]."""
        tokens, _patch_valid_mask, h_patches, w_patches = self.encode_tokens(
            x, valid_mask, nt_features, nt_valid_mask
        )
        B, _, D = tokens.shape
        grid = tokens.view(B, h_patches, w_patches, D).permute(0, 3, 1, 2)
        if seq_lens is not None:
            target_len = int(max(int(v) for v in seq_lens))
        else:
            target_len = seq_len if seq_len is not None else h_patches * self.patch_size
        grid = F.interpolate(
            grid, size=(target_len, target_len),
            mode="bilinear", align_corners=False,
        )
        return grid.permute(0, 2, 3, 1).contiguous()

    def encode_nucleotides(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        nt_features: Optional[torch.Tensor] = None,
        nt_valid_mask: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        seq_lens: Optional[list] = None,
        include_diagonal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-nucleotide embeddings (nt_tokens [B, L, D] or [B, L, 2D] with include_diagonal, nt_valid [B, L]).

        Tokens are averaged over the column axis, optionally concatenated with the diagonal token, then
        interpolated from patch rows to nucleotides. `seq_lens` interpolates each sample from its own
        valid rows to its own length; `seq_len` uses one target length for the whole batch.
        """
        tokens, patch_valid_mask, h_patches, w_patches = self.encode_tokens(
            x, valid_mask, nt_features, nt_valid_mask
        )
        return self._patch_tokens_to_nucleotides(
            tokens, patch_valid_mask, h_patches, w_patches, seq_len,
            seq_lens=seq_lens, include_diagonal=include_diagonal,
        )

    def _patch_tokens_to_nucleotides(
        self,
        tokens: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        h_patches: int,
        w_patches: int,
        seq_len: Optional[int],
        seq_lens: Optional[list] = None,
        include_diagonal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = tokens.shape
        tokens_2d = tokens.view(B, h_patches, w_patches, D)
        mask_2d = patch_valid_mask.view(B, h_patches, w_patches)

        mask_f = mask_2d.float().unsqueeze(-1)
        denom = mask_f.sum(dim=2).clamp(min=1.0)
        row_tokens = (tokens_2d * mask_f).sum(dim=2) / denom
        row_valid = mask_2d.any(dim=2)

        if include_diagonal:
            g = min(h_patches, w_patches)
            gi = torch.arange(g, device=tokens.device)
            diag_tokens = tokens_2d.new_zeros(B, h_patches, D)
            diag_tokens[:, :g, :] = tokens_2d[:, gi, gi, :]
            row_repr = torch.cat([row_tokens, diag_tokens], dim=-1)
        else:
            row_repr = row_tokens

        out_d = row_repr.shape[-1]

        if seq_lens is not None:
            target_len = int(max(int(v) for v in seq_lens))
            nt_tokens = row_repr.new_zeros(B, target_len, out_d)
            nt_valid = torch.zeros(B, target_len, dtype=torch.bool, device=tokens.device)
            for b, Lb in enumerate(seq_lens):
                Lb = int(Lb)
                hb = int(row_valid[b].sum().item()) or 1
                if hb == 1:
                    nt_tokens[b, :Lb] = row_repr[b, 0].unsqueeze(0).expand(Lb, -1)
                else:
                    src = row_repr[b, :hb].permute(1, 0).unsqueeze(0)
                    up = F.interpolate(src, size=Lb, mode="linear", align_corners=False)
                    nt_tokens[b, :Lb] = up.squeeze(0).permute(1, 0)
                nt_valid[b, :Lb] = True
            return nt_tokens, nt_valid

        target_len = seq_len if seq_len is not None else h_patches * self.patch_size
        nt_tokens = F.interpolate(
            row_repr.permute(0, 2, 1),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)

        nt_valid = F.interpolate(
            row_valid.float().unsqueeze(1),
            size=target_len,
            mode="nearest",
        ).squeeze(1).bool()

        return nt_tokens, nt_valid
