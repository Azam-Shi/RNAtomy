import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """Pre-norm residual block that scans the token sequence in both directions."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2, dt_rank="auto")

    def forward(self, x, *args, **kwargs):
        y = self.norm(x)
        fwd = self.mamba(y)
        bwd = self.mamba(y.flip(1)).flip(1)
        return x + fwd + bwd


def _window_partition(x, window_size):
    bsz, height, width, channels = x.shape
    x = x.view(bsz, height // window_size, window_size, width // window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)


def _window_reverse(windows, window_size, height, width):
    bsz = int(windows.shape[0] / (height * width / (window_size * window_size)))
    x = windows.view(bsz, height // window_size, width // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(bsz, height, width, -1)


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class WindowAttention(nn.Module):
    """Multi-head self-attention inside local windows with a learned relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords = torch.flatten(coords, 1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _relative_position_bias(self, dtype, device):
        n = self.window_size * self.window_size
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.view(n, n, self.num_heads).permute(2, 0, 1).contiguous()
        return bias.unsqueeze(0).to(device=device, dtype=dtype)

    def forward(self, x, key_padding_mask=None):
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_bias = self._relative_position_bias(dtype=q.dtype, device=q.device)
        if key_padding_mask is not None:
            attn_bias = attn_bias + key_padding_mask.unsqueeze(1).unsqueeze(2).to(dtype=q.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=False,
        )
        x = x.transpose(1, 2).reshape(bsz, num_tokens, channels)
        return self.proj_drop(self.proj(x))


class WindowedTransformerBlock(nn.Module):
    """Window attention over the patch grid; `shift_size > 0` gives the shifted-window variant."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.0, qkv_bias=True, drop=0.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias=qkv_bias, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x, h_patches, w_patches):
        bsz, num_tokens, channels = x.shape
        if num_tokens != h_patches * w_patches:
            raise ValueError(f"Expected N={h_patches * w_patches} tokens, got N={num_tokens}")
        x = x.view(bsz, h_patches, w_patches, channels)

        pad_h = (self.window_size - h_patches % self.window_size) % self.window_size
        pad_w = (self.window_size - w_patches % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = x.shape[1], x.shape[2]

        # Real tokens must not attend to the zero tokens added by the window padding.
        key_padding_mask = None
        if pad_h or pad_w:
            valid_map = x.new_zeros(bsz, padded_h, padded_w, 1)
            valid_map[:, :h_patches, :w_patches, :] = 1.0
            if self.shift_size > 0:
                valid_map = torch.roll(valid_map, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            valid_wins = _window_partition(valid_map, self.window_size).squeeze(-1)
            key_padding_mask = torch.where(
                valid_wins > 0.5,
                valid_wins.new_zeros(valid_wins.shape),
                valid_wins.new_full(valid_wins.shape, float("-inf")),
            )

        shortcut = x
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = self.norm1(_window_partition(x, self.window_size))
        attn_windows = self.attn(windows, key_padding_mask=key_padding_mask)
        x = _window_reverse(attn_windows, self.window_size, padded_h, padded_w)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x
        if pad_h or pad_w:
            x = x[:, :h_patches, :w_patches, :]
        x = x + self.mlp(self.norm2(x))
        return x.view(bsz, h_patches * w_patches, channels)


class Feat1DEncoder(nn.Module):
    """Per-nucleotide encoder for the residue-level descriptors: [B, L, F] -> [B, L, dim]."""

    def __init__(self, in_dim, dim, n_conv=2, kernel=5):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, dim)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_conv)])
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim),
                nn.GELU(),
                nn.Conv1d(dim, dim, 1),
            )
            for _ in range(n_conv)
        ])
        self.out = nn.Linear(dim, dim)

    def forward(self, feat, valid_mask):
        m = valid_mask.unsqueeze(-1).to(feat.dtype)
        x = self.proj(self.in_norm(feat)) * m
        for norm, conv in zip(self.norms, self.convs):
            h = conv(norm(x).transpose(1, 2)).transpose(1, 2)
            x = x + h * m
        return self.out(x) * m


class HybridNet(nn.Module):
    """Hybrid Mamba / shifted-window-attention encoder over the pairwise patch grid.

    The residue-level descriptors modulate the patch tokens with axis-aware FiLM:
        z'_ij = z_ij * (1 + gamma_row_i + gamma_col_j) + beta_row_i + beta_col_j
    The FiLM generators start near zero, so training begins from the pairwise-only model.
    """

    ANGLE_DIM, CONT_DIM, BIN_DIM = 14, 2, 7

    def __init__(self, patch_size=16, in_chans=4, embed_dim=768, depth=16, mamba_ratio=3,
                 window_size=8, window_num_heads=8,
                 use_1d_conditioning=False, nt_feat_dim=23, feat1d_cond_dim=256, feat1d_encoder_layers=2,
                 recon_1d_weight=0.5, recon_1d_mask_ratio=0.15, recon_1d_span=4,
                 moddrop_prob=0.3, moddrop_recon_weight=0.5,
                 max_seq_len=1088, contact_alpha=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.window_size = window_size
        self.window_num_heads = window_num_heads
        self.use_1d_conditioning = bool(use_1d_conditioning)
        self.nt_feat_dim = int(nt_feat_dim)
        self.feat1d_cond_dim = int(feat1d_cond_dim)
        self.feat1d_encoder_layers = int(feat1d_encoder_layers)
        self.recon_1d_weight = float(recon_1d_weight)
        self.recon_1d_mask_ratio = float(recon_1d_mask_ratio)
        self.recon_1d_span = int(recon_1d_span)
        self.moddrop_prob = float(moddrop_prob)
        self.moddrop_recon_weight = float(moddrop_recon_weight)
        self.contact_alpha = float(contact_alpha)

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Row and column position tables: the patch at grid cell (r, c) gets row_pos[r] + col_pos[c].
        self.max_grid = int(math.ceil(max_seq_len / patch_size)) + 1
        self.row_pos_embed = nn.Parameter(torch.zeros(1, self.max_grid, embed_dim))
        self.col_pos_embed = nn.Parameter(torch.zeros(1, self.max_grid, embed_dim))
        nn.init.trunc_normal_(self.row_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.col_pos_embed, std=0.02)

        blocks, n_attn = [], 0
        for i in range(depth):
            if (i + 1) % (mamba_ratio + 1) == 0:
                shift = 0 if n_attn % 2 == 0 else window_size // 2
                blocks.append(WindowedTransformerBlock(
                    embed_dim, num_heads=window_num_heads, window_size=window_size, shift_size=shift))
                n_attn += 1
            else:
                blocks.append(MambaBlock(embed_dim))
        self.blocks = nn.Sequential(*blocks)

        self.norm = nn.LayerNorm(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.recon_head = nn.Linear(embed_dim, patch_size * patch_size * in_chans)

        # Descriptor layout: 14 sin/cos angle values, 2 continuous values, 7 binary flags.
        self.cont_start = self.ANGLE_DIM
        self.bin_start = self.ANGLE_DIM + self.CONT_DIM
        if self.use_1d_conditioning:
            if self.nt_feat_dim < self.bin_start + self.BIN_DIM:
                raise ValueError(f"nt_feat_dim={self.nt_feat_dim} is smaller than the "
                                 f"{self.bin_start + self.BIN_DIM}-d descriptor layout")
            dc = self.feat1d_cond_dim
            self.feat1d_encoder = Feat1DEncoder(self.nt_feat_dim, dc, n_conv=self.feat1d_encoder_layers)
            self.film_row = nn.Linear(dc, 2 * embed_dim)
            self.film_col = nn.Linear(dc, 2 * embed_dim)
            for m in (self.film_row, self.film_col):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.zeros_(m.bias)
            self.no_1d_token = nn.Parameter(torch.zeros(embed_dim))
            if self.recon_1d_weight > 0.0:
                self.nt_mask_vec = nn.Parameter(torch.zeros(self.nt_feat_dim))
                nn.init.trunc_normal_(self.nt_mask_vec, std=0.02)
                self.recon_1d_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, dc),
                    nn.GELU(),
                    nn.Linear(dc, self.nt_feat_dim),
                )

    # ------------------------------------------------------------------ helpers
    def _off_diagonal_pixel_mask(self, h_patches, w_patches, device, dtype):
        """[N, P*P*C] mask that is 0 on the main diagonal (i == j) and 1 elsewhere."""
        P, C = self.patch_size, self.in_chans
        pm = torch.ones(h_patches * w_patches, P, P, C, device=device, dtype=dtype)
        a = torch.arange(P, device=device)
        for p in range(h_patches):
            for q in range(max(0, p - 1), min(w_patches, p + 2)):
                gi = (p * P + a).view(P, 1)
                gj = (q * P + a).view(1, P)
                keep = ((gi - gj).abs() >= 1).to(dtype)
                pm[p * w_patches + q] = keep[:, :, None].expand(P, P, C)
        return pm.reshape(h_patches * w_patches, P * P * C)

    def _get_pos_embed(self, h_patches, w_patches):
        if h_patches > self.max_grid or w_patches > self.max_grid:
            raise ValueError(f"Patch grid ({h_patches}, {w_patches}) exceeds max_grid={self.max_grid}; "
                             "raise max_seq_len")
        r = self.row_pos_embed[:, :h_patches, :]
        c = self.col_pos_embed[:, :w_patches, :]
        return (r.unsqueeze(2) + c.unsqueeze(1)).reshape(1, h_patches * w_patches, -1)

    def _patchify_pixels(self, x):
        bsz, channels, height, width = x.shape
        p = self.patch_size
        if height % p != 0 or width % p != 0:
            raise ValueError(f"Input size {(height, width)} must be divisible by patch size {p}")
        h_patches, w_patches = height // p, width // p
        x = x.reshape(bsz, channels, h_patches, p, w_patches, p)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(bsz, h_patches * w_patches, p * p * channels)

    def _patch_valid_mask(self, valid_mask, h_patches, w_patches, device, batch_size):
        if valid_mask is None:
            return torch.ones((batch_size, h_patches * w_patches), dtype=torch.bool, device=device)
        pooled = F.avg_pool2d(valid_mask.unsqueeze(1).float(), kernel_size=self.patch_size, stride=self.patch_size)
        return (pooled.squeeze(1) > 0.0).reshape(batch_size, h_patches * w_patches)

    def _check_nt_features(self, nt_features, nt_valid_mask):
        if nt_features is None:
            return None
        if nt_features.ndim != 3 or nt_features.shape[-1] != self.nt_feat_dim:
            raise ValueError(f"Expected nt_features [B, L, {self.nt_feat_dim}], got {tuple(nt_features.shape)}")
        if nt_valid_mask is None:
            nt_valid_mask = torch.ones(nt_features.shape[:2], dtype=torch.bool, device=nt_features.device)
        if nt_valid_mask.ndim != 2:
            raise ValueError(f"Expected nt_valid_mask [B, L], got {tuple(nt_valid_mask.shape)}")
        return nt_valid_mask

    def _pool_nt_to_patches(self, nt_emb, nt_valid_mask, n_patches):
        """Masked mean of per-nucleotide embeddings over patch-sized segments: [B, L, D] -> [B, n_patches, D]."""
        bsz, length, dc = nt_emb.shape
        pooled = nt_emb.new_zeros((bsz, n_patches, dc))
        vm = nt_valid_mask.unsqueeze(-1).to(nt_emb.dtype)
        P = self.patch_size
        for idx in range(n_patches):
            start = idx * P
            if start >= length:
                break
            end = min((idx + 1) * P, length)
            seg = nt_emb[:, start:end, :] * vm[:, start:end, :]
            denom = vm[:, start:end, :].sum(dim=1).clamp(min=1.0)
            pooled[:, idx, :] = seg.sum(dim=1) / denom
        return pooled

    def _film_params(self, nt_emb, nt_valid_mask, h_patches, w_patches):
        row_emb = self._pool_nt_to_patches(nt_emb, nt_valid_mask, h_patches)
        col_emb = self._pool_nt_to_patches(nt_emb, nt_valid_mask, w_patches)
        g_row, b_row = self.film_row(row_emb).chunk(2, dim=-1)
        g_col, b_col = self.film_col(col_emb).chunk(2, dim=-1)
        gamma = g_row.unsqueeze(2) + g_col.unsqueeze(1)
        beta = b_row.unsqueeze(2) + b_col.unsqueeze(1)
        return gamma, beta

    def _sample_nt_span_mask(self, nt_valid_mask):
        """[B, L] bool mask covering ~recon_1d_mask_ratio of the valid nucleotides in contiguous spans."""
        B, L = nt_valid_mask.shape
        span = max(1, self.recon_1d_span)
        out = torch.zeros_like(nt_valid_mask)
        for b in range(B):
            idx = nt_valid_mask[b].nonzero(as_tuple=False).flatten()
            n = int(idx.numel())
            if n == 0:
                continue
            target = max(span, int(round(self.recon_1d_mask_ratio * n)))
            got, guard = 0, 0
            while got < target and guard < 4 * n:
                guard += 1
                s = int(idx[torch.randint(0, n, (1,), device=idx.device)].item())
                e = min(L, s + span)
                got += int((~out[b, s:e] & nt_valid_mask[b, s:e]).sum().item())
                out[b, s:e] = True
        return out & nt_valid_mask

    def _nt_recon_loss(self, pred, target, weight):
        """Cosine loss on the sin/cos angles, Huber on continuous values, BCE on binary flags."""
        a, c, bstart, bdim = self.ANGLE_DIM, self.CONT_DIM, self.bin_start, self.BIN_DIM
        w = weight.to(pred.dtype)
        denom = w.sum().clamp(min=1.0)

        p_ang = pred[..., :a].reshape(*pred.shape[:2], a // 2, 2)
        t_ang = target[..., :a].reshape(*target.shape[:2], a // 2, 2)
        p_ang = p_ang / p_ang.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        t_ang = t_ang / t_ang.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ang = (1.0 - (p_ang * t_ang).sum(dim=-1)).mean(dim=-1)

        cont = F.huber_loss(pred[..., a:a + c], target[..., a:a + c], reduction="none", delta=1.0).mean(dim=-1)
        binr = F.binary_cross_entropy_with_logits(
            pred[..., bstart:bstart + bdim], target[..., bstart:bstart + bdim], reduction="none"
        ).mean(dim=-1)

        loss = ((ang + cont + binr) * w).sum() / denom
        parts = {
            "recon1d_ang": float(((ang * w).sum() / denom).detach()),
            "recon1d_cont": float(((cont * w).sum() / denom).detach()),
            "recon1d_bin": float(((binr * w).sum() / denom).detach()),
        }
        return loss, parts

    def _sample_patch_mask(self, patch_valid_mask, mask_ratio):
        batch_size, n_patches = patch_valid_mask.shape
        patch_mask = torch.zeros((batch_size, n_patches), dtype=torch.bool, device=patch_valid_mask.device)
        for b in range(batch_size):
            valid_idx = torch.nonzero(patch_valid_mask[b], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            n_mask = max(1, min(valid_idx.numel(), int(round(valid_idx.numel() * mask_ratio))))
            perm = torch.randperm(valid_idx.numel(), device=patch_valid_mask.device)
            patch_mask[b, valid_idx[perm[:n_mask]]] = True
        return patch_mask

    def _readout_1d(self, tokens, patch_valid_mask, h_patches, w_patches, L):
        """Per-nucleotide descriptor prediction: masked mean over the column axis, then interpolate to L."""
        B, _, D = tokens.shape
        xr = tokens.view(B, h_patches, w_patches, D)
        pvm = patch_valid_mask.view(B, h_patches, w_patches, 1).to(tokens.dtype)
        row_repr = (xr * pvm).sum(dim=2) / pvm.sum(dim=2).clamp(min=1.0)
        row_repr = F.interpolate(row_repr.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(1, 2)
        return self.recon_1d_head(row_repr)

    # ------------------------------------------------------------------ forward
    def forward_features(self, x, valid_mask=None, patch_mask=None, return_tokens=False,
                         nt_features=None, nt_valid_mask=None, mod_keep=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        _, _, h_patches, w_patches = x.shape
        x = x.flatten(2).transpose(1, 2)

        patch_valid_mask = self._patch_valid_mask(valid_mask, h_patches, w_patches, x.device, B)
        if patch_mask is not None:
            effective_mask = patch_mask & patch_valid_mask
            x = torch.where(effective_mask.unsqueeze(-1), self.mask_token.expand(B, x.shape[1], -1), x)

        x = x + self._get_pos_embed(h_patches, w_patches)

        if self.use_1d_conditioning:
            if nt_features is None:
                x = x + self.no_1d_token.view(1, 1, -1)
            else:
                nt_valid_mask = self._check_nt_features(nt_features, nt_valid_mask)
                keep = x.new_ones(B, 1, 1) if mod_keep is None else mod_keep.to(x.dtype).view(B, 1, 1)
                nt_emb = self.feat1d_encoder(nt_features, nt_valid_mask) * keep
                gamma, beta = self._film_params(nt_emb, nt_valid_mask, h_patches, w_patches)
                x = x.view(B, h_patches, w_patches, -1)
                x = x * (1.0 + gamma) + beta
                x = x.reshape(B, h_patches * w_patches, -1)
                x = x + (1.0 - keep) * self.no_1d_token.view(1, 1, -1)

        for blk in self.blocks:
            x = blk(x, h_patches, w_patches)
        x = self.norm(x)

        if return_tokens:
            return x, patch_valid_mask

        pooling_mask = patch_valid_mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * pooling_mask).sum(dim=1) / pooling_mask.sum(dim=1).clamp(min=1.0)

    def reconstruction_loss(self, x, valid_mask=None, mask_ratio=0.5, nt_features=None, nt_valid_mask=None):
        _, _, height, width = x.shape
        h_patches, w_patches = height // self.patch_size, width // self.patch_size
        B = x.shape[0]
        patch_valid_mask = self._patch_valid_mask(valid_mask, h_patches, w_patches, x.device, B)
        patch_mask = self._sample_patch_mask(patch_valid_mask, mask_ratio)

        has_1d = self.use_1d_conditioning and nt_features is not None

        mod_keep = None
        if has_1d and self.training and self.moddrop_prob > 0.0:
            mod_keep = torch.rand(B, device=x.device) >= self.moddrop_prob

        do_1d_recon = has_1d and self.recon_1d_weight > 0.0 and self.training
        nt_features_in, nt_span_mask = nt_features, None
        if do_1d_recon:
            nt_valid_mask = self._check_nt_features(nt_features, nt_valid_mask)
            nt_span_mask = self._sample_nt_span_mask(nt_valid_mask)
            nt_features_in = torch.where(
                nt_span_mask.unsqueeze(-1), self.nt_mask_vec.to(nt_features.dtype).view(1, 1, -1), nt_features)

        tokens, patch_valid_mask = self.forward_features(
            x, valid_mask=valid_mask, patch_mask=patch_mask, return_tokens=True,
            nt_features=nt_features_in, nt_valid_mask=nt_valid_mask, mod_keep=mod_keep)
        pred_patches = self.recon_head(tokens)
        target_patches = self._patchify_pixels(x)

        pix_mask = self._off_diagonal_pixel_mask(h_patches, w_patches, x.device, target_patches.dtype)
        pix_denom = pix_mask.sum(dim=-1).clamp(min=1.0)
        per_pix = (pred_patches - target_patches).pow(2)
        patch_loss = (per_pix * pix_mask).sum(dim=-1) / pix_denom

        effective_mask = (patch_mask & patch_valid_mask).to(dtype=patch_loss.dtype)
        denom = effective_mask.sum(dim=1).clamp(min=1.0)

        if self.contact_alpha > 0.0:
            # Up-weight patches with a strong geometric signal (RMS of the target values).
            with torch.no_grad():
                target_rms = ((target_patches.pow(2) * pix_mask).sum(dim=-1) / pix_denom).sqrt()
                raw_w = 1.0 + self.contact_alpha * target_rms
                w_sum = (raw_w * effective_mask).sum(dim=1, keepdim=True).clamp(min=1.0)
                patch_weights = raw_w * denom.unsqueeze(1) / w_sum
            sample_loss = (patch_loss * effective_mask * patch_weights).sum(dim=1) / denom
        else:
            sample_loss = (patch_loss * effective_mask).sum(dim=1) / denom

        loss_2d = sample_loss.mean()
        components = {"recon2d": float(loss_2d.detach())}
        total = loss_2d

        if do_1d_recon:
            pred_1d = self._readout_1d(tokens, patch_valid_mask, h_patches, w_patches, nt_features.shape[1])
            # Kept samples score only the masked spans; modality-dropped samples score every valid
            # nucleotide (2D -> 1D prediction), scaled by moddrop_recon_weight.
            weight = (nt_span_mask & nt_valid_mask).to(pred_1d.dtype)
            if mod_keep is not None:
                drop = (~mod_keep).to(pred_1d.dtype).view(B, 1)
                weight = weight * (1.0 - drop) + nt_valid_mask.to(pred_1d.dtype) * drop * self.moddrop_recon_weight
            loss_1d, parts_1d = self._nt_recon_loss(pred_1d, nt_features, weight)
            total = loss_2d + self.recon_1d_weight * loss_1d
            components["recon1d"] = float(loss_1d.detach())
            components["recon1d_frac_scored"] = float(
                ((weight > 0).sum() / nt_valid_mask.sum().clamp(min=1)).detach())
            if mod_keep is not None:
                components["moddrop_frac"] = float((~mod_keep).float().mean().detach())
            components.update(parts_1d)

        return total, components

    def aux_1d_recon_loss(self, x, valid_mask=None, nt_features=None, nt_valid_mask=None):
        """Descriptor-reconstruction loss alone, used as an auxiliary objective during fine-tuning.
        Returns None if the checkpoint has no reconstruction head."""
        if not self.use_1d_conditioning or not hasattr(self, "recon_1d_head") or nt_features is None:
            return None
        nt_valid_mask = self._check_nt_features(nt_features, nt_valid_mask)
        _, _, height, width = x.shape
        h_patches, w_patches = height // self.patch_size, width // self.patch_size
        nt_span_mask = self._sample_nt_span_mask(nt_valid_mask)
        nt_in = torch.where(
            nt_span_mask.unsqueeze(-1), self.nt_mask_vec.to(nt_features.dtype).view(1, 1, -1), nt_features)
        tokens, patch_valid_mask = self.forward_features(
            x, valid_mask=valid_mask, return_tokens=True, nt_features=nt_in, nt_valid_mask=nt_valid_mask)
        pred_1d = self._readout_1d(tokens, patch_valid_mask, h_patches, w_patches, nt_features.shape[1])
        weight = (nt_span_mask & nt_valid_mask).to(pred_1d.dtype)
        loss_1d, _ = self._nt_recon_loss(pred_1d, nt_features, weight)
        return loss_1d

    def forward(self, x, valid_mask=None, mask_ratio=0.5, compute_loss=False,
                nt_features=None, nt_valid_mask=None, return_components=False):
        if compute_loss:
            loss, components = self.reconstruction_loss(
                x, valid_mask=valid_mask, mask_ratio=mask_ratio,
                nt_features=nt_features, nt_valid_mask=nt_valid_mask)
            return (loss, components) if return_components else loss
        return self.forward_features(x, valid_mask=valid_mask, nt_features=nt_features,
                                     nt_valid_mask=nt_valid_mask)
