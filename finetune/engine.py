"""Train / validation loops shared by the fine-tuning scripts.

Batches are (samples, targets, valid_mask) or (samples, targets, valid_mask, nt_features, nt_valid).
"""

from __future__ import annotations

import math
import sys
from contextlib import suppress
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

import utils
from models.rna_encoder import RNAEncoder
from models.heads.base import FinetuneHead


def train_one_epoch(
    encoder: RNAEncoder,
    head: FinetuneHead,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    *,
    clip_grad: float = 1.0,
    accum_iter: int = 1,
    encode_mode: str = "pooled",
    seq_len: Optional[int] = None,
    amp_autocast=None,
    args=None,
) -> Dict[str, float]:
    encoder.train()
    head.train()
    head.on_train_epoch_start(epoch)
    if amp_autocast is None:
        amp_autocast = suppress()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Finetune Epoch [{epoch}]"
    print_freq = 10

    optimizer.zero_grad()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets, valid_mask, nt_features, nt_valid_mask = _unpack_batch(batch, device)

        with amp_autocast:
            features = _encode(
                encoder, encode_mode,
                samples, valid_mask, nt_features, nt_valid_mask, seq_len,
            )
            predictions = head(features,
                               valid_mask=valid_mask,
                               nt_features=nt_features,
                               nt_valid_mask=nt_valid_mask)
            loss = head.compute_loss(predictions, targets,
                                     valid_mask=valid_mask,
                                     nt_valid_mask=nt_valid_mask)
            aux_w = getattr(args, "aux_recon_1d_weight", 0.0) if args is not None else 0.0
            if aux_w > 0.0 and nt_features is not None:
                from finetune.utils import aux_1d_recon_loss
                loss = loss + aux_1d_recon_loss(
                    encoder, samples, nt_features, nt_valid_mask, aux_w,
                )

        if not math.isfinite(loss.item()):
            print(f"[FINETUNE] Non-finite loss at epoch={epoch} step={step}: {loss.item()}",
                  file=sys.stderr, flush=True)
            raise FloatingPointError("Non-finite finetune loss")

        (loss / accum_iter).backward()

        if (step + 1) % accum_iter == 0:
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()),
                    clip_grad,
                )
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Finetune train stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(
    encoder: RNAEncoder,
    head: FinetuneHead,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    *,
    encode_mode: str = "pooled",
    seq_len: Optional[int] = None,
    amp_autocast=None,
) -> Dict[str, float]:
    encoder.eval()
    head.eval()
    if amp_autocast is None:
        amp_autocast = suppress()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Finetune Val [{epoch}]"
    print_freq = 10

    all_preds, all_targets = [], []

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        samples, targets, valid_mask, nt_features, nt_valid_mask = _unpack_batch(batch, device)

        with amp_autocast:
            features = _encode(
                encoder, encode_mode,
                samples, valid_mask, nt_features, nt_valid_mask, seq_len,
            )
            predictions = head(features,
                               valid_mask=valid_mask,
                               nt_features=nt_features,
                               nt_valid_mask=nt_valid_mask)
            loss = head.compute_loss(predictions, targets,
                                     valid_mask=valid_mask,
                                     nt_valid_mask=nt_valid_mask)

        metric_logger.update(loss=loss.item())
        all_preds.append(predictions.cpu())
        all_targets.append(targets.cpu())

    preds_cat = torch.cat(all_preds, dim=0)
    tgts_cat = torch.cat(all_targets, dim=0)
    metrics = head.compute_metrics(preds_cat, tgts_cat)
    head.on_val_epoch_end(metrics)

    metric_logger.synchronize_between_processes()
    print("Finetune val stats:", metric_logger)
    result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    result.update(metrics)
    return result


def _unpack_batch(batch, device):
    """Unpack any batch format into a uniform 5-tuple."""
    if len(batch) == 5:
        samples, targets, valid_mask, nt_features, nt_valid_mask = batch
    elif len(batch) == 3:
        samples, targets, valid_mask = batch
        nt_features, nt_valid_mask = None, None
    else:
        samples, targets = batch
        valid_mask, nt_features, nt_valid_mask = None, None, None

    samples = samples.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if valid_mask is not None:
        valid_mask = valid_mask.to(device, non_blocking=True)
    if nt_features is not None:
        nt_features = nt_features.to(device, non_blocking=True)
    if nt_valid_mask is not None:
        nt_valid_mask = nt_valid_mask.to(device, non_blocking=True)

    return samples, targets, valid_mask, nt_features, nt_valid_mask


def _encode(encoder, mode, samples, valid_mask, nt_features, nt_valid_mask, seq_len):
    """Dispatch to the correct encoder output mode."""
    if mode == "pooled":
        return encoder.encode_pooled(samples, valid_mask, nt_features, nt_valid_mask)
    if mode == "tokens":
        tokens, patch_valid_mask, h, w = encoder.encode_tokens(
            samples, valid_mask, nt_features, nt_valid_mask
        )
        return tokens
    if mode == "nucleotides":
        nt_tokens, nt_valid = encoder.encode_nucleotides(
            samples, valid_mask, nt_features, nt_valid_mask, seq_len=seq_len
        )
        return nt_tokens
    raise ValueError(f"Unknown encode_mode '{mode}'. Choose: pooled | tokens | nucleotides")
