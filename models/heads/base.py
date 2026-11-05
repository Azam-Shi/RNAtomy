"""Base class for the task heads. A head implements forward() (raw predictions), compute_loss() and
compute_metrics(); it consumes pooled [B, D], token [B, N, D] or per-nucleotide [B, L, D] encoder outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class FinetuneHead(ABC, nn.Module):
    """
    Base class.  Subclass this in models/heads/<task>.py for each task.
    """

    @abstractmethod
    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        features: the embedding tensor from RNAEncoder (pooled / tokens / nt).
        Returns raw predictions (before loss).
        """

    @abstractmethod
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Returns a scalar loss tensor."""

    @abstractmethod
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Returns a dict of metric name → value for logging.
        Called inside torch.no_grad() during validation.
        """


    def on_train_epoch_start(self, epoch: int) -> None:
        """Called at the beginning of each training epoch."""

    def on_val_epoch_end(self, metrics: Dict[str, float]) -> None:
        """Called after each validation epoch with the aggregated metrics."""


    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
