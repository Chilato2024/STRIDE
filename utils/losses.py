from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer('weight', weight if weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        logits = logits.reshape(-1, num_classes)
        labels = labels.reshape(-1)
        valid = (mask.reshape(-1) > 0) & (labels >= 0)
        if not valid.any():
            return logits.sum() * 0.0
        logits = logits[valid]
        labels = labels[valid]
        return F.cross_entropy(
            logits,
            labels,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )


class MaskedFocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer('weight', weight if weight is not None else None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        logits = logits.reshape(-1, num_classes)
        labels = labels.reshape(-1)
        valid = (mask.reshape(-1) > 0) & (labels >= 0)
        if not valid.any():
            return logits.sum() * 0.0
        logits = logits[valid]
        labels = labels[valid]

        ce = F.cross_entropy(
            logits,
            labels,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()
