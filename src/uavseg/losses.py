from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = 0, eps: float = 1e-6
) -> torch.Tensor:
    num_classes = logits.shape[1]
    probabilities = logits.softmax(dim=1)
    valid = labels != ignore_index
    safe_labels = labels.masked_fill(~valid, 0)
    targets = F.one_hot(safe_labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_mask = valid.unsqueeze(1)
    probabilities = probabilities * valid_mask
    targets = targets * valid_mask
    intersection = (probabilities * targets).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    class_ids = torch.arange(num_classes, device=logits.device)
    evaluated = (class_ids != ignore_index) & (targets.sum(dim=(0, 2, 3)) > 0)
    return 1.0 - dice[evaluated].mean() if evaluated.any() else logits.sum() * 0.0


def boundary_target(labels: torch.Tensor, ignore_index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    valid = labels != ignore_index
    edges = torch.zeros_like(labels, dtype=torch.bool)
    horizontal_valid = valid[:, :, 1:] & valid[:, :, :-1]
    vertical_valid = valid[:, 1:, :] & valid[:, :-1, :]
    horizontal_edge = (labels[:, :, 1:] != labels[:, :, :-1]) & horizontal_valid
    vertical_edge = (labels[:, 1:, :] != labels[:, :-1, :]) & vertical_valid
    edges[:, :, 1:] |= horizontal_edge
    edges[:, :, :-1] |= horizontal_edge
    edges[:, 1:, :] |= vertical_edge
    edges[:, :-1, :] |= vertical_edge
    edges = F.max_pool2d(edges.float().unsqueeze(1), kernel_size=3, stride=1, padding=1)
    return edges, valid.float().unsqueeze(1)


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor,
        ignore_index: int = 0,
        dice_weight: float = 0.3,
        boundary_weight: float = 0.1,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        boundary_logits = F.interpolate(
            boundary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        ce = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        dice = soft_dice_loss(logits, labels, ignore_index=self.ignore_index)
        target, valid = boundary_target(labels, ignore_index=self.ignore_index)
        positive = (target * valid).sum()
        negative = valid.sum() - positive
        positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
        boundary_map = F.binary_cross_entropy_with_logits(
            boundary_logits, target, reduction="none", pos_weight=positive_weight
        )
        boundary = (boundary_map * valid).sum() / valid.sum().clamp_min(1.0)
        total = ce + self.dice_weight * dice + self.boundary_weight * boundary
        parts = {
            "ce": float(ce.detach()),
            "dice": float(dice.detach()),
            "boundary": float(boundary.detach()),
        }
        return total, parts
