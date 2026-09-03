from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )


class DepthwiseRefinement(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
        )


class BoundaryAwareSegFormer(nn.Module):
    """One SegFormer encoder with a lightweight boundary-gated local decoder."""

    def __init__(
        self,
        encoder: SegformerModel,
        num_classes: int = 9,
        decoder_channels: int = 192,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.decoder_channels = decoder_channels
        hidden_sizes = list(encoder.config.hidden_sizes)
        self.projections = nn.ModuleList(
            ConvNormAct(channels, decoder_channels) for channels in hidden_sizes
        )
        self.fuse = ConvNormAct(decoder_channels * len(hidden_sizes), decoder_channels, 1)
        self.refine = DepthwiseRefinement(decoder_channels)
        self.boundary_head = nn.Sequential(
            ConvNormAct(decoder_channels, decoder_channels // 2, 3),
            nn.Conv2d(decoder_channels // 2, 1, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, num_classes, 1),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_classes: int = 9,
        decoder_channels: int = 192,
        local_files_only: bool = False,
    ) -> "BoundaryAwareSegFormer":
        encoder = SegformerModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        return cls(encoder, num_classes=num_classes, decoder_channels=decoder_channels)

    @classmethod
    def from_model_config(cls, payload: dict[str, Any]) -> "BoundaryAwareSegFormer":
        config = SegformerConfig.from_dict(payload["encoder_config"])
        encoder = SegformerModel(config)
        return cls(
            encoder,
            num_classes=int(payload["num_classes"]),
            decoder_channels=int(payload["decoder_channels"]),
        )

    def export_config(self) -> dict[str, Any]:
        return {
            "encoder_config": self.encoder.config.to_dict(),
            "num_classes": self.num_classes,
            "decoder_channels": self.decoder_channels,
        }

    def gradient_checkpointing_enable(self) -> bool:
        if not getattr(self.encoder, "supports_gradient_checkpointing", False):
            return False
        self.encoder.gradient_checkpointing_enable()
        return True

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        target_size = hidden_states[0].shape[-2:]
        projected = []
        for feature, projection in zip(hidden_states, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=False
                )
            projected.append(feature)
        fused = self.fuse(torch.cat(projected, dim=1))
        boundary_logits = self.boundary_head(projected[0])
        boundary_gate = torch.sigmoid(boundary_logits)
        refined = self.refine(fused) * (1.0 + boundary_gate)
        logits = self.classifier(refined)
        return {"logits": logits, "boundary_logits": boundary_logits}
