from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
from typing import Any, Sequence

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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
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
            "architecture": "segformer_boundary",
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


DINOV3_VARIANTS: dict[str, dict[str, Any]] = {
    "vits16": {
        "builder": "dinov3_vits16",
        "embed_dim": 384,
        "depth": 12,
        "default_blocks": (2, 5, 8, 11),
        "sha256": "08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d",
    },
    "vits16plus": {
        "builder": "dinov3_vits16plus",
        "embed_dim": 384,
        "depth": 12,
        "default_blocks": (2, 5, 8, 11),
        "sha256": "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea",
    },
    "vitb16": {
        "builder": "dinov3_vitb16",
        "embed_dim": 768,
        "depth": 12,
        "default_blocks": (2, 5, 8, 11),
        "sha256": "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c",
    },
    "vitl16": {
        "builder": "dinov3_vitl16",
        "embed_dim": 1024,
        "depth": 24,
        "default_blocks": (5, 11, 17, 23),
        "sha256": "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035",
    },
}
DINOV3_SOURCE_REVISION = "6876159a11b4df116f30f667f8c9888617df0751"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dinov3_source_revision(source: str | Path) -> str | None:
    git_dir = Path(source).expanduser().resolve() / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="ascii").strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    loose = git_dir / reference
    if loose.is_file():
        return loose.read_text(encoding="ascii").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="ascii").splitlines():
            if line and not line.startswith(("#", "^")):
                revision, name = line.split(" ", 1)
                if name == reference:
                    return revision
    return None


def default_dinov3_blocks(variant: str) -> tuple[int, int, int, int]:
    try:
        return tuple(DINOV3_VARIANTS[variant]["default_blocks"])
    except KeyError as error:
        raise ValueError(
            f"Unsupported DINOv3 variant {variant!r}; choose from {sorted(DINOV3_VARIANTS)}"
        ) from error


def parse_dinov3_blocks(value: str | Sequence[int] | None, variant: str) -> tuple[int, ...]:
    if value is None:
        blocks = default_dinov3_blocks(variant)
    elif isinstance(value, str):
        try:
            blocks = tuple(int(part.strip()) for part in value.split(",") if part.strip())
        except ValueError as error:
            raise ValueError("DINOv3 blocks must be comma-separated integers") from error
    else:
        blocks = tuple(int(part) for part in value)
    depth = int(DINOV3_VARIANTS[variant]["depth"])
    if len(blocks) != 4 or len(set(blocks)) != 4:
        raise ValueError("Exactly four distinct DINOv3 feature blocks are required")
    if tuple(sorted(blocks)) != blocks or blocks[0] < 0 or blocks[-1] >= depth:
        raise ValueError(f"DINOv3 blocks must be increasing indices in [0,{depth - 1}]")
    return blocks


def _dinov3_backbones(source: str | Path | None):
    if source:
        source_path = Path(source).expanduser().resolve()
        expected = source_path / "dinov3" / "hub" / "backbones.py"
        if not expected.is_file():
            raise FileNotFoundError(
                f"DINOv3 source does not contain dinov3/hub/backbones.py: {source_path}"
            )
        source_text = str(source_path)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
    try:
        return importlib.import_module("dinov3.hub.backbones")
    except ImportError as error:
        raise ImportError(
            "DINOv3 source is unavailable. Pass --dinov3-source pointing to a pinned "
            "checkout of https://github.com/facebookresearch/dinov3."
        ) from error


def create_dinov3_encoder(variant: str, source: str | Path | None) -> nn.Module:
    if variant not in DINOV3_VARIANTS:
        raise ValueError(
            f"Unsupported DINOv3 variant {variant!r}; choose from {sorted(DINOV3_VARIANTS)}"
        )
    module = _dinov3_backbones(source)
    builder_name = str(DINOV3_VARIANTS[variant]["builder"])
    builder = getattr(module, builder_name, None)
    if builder is None:
        raise RuntimeError(f"Pinned DINOv3 source does not expose {builder_name}")
    return builder(pretrained=False)


class DinoV3BoundarySegmenter(nn.Module):
    """DINOv3 token backbone with dense multi-depth fusion and a local detail path.

    DINOv3's selected blocks all have a 1/16 token grid. A shallow RGB stem supplies
    true 1/4 spatial detail, while normalized intermediate blocks contribute increasing
    semantic depth. The result remains one end-to-end segmentation model.
    """

    def __init__(
        self,
        encoder: nn.Module,
        variant: str = "vits16plus",
        feature_blocks: Sequence[int] | None = None,
        num_classes: int = 9,
        decoder_channels: int = 192,
        dinov3_source: str | Path | None = None,
        pretrained_sha256: str | None = None,
        source_revision: str | None = None,
    ) -> None:
        super().__init__()
        if variant not in DINOV3_VARIANTS:
            raise ValueError(f"Unsupported DINOv3 variant: {variant}")
        if decoder_channels < 16:
            raise ValueError("decoder_channels must be at least 16")
        self.encoder = encoder
        self.variant = variant
        self.feature_blocks = parse_dinov3_blocks(feature_blocks, variant)
        self.num_classes = int(num_classes)
        self.decoder_channels = int(decoder_channels)
        self.dinov3_source = str(Path(dinov3_source).resolve()) if dinov3_source else None
        self.pretrained_sha256 = pretrained_sha256
        self.source_revision = source_revision

        embed_dim = int(DINOV3_VARIANTS[variant]["embed_dim"])
        self.projections = nn.ModuleList(
            ConvNormAct(embed_dim, decoder_channels) for _ in self.feature_blocks
        )
        detail_channels = max(16, decoder_channels // 2)
        self.detail_stem = nn.Sequential(
            ConvNormAct(3, detail_channels, 3, stride=2),
            ConvNormAct(detail_channels, decoder_channels, 3, stride=2),
        )
        self.boundary_fuse = ConvNormAct(decoder_channels * 2, decoder_channels, 3)
        self.fuse = ConvNormAct(decoder_channels * 5, decoder_channels)
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
        weights: str | Path,
        dinov3_source: str | Path,
        variant: str = "vits16plus",
        feature_blocks: Sequence[int] | None = None,
        num_classes: int = 9,
        decoder_channels: int = 192,
        verify_hash: bool = True,
    ) -> "DinoV3BoundarySegmenter":
        if variant not in DINOV3_VARIANTS:
            raise ValueError(
                f"Unsupported DINOv3 variant {variant!r}; choose from {sorted(DINOV3_VARIANTS)}"
            )
        weights_path = Path(weights).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
        actual_sha256 = sha256_file(weights_path)
        expected_sha256 = str(DINOV3_VARIANTS[variant]["sha256"])
        if verify_hash and actual_sha256 != expected_sha256:
            raise ValueError(
                f"DINOv3 weight SHA256 mismatch for {variant}: "
                f"expected {expected_sha256}, received {actual_sha256}"
            )
        revision = dinov3_source_revision(dinov3_source)
        if revision is not None and revision != DINOV3_SOURCE_REVISION:
            raise ValueError(
                "DINOv3 source revision mismatch: "
                f"expected {DINOV3_SOURCE_REVISION}, received {revision}"
            )
        encoder = create_dinov3_encoder(variant, dinov3_source)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if not isinstance(state, dict):
            raise TypeError("DINOv3 checkpoint must contain a state_dict")
        encoder.load_state_dict(state, strict=True)
        return cls(
            encoder,
            variant=variant,
            feature_blocks=feature_blocks,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
            dinov3_source=dinov3_source,
            pretrained_sha256=actual_sha256,
            source_revision=revision,
        )

    @classmethod
    def from_model_config(
        cls,
        payload: dict[str, Any],
        dinov3_source: str | Path | None = None,
    ) -> "DinoV3BoundarySegmenter":
        source = dinov3_source or payload.get("dinov3_source")
        revision = dinov3_source_revision(source) if source else None
        expected_revision = payload.get("source_revision")
        if revision is not None and expected_revision is not None and revision != expected_revision:
            raise ValueError(
                "DINOv3 source differs from the training checkpoint: "
                f"expected {expected_revision}, received {revision}"
            )
        encoder = create_dinov3_encoder(str(payload["variant"]), source)
        return cls(
            encoder,
            variant=str(payload["variant"]),
            feature_blocks=payload["feature_blocks"],
            num_classes=int(payload["num_classes"]),
            decoder_channels=int(payload["decoder_channels"]),
            dinov3_source=source,
            pretrained_sha256=payload.get("pretrained_sha256"),
            source_revision=revision or expected_revision,
        )

    def export_config(self) -> dict[str, Any]:
        return {
            "architecture": "dinov3_boundary",
            "variant": self.variant,
            "feature_blocks": list(self.feature_blocks),
            "num_classes": self.num_classes,
            "decoder_channels": self.decoder_channels,
            "dinov3_source": self.dinov3_source,
            "pretrained_sha256": self.pretrained_sha256,
            "source_revision": self.source_revision,
        }

    def gradient_checkpointing_enable(self) -> bool:
        return False

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = pixel_values.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError(
                f"DINOv3 inputs must be divisible by 16, received {height}x{width}"
            )
        features = self.encoder.get_intermediate_layers(
            pixel_values,
            n=self.feature_blocks,
            reshape=True,
            norm=True,
        )
        if len(features) != 4:
            raise RuntimeError(f"DINOv3 returned {len(features)} features instead of four")

        detail = self.detail_stem(pixel_values)
        target_size = detail.shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            feature = F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            projected.append(feature)

        boundary_features = self.boundary_fuse(torch.cat([detail, projected[0]], dim=1))
        boundary_logits = self.boundary_head(boundary_features)
        fused = self.fuse(torch.cat([detail, *projected], dim=1))
        boundary_gate = torch.sigmoid(boundary_logits)
        refined = self.refine(fused) * (1.0 + boundary_gate)
        logits = self.classifier(refined)
        return {"logits": logits, "boundary_logits": boundary_logits}


def model_from_config(
    payload: dict[str, Any],
    dinov3_source: str | Path | None = None,
) -> nn.Module:
    architecture = payload.get("architecture", "segformer_boundary")
    if architecture == "segformer_boundary":
        return BoundaryAwareSegFormer.from_model_config(payload)
    if architecture == "dinov3_boundary":
        return DinoV3BoundarySegmenter.from_model_config(
            payload, dinov3_source=dinov3_source
        )
    raise ValueError(f"Unsupported model architecture in checkpoint: {architecture!r}")
