"""Verify one official DINOv3 backbone and the B-line segmentation forward pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uavseg.model import (
    DINOV3_SOURCE_REVISION,
    DINOV3_VARIANTS,
    DinoV3BoundarySegmenter,
    dinov3_source_revision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Official DINOv3 repository checkout")
    parser.add_argument("--weights", required=True, help="Official DINOv3 .pth backbone")
    parser.add_argument(
        "--variant", choices=tuple(DINOV3_VARIANTS), default="vits16plus"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--decoder-channels", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    if args.size <= 0 or args.size % 16:
        raise ValueError("--size must be a positive multiple of 16")

    revision = dinov3_source_revision(source)
    if revision is not None and revision != DINOV3_SOURCE_REVISION:
        raise ValueError(
            "DINOv3 source revision mismatch: "
            f"expected {DINOV3_SOURCE_REVISION}, received {revision}"
        )

    model = DinoV3BoundarySegmenter.from_pretrained(
        weights,
        dinov3_source=source,
        variant=args.variant,
        decoder_channels=args.decoder_channels,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model.to(device).eval()
    pixels = torch.zeros(1, 3, args.size, args.size, device=device)
    with torch.inference_mode():
        output = model(pixels)

    report = {
        "status": "ok",
        "source": str(source),
        "source_revision": revision,
        "expected_source_revision": DINOV3_SOURCE_REVISION,
        "weights": str(weights),
        "weights_sha256": model.pretrained_sha256,
        "variant": model.variant,
        "feature_blocks": list(model.feature_blocks),
        "encoder_parameters": sum(value.numel() for value in model.encoder.parameters()),
        "decoder_channels": args.decoder_channels,
        "total_parameters": sum(value.numel() for value in model.parameters()),
        "input_shape": list(pixels.shape),
        "logits_shape": list(output["logits"].shape),
        "boundary_logits_shape": list(output["boundary_logits"].shape),
        "torch": torch.__version__,
        "device": str(device),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
