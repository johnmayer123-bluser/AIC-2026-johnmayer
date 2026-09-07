from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uavseg.data import IMAGENET_MEAN, IMAGENET_STD, indexed_files
from uavseg.model import model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate single-channel masks with one model.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dinov3-source",
        default=None,
        help="Override the official DINOv3 source checkout stored in a B-line checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    amp = device.type == "cuda" and not args.no_amp
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_config(
        checkpoint["model_config"], dinov3_source=args.dinov3_source
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    for _, path in tqdm(indexed_files(args.images).items(), desc="predict"):
        if path.name in used_names:
            raise ValueError(f"Duplicate output filename: {path.name}")
        used_names.add(path.name)
        with Image.open(path) as loaded:
            image = loaded.convert("RGB")
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        pixels = ((torch.from_numpy(array) - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)
        pixels = pixels.to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            logits = model(pixels)["logits"]
            logits = F.interpolate(
                logits,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            prediction = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        Image.fromarray(prediction).save(output / path.name)


if __name__ == "__main__":
    main()
