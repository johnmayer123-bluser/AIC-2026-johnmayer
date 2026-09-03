#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uavseg.data import paired_samples


CLASS_NAMES = [
    "Ignore",
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Vegetation",
    "Agricultural",
    "Vehicle",
]


def image_features(path: Path) -> np.ndarray:
    with Image.open(path) as loaded:
        image = loaded.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = array.mean(axis=(0, 1))
    std = array.std(axis=(0, 1))
    brightness = array.mean(axis=2)
    saturation = array.max(axis=2) - array.min(axis=2)
    return np.concatenate([mean, std, [brightness.mean(), brightness.std(), saturation.mean()]])


def mask_features(path: Path, num_classes: int) -> np.ndarray:
    with Image.open(path) as loaded:
        array = np.asarray(loaded)
    if array.ndim != 2:
        raise ValueError(f"Mask is not single channel: {path}")
    if array.shape != (1024, 1024):
        raise ValueError(f"Unexpected mask size {array.shape}: {path}")
    invalid = (array < 0) | (array >= num_classes)
    if invalid.any():
        raise ValueError(f"Unexpected IDs {np.unique(array[invalid]).tolist()}: {path}")
    return np.bincount(array.reshape(-1), minlength=num_classes).astype(np.int64)


def choose_split(
    descriptors: np.ndarray,
    val_count: int,
    trials: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Choose the random split whose label/appearance means best match the full set."""
    center = descriptors.mean(axis=0)
    scale = descriptors.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (descriptors - center) / scale
    rng = np.random.default_rng(seed)
    best_indices = None
    best_score = float("inf")
    for _ in range(trials):
        indices = rng.permutation(len(descriptors))[:val_count]
        score = float(np.abs(normalized[indices].mean(axis=0)).max())
        if score < best_score:
            best_indices = np.sort(indices)
            best_score = score
    assert best_indices is not None
    return best_indices, best_score


def class_weights(pixel_counts: np.ndarray, ignore_index: int) -> np.ndarray:
    evaluated = pixel_counts.astype(np.float64)
    evaluated[ignore_index] = 0
    frequencies = evaluated / max(1.0, evaluated.sum())
    weights = np.zeros_like(frequencies)
    valid = frequencies > 0
    weights[valid] = 1.0 / np.sqrt(frequencies[valid])
    weights[valid] /= weights[valid].mean()
    weights[valid] = np.clip(weights[valid], 0.5, 3.0)
    weights[ignore_index] = 0.0
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile labels and create a covariate-aware split.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_profile"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=None, help="Debug only: inspect the first N pairs")
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")

    samples = paired_samples(args.images, args.masks)
    if args.limit is not None:
        samples = samples[: args.limit]
    if len(samples) < 10:
        raise ValueError("At least 10 pairs are required")
    image_values = []
    mask_values = []
    for index, (image_path, mask_path) in enumerate(samples, start=1):
        image_values.append(image_features(image_path))
        mask_values.append(mask_features(mask_path, len(CLASS_NAMES)))
        if index % 100 == 0 or index == len(samples):
            print(f"Profiled {index}/{len(samples)}")

    images = np.stack(image_values)
    masks = np.stack(mask_values)
    presence = (masks[:, 1:] > 0).astype(np.float64)
    proportions = masks[:, 1:] / masks[:, 1:].sum(axis=1, keepdims=True).clip(min=1)
    descriptors = np.concatenate([presence, np.sqrt(proportions), images], axis=1)
    val_count = max(1, min(len(samples) - 1, round(len(samples) * args.val_fraction)))
    val_indices, split_score = choose_split(descriptors, val_count, args.trials, args.seed)
    val_set = set(val_indices.tolist())
    keys = [image.stem for image, _ in samples]
    split = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "selection_trials": args.trials,
        "max_standardized_mean_deviation": split_score,
        "train": [key for index, key in enumerate(keys) if index not in val_set],
        "val": [key for index, key in enumerate(keys) if index in val_set],
    }

    pixels = masks.sum(axis=0)
    weights = class_weights(pixels, ignore_index=0)
    total_pixels = max(1, int(pixels.sum()))
    profile = {
        "samples": len(samples),
        "classes": [
            {
                "id": class_id,
                "name": name,
                "pixel_count": int(pixels[class_id]),
                "pixel_percent": float(100.0 * pixels[class_id] / total_pixels),
                "image_presence_percent": float(100.0 * (masks[:, class_id] > 0).mean()),
                "training_weight": float(weights[class_id]),
            }
            for class_id, name in enumerate(CLASS_NAMES)
        ],
        "image_rgb_mean": images[:, :3].mean(axis=0).tolist(),
        "image_rgb_std": images[:, 3:6].mean(axis=0).tolist(),
        "split_summary": {
            "train_count": len(split["train"]),
            "val_count": len(split["val"]),
            "score": split_score,
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "class_weights.json").write_text(
        json.dumps({"weights": weights.tolist()}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote profile, split and class weights to {args.output.resolve()}")


if __name__ == "__main__":
    random.seed(3407)
    main()
