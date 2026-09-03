#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def png_files(path: Path) -> dict[str, Path]:
    files = {
        item.name: item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() == ".png"
    }
    return files


def validate_predictions(images_dir: Path, predictions_dir: Path) -> list[Path]:
    images = png_files(images_dir)
    predictions = png_files(predictions_dir)
    missing = sorted(set(images) - set(predictions))
    extra = sorted(set(predictions) - set(images))
    if missing or extra:
        raise ValueError(f"Filename mismatch: missing={len(missing)}, extra={len(extra)}")

    valid_paths = []
    for name in sorted(images):
        with Image.open(images[name]) as source:
            expected_size = source.size
        with Image.open(predictions[name]) as prediction:
            prediction.load()
            if prediction.mode != "L":
                raise ValueError(f"{name}: expected grayscale mode L, found {prediction.mode}")
            if prediction.size != expected_size:
                raise ValueError(f"{name}: expected {expected_size}, found {prediction.size}")
            array = np.asarray(prediction)
        values = np.unique(array)
        if values.size and (int(values.min()) < 0 or int(values.max()) > 8):
            raise ValueError(f"{name}: unexpected class IDs {values.tolist()}")
        valid_paths.append(predictions[name])
    return valid_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and zip official-format predictions.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = validate_predictions(args.images, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    print(f"Validated and packed {len(paths)} predictions: {args.output.resolve()}")


if __name__ == "__main__":
    main()
