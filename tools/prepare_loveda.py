"""Create a research-only, zero-copy LoveDA view with a fixed official split.

The official Train/Val archives contain Urban and Rural ``images_png`` / ``masks_png``
directories. This tool validates IDs 0..7, creates uniquely named links, and writes a
split compatible with scripts/train.py. It never reads or stages LoveDA Test data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CLASS_NAMES = [
    "Ignore",
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest_to_Vegetation",
    "Agricultural",
]


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return normalized or "root"


def _files_by_stem(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Duplicate stem {path.stem!r} in {root}")
            result[path.stem] = path
    return result


def discover_partition(root: Path, split_name: str) -> list[tuple[str, Path, Path]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LoveDA {split_name} root not found: {root}")
    pairs: list[tuple[str, Path, Path]] = []
    image_directories = sorted(path for path in root.rglob("images_png") if path.is_dir())
    if not image_directories:
        raise ValueError(f"No images_png directories found below {root}")
    for image_directory in image_directories:
        mask_directory = image_directory.with_name("masks_png")
        if not mask_directory.is_dir():
            raise ValueError(f"Missing sibling masks_png directory: {mask_directory}")
        images = _files_by_stem(image_directory)
        masks = _files_by_stem(mask_directory)
        if set(images) != set(masks):
            raise ValueError(
                f"Unpaired LoveDA files in {image_directory.parent}: "
                f"images_only={sorted(set(images) - set(masks))[:5]}, "
                f"masks_only={sorted(set(masks) - set(images))[:5]}"
            )
        scene = "__".join(
            _safe_component(part)
            for part in image_directory.parent.relative_to(root).parts
        ) or "root"
        for stem in sorted(images):
            key = f"loveda_{split_name}__{scene}__{_safe_component(stem)}"
            pairs.append((key, images[stem], masks[stem]))
    return pairs


def validate_pair(image_path: Path, mask_path: Path) -> tuple[int, ...]:
    with Image.open(image_path) as image:
        image_size = image.size
    with Image.open(mask_path) as mask:
        mask_array = np.asarray(mask)
        mask_size = mask.size
        mask_mode = mask.mode
    if image_size != (1024, 1024) or mask_size != image_size:
        raise ValueError(
            f"LoveDA expects aligned 1024x1024 data: {image_path}={image_size}, "
            f"{mask_path}={mask_size}"
        )
    if mask_array.ndim != 2 or mask_mode == "P":
        raise ValueError(f"LoveDA mask must be a single-channel ID image, not {mask_mode}: {mask_path}")
    ids = tuple(int(value) for value in np.unique(mask_array))
    invalid = sorted(set(ids) - set(range(8)))
    if invalid:
        raise ValueError(f"Unexpected LoveDA label IDs {invalid}: {mask_path}")
    return ids


def _materialize(source: Path, destination: Path, copy_files: bool) -> None:
    if copy_files:
        shutil.copy2(source, destination)
    else:
        os.symlink(source, destination)


def prepare(train_root: Path, val_root: Path, output: Path, copy_files: bool = False) -> dict:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    train_pairs = discover_partition(train_root, "train")
    val_pairs = discover_partition(val_root, "val")
    keys = [key for key, _, _ in train_pairs + val_pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("Normalized LoveDA keys are not unique")

    label_ids: set[int] = set()
    manifest = hashlib.sha256()
    for key, image_path, mask_path in train_pairs + val_pairs:
        label_ids.update(validate_pair(image_path, mask_path))
        manifest.update(key.encode("utf-8") + b"\0")
        manifest.update(str(image_path.resolve()).encode("utf-8") + b"\0")
        manifest.update(str(mask_path.resolve()).encode("utf-8") + b"\0")

    image_output = output / "images"
    mask_output = output / "masks"
    image_output.mkdir(parents=True)
    mask_output.mkdir(parents=True)
    for index, (key, image_path, mask_path) in enumerate(train_pairs + val_pairs, start=1):
        image_destination = image_output / f"{key}{image_path.suffix.lower()}"
        mask_destination = mask_output / f"{key}{mask_path.suffix.lower()}"
        _materialize(image_path.resolve(), image_destination, copy_files)
        _materialize(mask_path.resolve(), mask_destination, copy_files)
        if index % 250 == 0:
            print(f"Prepared {index}/{len(keys)} LoveDA pairs")

    split = {
        "purpose": "research_only_loveda_official_train_val",
        "train": [key for key, _, _ in train_pairs],
        "val": [key for key, _, _ in val_pairs],
    }
    metadata = {
        "research_only": True,
        "competition_submission_allowed": False,
        "train_root": str(Path(train_root).expanduser().resolve()),
        "val_root": str(Path(val_root).expanduser().resolve()),
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "label_ids": sorted(label_ids),
        "classes": CLASS_NAMES,
        "class_mapping": {
            "0": "Ignore",
            "1": "Background",
            "2": "Building",
            "3": "Road",
            "4": "Water",
            "5": "Barren",
            "6": "Forest -> competition Vegetation",
            "7": "Agriculture -> competition Agricultural",
            "8": "Absent in LoveDA; target classifier is reinitialized",
        },
        "link_mode": "copy" if copy_files else "symbolic_link",
        "path_manifest_sha256": manifest.hexdigest(),
    }
    (output / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "DO_NOT_SUBMIT.txt").write_text(
        "LoveDA is external data. Models influenced by this directory are research-only "
        "and must not be submitted to the AIC competition.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating absolute symbolic links",
    )
    args = parser.parse_args()
    prepare(args.train_root, args.val_root, args.output, copy_files=args.copy)


if __name__ == "__main__":
    main()
