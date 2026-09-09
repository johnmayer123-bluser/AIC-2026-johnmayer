"""Validate UAVid++ archives and create AIC-compatible 1088x1088 tiles.

Only the official UAVid++ train and validation partitions are decoded.  Test labels
are deliberately not opened or staged.  Images are read directly from ZIP archives,
which avoids keeping a second extracted copy of the 4K RGB frames on the server.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

import numpy as np
from PIL import Image


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

# Published UAVid++ RGB taxonomy -> AIC competition IDs.
COLOR_TO_AIC = {
    (0, 0, 0): 1,        # Background Clutter -> Background
    (128, 0, 0): 2,      # Building (Wall) -> Building
    (128, 64, 128): 3,   # Road -> Road
    (0, 128, 0): 6,      # Tree -> Vegetation
    (128, 128, 0): 6,    # Low Vegetation -> Vegetation
    (64, 0, 128): 8,     # Dynamic Car -> Vehicle
    (192, 0, 192): 8,    # Static Car -> Vehicle
    (64, 64, 0): 0,      # Human has no safe AIC equivalent -> Ignore
    (0, 0, 255): 4,      # Water -> Water
    (128, 255, 255): 0,  # Sky has no safe AIC equivalent -> Ignore
    (70, 70, 70): 2,     # Roof -> Building
}
PARTITIONS = {"uavid_train": "train", "uavid_val": "val", "uavid_test": "test"}
OFFICIAL_SIZES = {(3840, 2160), (4096, 2160)}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return normalized or "unnamed"


def _archive_members(
    archive: zipfile.ZipFile, expected_directory: str
) -> tuple[dict[tuple[str, str, str], str], dict[str, int]]:
    members: dict[tuple[str, str, str], str] = {}
    counts = {"train": 0, "val": 0, "test": 0}
    for info in archive.infolist():
        if info.is_dir() or PurePosixPath(info.filename).suffix.lower() != ".png":
            continue
        parts = PurePosixPath(info.filename.replace("\\", "/")).parts
        partition_positions = [i for i, part in enumerate(parts) if part in PARTITIONS]
        if len(partition_positions) != 1:
            continue
        index = partition_positions[0]
        if len(parts) != index + 4 or parts[index + 2].lower() != expected_directory.lower():
            continue
        split = PARTITIONS[parts[index]]
        sequence = parts[index + 1]
        stem = PurePosixPath(parts[index + 3]).stem
        key = (split, sequence, stem)
        if key in members:
            raise ValueError(f"Duplicate {expected_directory} member for {key}: {info.filename}")
        members[key] = info.filename
        counts[split] += 1
    return members, counts


def _decode_image(archive: zipfile.ZipFile, member: str, mode: str) -> Image.Image:
    with archive.open(member) as source:
        payload = source.read()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        return image.convert(mode)


def _map_mask(mask: Image.Image, member: str) -> tuple[Image.Image, np.ndarray]:
    array = np.asarray(mask.convert("RGB"), dtype=np.uint8)
    codes = (
        (array[..., 0].astype(np.uint32) << 16)
        | (array[..., 1].astype(np.uint32) << 8)
        | array[..., 2].astype(np.uint32)
    )
    mapping = {
        (red << 16) | (green << 8) | blue: target
        for (red, green, blue), target in COLOR_TO_AIC.items()
    }
    unique_codes = np.unique(codes)
    invalid = [
        ((int(code) >> 16) & 255, (int(code) >> 8) & 255, int(code) & 255)
        for code in unique_codes
        if int(code) not in mapping
    ]
    if invalid:
        raise ValueError(f"Unexpected UAVid++ colors {invalid[:10]} in {member}")
    target = np.zeros(codes.shape, dtype=np.uint8)
    source_color_counts = np.zeros(len(COLOR_TO_AIC), dtype=np.int64)
    for color_index, (color, target_id) in enumerate(COLOR_TO_AIC.items()):
        code = (color[0] << 16) | (color[1] << 8) | color[2]
        selected = codes == code
        source_color_counts[color_index] = int(selected.sum())
        target[selected] = target_id
    return Image.fromarray(target), source_color_counts


def _tile_key(split: str, sequence: str, stem: str, row: int, column: int) -> str:
    return (
        f"uavidpp_{split}__{_safe_component(sequence)}__{_safe_component(stem)}"
        f"__r{row}c{column}"
    )


def _process_pair(
    rgb_zip: zipfile.ZipFile,
    labels_zip: zipfile.ZipFile,
    key: tuple[str, str, str],
    rgb_member: str,
    label_member: str,
    image_output: Path,
    mask_output: Path,
    tile_size: int,
    allowed_sizes: set[tuple[int, int]] | None,
) -> dict:
    split, sequence, stem = key
    image = _decode_image(rgb_zip, rgb_member, "RGB")
    label_rgb = _decode_image(labels_zip, label_member, "RGB")
    if image.size != label_rgb.size:
        raise ValueError(f"Unaligned pair {key}: image={image.size}, label={label_rgb.size}")
    if allowed_sizes is not None and image.size not in allowed_sizes:
        raise ValueError(f"Unexpected UAVid++ frame size {image.size} for {rgb_member}")
    label, source_color_counts = _map_mask(label_rgb, label_member)

    columns = math.ceil(image.width / tile_size)
    rows = math.ceil(image.height / tile_size)
    tile_keys: list[str] = []
    class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for row in range(rows):
        for column in range(columns):
            name = _tile_key(split, sequence, stem, row, column)
            box = (
                column * tile_size,
                row * tile_size,
                (column + 1) * tile_size,
                (row + 1) * tile_size,
            )
            image_tile = image.crop(box)
            mask_tile = label.crop(box)
            image_tile.save(image_output / f"{name}.png", compress_level=1)
            mask_tile.save(mask_output / f"{name}.png", compress_level=6)
            class_counts += np.bincount(
                np.asarray(mask_tile, dtype=np.uint8).reshape(-1), minlength=len(CLASS_NAMES)
            )
            tile_keys.append(name)
    return {
        "split": split,
        "sequence": sequence,
        "stem": stem,
        "size": list(image.size),
        "tiles": tile_keys,
        "class_counts": class_counts.tolist(),
        "source_color_counts": source_color_counts.tolist(),
    }


def _check_expected_counts(
    counts: dict[str, int], expected: dict[str, int] | None, archive_name: str
) -> None:
    if expected is None:
        return
    mismatches = {
        split: (counts[split], wanted)
        for split, wanted in expected.items()
        if wanted >= 0 and counts[split] != wanted
    }
    if mismatches:
        raise ValueError(f"Unexpected {archive_name} partition counts: {mismatches}")


def prepare(
    rgb_archive: Path,
    labels_archive: Path,
    output: Path,
    *,
    tile_size: int = 1088,
    workers: int = 4,
    expected_counts: dict[str, int] | None = None,
    expected_rgb_sha256: str | None = None,
    expected_labels_sha256: str | None = None,
    approval_reference: str = "not-recorded",
    allowed_sizes: set[tuple[int, int]] | None = OFFICIAL_SIZES,
) -> dict:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    rgb_archive = rgb_archive.expanduser().resolve()
    labels_archive = labels_archive.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not rgb_archive.is_file() or not labels_archive.is_file():
        raise FileNotFoundError("Both UAVid++ RGB and label ZIP archives are required")

    print(
        f"Hashing RGB archive: {rgb_archive} ({rgb_archive.stat().st_size} bytes)",
        flush=True,
    )
    rgb_sha256 = sha256_file(rgb_archive)
    print(f"RGB archive SHA256: {rgb_sha256}", flush=True)
    print(
        f"Hashing label archive: {labels_archive} ({labels_archive.stat().st_size} bytes)",
        flush=True,
    )
    labels_sha256 = sha256_file(labels_archive)
    print(f"Label archive SHA256: {labels_sha256}", flush=True)
    if expected_rgb_sha256 and rgb_sha256.lower() != expected_rgb_sha256.lower():
        raise ValueError(f"RGB archive SHA256 mismatch: {rgb_sha256}")
    if expected_labels_sha256 and labels_sha256.lower() != expected_labels_sha256.lower():
        raise ValueError(f"Labels archive SHA256 mismatch: {labels_sha256}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-partial-", dir=output.parent))
    try:
        image_output = staging / "images"
        mask_output = staging / "masks"
        image_output.mkdir()
        mask_output.mkdir()
        with zipfile.ZipFile(rgb_archive) as rgb_zip, zipfile.ZipFile(labels_archive) as labels_zip:
            rgb_members, rgb_counts = _archive_members(rgb_zip, "Images")
            label_members, label_counts = _archive_members(labels_zip, "Labels")
            _check_expected_counts(rgb_counts, expected_counts, "RGB")
            _check_expected_counts(label_counts, expected_counts, "label")

            selected_rgb = {key: value for key, value in rgb_members.items() if key[0] != "test"}
            selected_labels = {
                key: value for key, value in label_members.items() if key[0] != "test"
            }
            if set(selected_rgb) != set(selected_labels):
                raise ValueError(
                    "Unpaired UAVid++ train/val members: "
                    f"images_only={sorted(set(selected_rgb) - set(selected_labels))[:5]}, "
                    f"labels_only={sorted(set(selected_labels) - set(selected_rgb))[:5]}"
                )
            selected_keys = sorted(selected_rgb)

            def process(key: tuple[str, str, str]) -> dict:
                return _process_pair(
                    rgb_zip,
                    labels_zip,
                    key,
                    selected_rgb[key],
                    selected_labels[key],
                    image_output,
                    mask_output,
                    tile_size,
                    allowed_sizes,
                )

            results: list[dict] = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for index, result in enumerate(executor.map(process, selected_keys), start=1):
                    results.append(result)
                    if index % 20 == 0 or index == len(selected_keys):
                        print(f"Prepared {index}/{len(selected_keys)} UAVid++ frames")

        split = {
            "purpose": "c01_uavidplusplus_official_sequence_split",
            "train": [tile for row in results if row["split"] == "train" for tile in row["tiles"]],
            "val": [tile for row in results if row["split"] == "val" for tile in row["tiles"]],
        }
        if set(split["train"]) & set(split["val"]):
            raise ValueError("UAVid++ train/validation tile leakage detected")

        class_counts = {
            split_name: np.sum(
                [row["class_counts"] for row in results if row["split"] == split_name], axis=0
            ).astype(np.int64).tolist()
            for split_name in ("train", "val")
        }
        source_color_counts = np.sum(
            [row["source_color_counts"] for row in results], axis=0
        ).astype(np.int64)
        sequences = {
            split_name: sorted({row["sequence"] for row in results if row["split"] == split_name})
            for split_name in ("train", "val")
        }
        metadata = {
            "purpose": "C01 external-domain adaptation before official AIC fine-tuning",
            "competition_submission_allowed": True,
            "approval_basis": "User reported explicit organizer approval for external labeled data",
            "approval_reference": approval_reference,
            "rgb_archive": str(rgb_archive),
            "labels_archive": str(labels_archive),
            "rgb_archive_sha256": rgb_sha256,
            "labels_archive_sha256": labels_sha256,
            "integrity_validation": (
                "Exact whole-archive SHA256 plus CRC validation while every selected "
                "Train/Val ZIP member is decoded"
            ),
            "archive_member_counts": {"rgb": rgb_counts, "labels": label_counts},
            "decoded_partitions": ["train", "val"],
            "test_labels_opened": False,
            "tile_size": [tile_size, tile_size],
            "tile_count": {"train": len(split["train"]), "val": len(split["val"])},
            "frame_count": {
                split_name: sum(row["split"] == split_name for row in results)
                for split_name in ("train", "val")
            },
            "sequences": sequences,
            "classes": CLASS_NAMES,
            "class_pixel_counts": class_counts,
            "source_color_pixel_counts_train_and_val": {
                str(color): int(count)
                for color, count in zip(COLOR_TO_AIC, source_color_counts)
            },
            "class_mapping": {
                "Background Clutter": "1 Background",
                "Building (Wall)": "2 Building",
                "Road": "3 Road",
                "Tree": "6 Vegetation",
                "Low Vegetation": "6 Vegetation",
                "Dynamic Car": "8 Vehicle",
                "Static Car": "8 Vehicle",
                "Human": "0 Ignore (no safe AIC equivalent)",
                "Water": "4 Water",
                "Sky": "0 Ignore (no safe AIC equivalent)",
                "Roof": "2 Building",
            },
            "absent_aic_classes": ["5 Barren", "7 Agricultural"],
            "padding": "right/bottom only; RGB=black and mask=0 Ignore",
        }
        (staging / "split.json").write_text(
            json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(output)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return metadata
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-archive", type=Path, required=True)
    parser.add_argument("--labels-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1088)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-train", type=int, default=200)
    parser.add_argument("--expected-val", type=int, default=70)
    parser.add_argument("--expected-test", type=int, default=150)
    parser.add_argument("--expected-rgb-sha256")
    parser.add_argument("--expected-labels-sha256")
    parser.add_argument(
        "--approval-reference",
        required=True,
        help="Stable reference to the saved organizer permission evidence",
    )
    args = parser.parse_args()
    prepare(
        args.rgb_archive,
        args.labels_archive,
        args.output,
        tile_size=args.tile_size,
        workers=args.workers,
        expected_counts={
            "train": args.expected_train,
            "val": args.expected_val,
            "test": args.expected_test,
        },
        expected_rgb_sha256=args.expected_rgb_sha256,
        expected_labels_sha256=args.expected_labels_sha256,
        approval_reference=args.approval_reference,
    )


if __name__ == "__main__":
    main()
