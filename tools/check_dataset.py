#!/usr/bin/env python3
"""Validate an image-segmentation dataset and summarize its label distribution.

The checker is intentionally independent of a specific dataset directory layout.
Images and masks are paired by their relative path without the file extension.
For example, ``images/area/a.jpg`` matches ``masks/area/a.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
}
DEFAULT_CLASS_IDS = tuple(range(9))


@dataclass
class FileRecord:
    key: str
    image_path: str = ""
    mask_path: str = ""
    image_size: str = ""
    mask_size: str = ""
    image_mode: str = ""
    mask_mode: str = ""
    mask_ids: str = ""
    status: str = "ok"
    issues: str = ""


def _relative_key(path: Path, root: Path) -> str:
    """Return a platform-independent relative path without its final suffix."""
    return path.relative_to(root).with_suffix("").as_posix()


def discover_files(root: Path) -> tuple[dict[str, Path], list[str]]:
    """Discover supported files and report duplicate relative keys."""
    files: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        key = _relative_key(path, root)
        if key in files:
            duplicates.append(key)
        else:
            files[key] = path
    return files, duplicates


def _format_size(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"


def _load_image_metadata(path: Path) -> tuple[tuple[int, int], str]:
    with Image.open(path) as image:
        image.load()
        return image.size, image.mode


def _load_mask(path: Path) -> tuple[np.ndarray, tuple[int, int], str]:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image), image.size, image.mode


def _add_issue(issues: list[str], issue: str) -> None:
    if issue not in issues:
        issues.append(issue)


def analyze_dataset(
    image_root: Path,
    mask_root: Path,
    expected_size: tuple[int, int] | None = (1024, 1024),
    allowed_class_ids: Sequence[int] = DEFAULT_CLASS_IDS,
    ignore_class_ids: Sequence[int] = (0,),
) -> tuple[dict[str, object], list[FileRecord]]:
    """Inspect a dataset and return a JSON-friendly summary plus per-file rows."""
    image_root = image_root.resolve()
    mask_root = mask_root.resolve()
    if not image_root.is_dir():
        raise ValueError(f"Image directory does not exist: {image_root}")
    if not mask_root.is_dir():
        raise ValueError(f"Mask directory does not exist: {mask_root}")

    images, duplicate_image_keys = discover_files(image_root)
    masks, duplicate_mask_keys = discover_files(mask_root)
    all_keys = sorted(set(images) | set(masks))
    allowed_ids = set(allowed_class_ids)
    ignore_ids = set(ignore_class_ids)

    issue_counts: Counter[str] = Counter()
    pixel_counts: Counter[int] = Counter()
    records: list[FileRecord] = []
    valid_pairs = 0

    if duplicate_image_keys:
        issue_counts["duplicate_image_keys"] = len(duplicate_image_keys)
    if duplicate_mask_keys:
        issue_counts["duplicate_mask_keys"] = len(duplicate_mask_keys)

    for index, key in enumerate(all_keys, start=1):
        image_path = images.get(key)
        mask_path = masks.get(key)
        issues: list[str] = []
        record = FileRecord(
            key=key,
            image_path=str(image_path) if image_path else "",
            mask_path=str(mask_path) if mask_path else "",
        )

        if image_path is None:
            _add_issue(issues, "missing_image")
        if mask_path is None:
            _add_issue(issues, "missing_mask")

        image_size: tuple[int, int] | None = None
        mask_size: tuple[int, int] | None = None

        if image_path is not None:
            try:
                image_size, record.image_mode = _load_image_metadata(image_path)
                record.image_size = _format_size(image_size)
                if expected_size and image_size != expected_size:
                    _add_issue(issues, "unexpected_image_size")
            except (OSError, ValueError, UnidentifiedImageError):
                _add_issue(issues, "corrupted_image")

        if mask_path is not None:
            try:
                mask_array, mask_size, record.mask_mode = _load_mask(mask_path)
                record.mask_size = _format_size(mask_size)
                if expected_size and mask_size != expected_size:
                    _add_issue(issues, "unexpected_mask_size")
                if mask_array.ndim != 2:
                    _add_issue(issues, "mask_not_single_channel")
                else:
                    unique_ids, counts = np.unique(mask_array, return_counts=True)
                    ids = [int(value) for value in unique_ids]
                    record.mask_ids = ",".join(str(value) for value in ids)
                    unexpected_ids = set(ids) - allowed_ids
                    if unexpected_ids:
                        _add_issue(issues, "unexpected_class_id")
                    for class_id, count in zip(unique_ids, counts, strict=True):
                        pixel_counts[int(class_id)] += int(count)
                    if record.mask_mode == "P":
                        _add_issue(issues, "palette_mask")
            except (OSError, ValueError, UnidentifiedImageError):
                _add_issue(issues, "corrupted_mask")

        if image_size and mask_size and image_size != mask_size:
            _add_issue(issues, "image_mask_size_mismatch")

        if not issues and image_path is not None and mask_path is not None:
            valid_pairs += 1
        else:
            for issue in issues:
                issue_counts[issue] += 1
            record.status = "issue"
            record.issues = ";".join(issues)
        records.append(record)

        if index % 250 == 0:
            print(f"Checked {index}/{len(all_keys)} file keys...", file=sys.stderr)

    total_pixels = sum(pixel_counts.values())
    evaluated_pixels = sum(
        count for class_id, count in pixel_counts.items() if class_id not in ignore_ids
    )
    class_statistics = []
    for class_id in sorted(set(allowed_ids) | set(pixel_counts)):
        count = pixel_counts.get(class_id, 0)
        class_statistics.append(
            {
                "class_id": class_id,
                "pixel_count": count,
                "percentage_all_pixels": round(100.0 * count / total_pixels, 6)
                if total_pixels
                else 0.0,
                "percentage_evaluated_pixels": (
                    None
                    if class_id in ignore_ids
                    else round(100.0 * count / evaluated_pixels, 6)
                    if evaluated_pixels
                    else 0.0
                ),
                "ignored_in_metric": class_id in ignore_ids,
            }
        )

    summary: dict[str, object] = {
        "image_root": str(image_root),
        "mask_root": str(mask_root),
        "expected_size": _format_size(expected_size) if expected_size else None,
        "allowed_class_ids": sorted(allowed_ids),
        "ignore_class_ids": sorted(ignore_ids),
        "images_found": len(images),
        "masks_found": len(masks),
        "matched_keys": len(set(images) & set(masks)),
        "valid_pairs": valid_pairs,
        "records_with_issues": sum(record.status == "issue" for record in records),
        "issue_counts": dict(sorted(issue_counts.items())),
        "duplicate_image_keys": duplicate_image_keys,
        "duplicate_mask_keys": duplicate_mask_keys,
        "total_mask_pixels": total_pixels,
        "class_statistics": class_statistics,
        "healthy": not issue_counts and all(record.status == "ok" for record in records),
    }
    return summary, records


def write_reports(
    summary: dict[str, object], records: Iterable[FileRecord], output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dataset_summary.json"
    records_path = output_dir / "dataset_files.csv"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = [field.name for field in FileRecord.__dataclass_fields__.values()]
    with records_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return summary_path, records_path


def parse_size(value: str) -> tuple[int, int] | None:
    if value.lower() in {"none", "off"}:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Size must look like 1024x1024 or 'none'")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Size values must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Size values must be positive")
    return width, height


def parse_id_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Class IDs must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one class ID is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check segmentation image/mask integrity and class distribution."
    )
    parser.add_argument("--images", type=Path, required=True, help="Image directory")
    parser.add_argument("--masks", type=Path, required=True, help="Mask directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dataset_check"),
        help="Report directory (default: outputs/dataset_check)",
    )
    parser.add_argument(
        "--expected-size",
        type=parse_size,
        default=(1024, 1024),
        help="Expected WIDTHxHEIGHT, or 'none' (default: 1024x1024)",
    )
    parser.add_argument(
        "--class-ids",
        type=parse_id_list,
        default=DEFAULT_CLASS_IDS,
        help="Allowed comma-separated label IDs (default: 0,1,...,8)",
    )
    parser.add_argument(
        "--ignore-ids",
        type=parse_id_list,
        default=(0,),
        help="Metric-ignore label IDs (default: 0)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, records = analyze_dataset(
            args.images,
            args.masks,
            expected_size=args.expected_size,
            allowed_class_ids=args.class_ids,
            ignore_class_ids=args.ignore_ids,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary_path, records_path = write_reports(summary, records, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSummary report: {summary_path.resolve()}")
    print(f"Per-file report: {records_path.resolve()}")
    return 0 if summary["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
