#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


EXPECTED_COUNTS = {
    "train/images": 6996,
    "train/masks": 6996,
    "test1/images": 500,
}


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == member.file_size:
                continue
            with handle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def count_pngs(path: Path) -> int:
    return sum(file.suffix.lower() == ".png" for file in path.rglob("*"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely extract the official UAV archives.")
    parser.add_argument("--archives", type=Path, required=True, help="Directory containing train.zip/test_1.zip")
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()

    train_archive = args.archives / "train.zip"
    test_archive = args.archives / "test_1.zip"
    for path in (train_archive, test_archive):
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"Extracting {train_archive} -> {args.output}")
    safe_extract(train_archive, args.output)
    print(f"Extracting {test_archive} -> {args.output / 'test1'}")
    safe_extract(test_archive, args.output / "test1")

    examples_archive = args.archives / "Examples.zip"
    if examples_archive.is_file():
        print(f"Extracting {examples_archive} -> {args.output}")
        safe_extract(examples_archive, args.output)
        EXPECTED_COUNTS["Examples/images"] = 11
        EXPECTED_COUNTS["Examples/masks"] = 11

    for relative, expected in EXPECTED_COUNTS.items():
        actual = count_pngs(args.output / relative)
        if actual != expected:
            raise RuntimeError(f"{relative}: expected {expected} PNG files, found {actual}")
        print(f"OK {relative}: {actual}")


if __name__ == "__main__":
    main()
