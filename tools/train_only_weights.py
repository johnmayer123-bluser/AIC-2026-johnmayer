"""Compute the original weight formula on a fixed split's train masks only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from uavseg.data import paired_samples, split_samples_from_file
from profile_dataset import CLASS_NAMES, class_weights, mask_features


def build(images, masks, split_path, output):
    output, split_path = Path(output), Path(split_path)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    for key in ("train", "val"):
        if len(split[key]) != len(set(split[key])):
            raise ValueError(f"Duplicate split.{key} keys")
    train, val = split_samples_from_file(paired_samples(images, masks), split_path)
    if not train:
        raise ValueError("Training split is empty")
    total = np.zeros(9, dtype=np.int64)
    per_image = []
    digest = hashlib.sha256()
    for image, mask in tqdm(train, desc="train-only mask statistics"):
        counts = mask_features(mask, 9)
        total += counts
        # Fingerprint exact training label bytes, bound to their filename stems.
        digest.update(image.stem.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(mask.read_bytes()).digest())
        per_image.append(dict(name=image.stem, pixel_counts=counts.tolist()))
    payload = dict(
        weights=class_weights(total, 0).tolist(), scope="train_only",
        train_count=len(train), val_count=len(val), split_sha256=hashlib.sha256(split_bytes).hexdigest(),
        training_mask_manifest_sha256=digest.hexdigest(),
        formula="1/sqrt(valid pixel frequency); normalize mean of present classes to 1; clip [0.5,3]; ignore0=0",
        classes=CLASS_NAMES, pixel_counts=total.tolist(),
    )
    output.mkdir(parents=True)
    (output / "class_weights.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "train_mask_stats.json").write_text(json.dumps(dict(
        split_sha256=payload["split_sha256"], scope="train_only", images=per_image), indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "masks", "split", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    build(args.images, args.masks, args.split, args.output)


if __name__ == "__main__":
    main()
