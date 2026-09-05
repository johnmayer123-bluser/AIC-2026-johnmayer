"""Compare crop exposure on fixed TRAIN-only strata, without model training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uavseg.data import SegmentationDataset, paired_samples, split_samples_from_file

COLORS = np.array([[0,0,0],[150,150,150],[220,70,70],[240,200,50],[40,100,230],
                   [180,120,65],[30,160,70],[155,210,90],[200,60,220]], dtype=np.uint8)


def save_panel(image, mask, crops, path):
    canvas = Image.new("RGB", (960, 700), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (rgb, label, title) in enumerate([(image, mask, "Original"),
                                            (*crops["legacy"], "Legacy crop"),
                                            (*crops["targeted"], "Targeted strategy crop")]):
        draw.text((i * 320 + 5, 5), title, fill="black")
        canvas.paste(rgb.resize((320, 320)), (i * 320, 25))
        color = Image.fromarray(COLORS[np.asarray(label)])
        canvas.paste(color.resize((320, 320), Image.Resampling.NEAREST), (i * 320, 350))
    draw.text((5, 678), "GT colors: gray=background; brown=barren; light green=agricultural; black=ignore", fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("images", "masks", "split", "train-stats", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--per-group", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.per_group < 1 or args.repeats < 1:
        parser.error("per-group and repeats must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    split_sha = hashlib.sha256(Path(args.split).read_bytes()).hexdigest()
    stats = json.loads(Path(args.train_stats).read_text(encoding="utf-8"))
    if stats["scope"] != "train_only" or stats["split_sha256"] != split_sha:
        raise ValueError("Training stats must belong to this exact split")
    train, _ = split_samples_from_file(paired_samples(args.images, args.masks), args.split)
    by_name = {image.stem: (image, mask) for image, mask in train}
    if len(stats["images"]) != len(train) or {row["name"] for row in stats["images"]} != set(by_name):
        raise ValueError("Training stats names differ from train split")
    groups = {name: [] for name in ("small_barren", "other_barren", "no_barren")}
    for row in stats["images"]:
        fraction = row["pixel_counts"][5] / sum(row["pixel_counts"])
        group = "no_barren" if fraction == 0 else "small_barren" if fraction < .05 else "other_barren"
        groups[group].append(row["name"])
    rng = random.Random(args.seed)
    selected = [(group, name) for group, names in groups.items()
                for name in rng.sample(sorted(names), min(args.per_group, len(names)))]
    output.mkdir(parents=True)
    (output / "examples").mkdir()
    datasets = {mode: SegmentationDataset([], 640, True, crop_strategy=mode)
                for mode in ("legacy", "targeted")}
    rows = []
    for index, (group, name) in enumerate(tqdm(selected, desc="paired crop audit")):
        image_path, mask_path = by_name[name]
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        with Image.open(mask_path) as loaded:
            mask = loaded.convert("L")
        for repeat in range(args.repeats):
            seed = args.seed + index * args.repeats + repeat
            random.seed(seed)
            resized_image, resized_mask = datasets["legacy"]._resize_for_training(image, mask)
            crops = {}
            for mode, dataset in datasets.items():
                random.seed(seed + 1000000)  # Same branch coin; different strategies then draw differently.
                rgb, label, info = dataset.crop_with_info(resized_image, resized_mask)
                crops[mode] = (rgb, label)
                values = np.asarray(label)
                rows.append(dict(name=name, group=group, repeat=repeat, strategy=mode,
                                 branch=info["branch"], target_class=info["target_class"],
                                 attempts=info["attempts"], box=str(info["box"]),
                                 threshold_met=info.get("threshold_met"),
                                 barren_present=bool((values == 5).any()),
                                 barren_fraction=float((values == 5).mean()),
                                 background_fraction=float((values == 1).mean()),
                                 agricultural_fraction=float((values == 7).mean())))
            # Two preselected images per stratum, first repeat; not cherry-picked for benefit.
            if repeat == 0 and sum(1 for g, _ in selected[:index] if g == group) < 2:
                save_panel(image, mask, crops, output / "examples" / f"{group}_{name}.png")
    summaries = []
    for group in groups:
        for mode in datasets:
            subset = [row for row in rows if row["group"] == group and row["strategy"] == mode]
            if not subset:
                continue
            aimed = [row for row in subset if row["target_class"] == 5]
            summaries.append(dict(group=group, strategy=mode, crops=len(subset),
                                  barren_hit_rate=float(np.mean([row["barren_present"] for row in subset])),
                                  mean_barren_fraction=float(np.mean([row["barren_fraction"] for row in subset])),
                                  mean_background_fraction=float(np.mean([row["background_fraction"] for row in subset])),
                                  barren_selected=len(aimed),
                                  barren_selected_and_present=sum(row["barren_present"] for row in aimed),
                                  barren_selected_threshold_met=sum(bool(row["threshold_met"]) for row in aimed)))
    report = dict(status="complete", args=vars(args), split_sha256=split_sha,
                  data_code_sha256=hashlib.sha256((ROOT / "src/uavseg/data.py").read_bytes()).hexdigest(),
                  available_train_images={key: len(value) for key, value in groups.items()},
                  selected_images=len(selected), summaries=summaries,
                  notes=["Balanced strata are NOT the natural training distribution; compare within strata.",
                         "Exposure audit only, not mIoU evidence. Same resized inputs; no flips or photometric changes.",
                         "No image oversampling in training; repeats are for this offline audit only."])
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "crops.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
