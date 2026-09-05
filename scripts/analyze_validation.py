"""Read-only A00 validation diagnostics; no training, downloads, or split generation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from uavseg.data import SegmentationDataset, paired_samples, split_samples_from_file
from uavseg.metrics import confusion_matrix, iou_from_confusion

NAMES = ["Ignore", "Background", "Building", "Road", "Water", "Barren",
         "Vegetation", "Agricultural", "Vehicle"]
# Diagnostic colors only; saved predictions always contain raw IDs 0..8.
PALETTE = np.array([[0, 0, 0], [150, 150, 150], [220, 70, 70], [240, 200, 50],
                    [40, 100, 230], [180, 120, 65], [30, 160, 70],
                    [155, 210, 90], [200, 60, 220]], dtype=np.uint8)
BINS = ["absent", "(0,5%)", "[5%,20%)", "[20%,50%)", "[50%,100%]"]


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def barren_bin(fraction: float) -> str:
    if not 0 <= fraction <= 1:
        raise ValueError("Area fraction must be in [0,1]")
    if fraction == 0:
        return BINS[0]
    return BINS[1 if fraction < .05 else 2 if fraction < .20 else 3 if fraction < .50 else 4]


def summarize(matrix: np.ndarray) -> dict:
    miou, ious = iou_from_confusion(matrix, ignore_index=0)
    rows, cols = matrix.sum(1), matrix.sum(0)
    classes = []
    for index, name in enumerate(NAMES):
        classes.append(dict(
            id=index, name=name, target_pixels=int(rows[index]),
            predicted_pixels_on_valid_target=int(cols[index]),
            true_positive_pixels=int(matrix[index, index]), iou=ious[index],
            precision=ratio(matrix[index, index], cols[index]) if index else None,
            recall=ratio(matrix[index, index], rows[index]) if index else None,
        ))
    return dict(miou=miou if matrix.sum() else None, classes=classes,
                valid_pixels=int(matrix.sum()),
                pixel_accuracy=ratio(np.trace(matrix), matrix.sum()),
                confusion_counts=matrix.tolist(),
                confusion_row_rates=[[ratio(value, total) for value in row]
                                     for row, total in zip(matrix, rows)],
                barren_to_background_rate=ratio(matrix[5, 1], rows[5]),
                barren_to_agricultural_rate=ratio(matrix[5, 7], rows[5]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_case(image_path, mask_path, prediction_path, output_path):
    with Image.open(image_path) as image:
        original = image.convert("RGB")
    with Image.open(mask_path) as image:
        target = np.array(image.convert("L"))
    with Image.open(prediction_path) as image:
        prediction = np.array(image)
    # Gray = ignored GT; white = correct; red = incorrect valid target.
    error = np.full((*target.shape, 3), 255, dtype=np.uint8)
    error[(prediction != target) & (target != 0)] = [230, 40, 40]
    error[target == 0] = [100, 100, 100]
    panels = [original, Image.fromarray(PALETTE[target]),
              Image.fromarray(PALETTE[prediction]), Image.fromarray(error)]
    width = 480
    height = max(1, round(original.height * width / original.width))
    canvas = Image.new("RGB", (width * 2, (height + 28) * 2 + 98), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(panels, ["Original", "Ground truth", "Prediction", "Error map"])):
        x, y = (index % 2) * width, (index // 2) * (height + 28)
        draw.text((x + 8, y + 6), title, fill="black")
        canvas.paste(panel.resize((width, height), Image.Resampling.BILINEAR if index == 0
                                 else Image.Resampling.NEAREST), (x, y + 28))
    y = (height + 28) * 2
    for index, name in enumerate(NAMES):
        x, yy = (index % 5) * 190, y + (index // 5) * 24
        draw.rectangle((x + 5, yy + 4, x + 19, yy + 18), fill=tuple(PALETTE[index]))
        draw.text((x + 24, yy + 5), f"{index}: {name}", fill="black")
    draw.text((8, y + 53), "Error: red=wrong, white=correct, gray=ignored GT. Display resized; metrics use full resolution.", fill="black")
    draw.text((8, y + 75), f"Image: {image_path.name}", fill="black")
    canvas.save(output_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "masks", "split", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="First N validation images; 0 = full split")
    parser.add_argument("--top-k", type=int, default=12, help="Worst overall plus worst barren cases")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.top_k < 0:
        parser.error("--limit and --top-k must be nonnegative")
    return args


def run(args):
    from uavseg.model import BoundaryAwareSegFormer
    import transformers

    started = time.perf_counter()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}. Use a new directory.")
    split_path, checkpoint_path = Path(args.split), Path(args.checkpoint)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    for name in ("train", "val"):
        if len(split[name]) != len(set(split[name])):
            raise ValueError(f"Duplicate keys in split.{name}")
    _, samples = split_samples_from_file(paired_samples(args.images, args.masks), split_path)
    expected_count = len(samples)
    if not samples:
        raise ValueError("Validation split is empty")
    if args.limit:
        samples = samples[:args.limit]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run on a GPU instance, or explicitly use --device cpu.")
    amp = device.type == "cuda" and not args.no_amp
    torch.manual_seed(3407)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["model_config"]["num_classes"] != 9 or checkpoint.get("args", {}).get("ignore_index", 0) != 0:
        raise ValueError("This A00 diagnostic requires 9 classes and ignore_index=0")
    model = BoundaryAwareSegFormer.from_model_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    reference_miou = checkpoint.get("best_miou")
    epoch = checkpoint.get("epoch")
    del checkpoint
    output.mkdir(parents=True)
    predictions_dir = output / "predictions"
    predictions_dir.mkdir()
    provenance = dict(
        status="running", command=sys.argv, arguments=vars(args),
        checkpoint_sha256=sha256(checkpoint_path), split_sha256=sha256(split_path),
        source_sha256={str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in
                       [Path(__file__).resolve(), PROJECT_ROOT / "src/uavseg/data.py",
                        PROJECT_ROOT / "src/uavseg/model.py", PROJECT_ROOT / "src/uavseg/metrics.py"]},
        python=platform.python_version(), torch=str(torch.__version__),
        transformers=transformers.__version__, numpy=np.__version__,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        cuda_runtime=torch.version.cuda, amp=amp, seed=3407, checkpoint_epoch=epoch,
        checkpoint_best_miou=reference_miou, expected_validation_images=expected_count,
        evaluated_images=len(samples), partial=len(samples) != expected_count,
        protocol="train.validate: full image, batch 1, ImageNet normalization, no TTA; target 0 ignored; row=true, column=prediction",
    )
    write_json(output / "run.json", provenance)
    dataset = SegmentationDataset(samples, crop_size=640, training=False)
    total = np.zeros((9, 9), dtype=np.int64)
    groups = {key: dict(count=0, matrix=np.zeros((9, 9), dtype=np.int64)) for key in BINS}
    rows = []
    with torch.inference_mode():
        for index in tqdm(range(len(dataset)), desc="analyze validation"):
            batch = dataset[index]
            target = batch["labels"].numpy()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(batch["pixel_values"].unsqueeze(0).to(device))["logits"]
            # Same placement outside autocast as scripts/train.py::validate.
            logits = F.interpolate(logits, size=target.shape, mode="bilinear", align_corners=False)
            prediction = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
            matrix = confusion_matrix(prediction, target, 9, 0)
            total += matrix
            stats = summarize(matrix)
            fraction = float((target == 5).mean())
            group = barren_bin(fraction)
            groups[group]["count"] += 1
            groups[group]["matrix"] += matrix
            row = dict(name=batch["name"], miou=stats["miou"], valid_pixels=stats["valid_pixels"],
                       barren_fraction=fraction, barren_group=group,
                       barren_recall=stats["classes"][5]["recall"],
                       barren_to_background_rate=stats["barren_to_background_rate"],
                       barren_to_agricultural_rate=stats["barren_to_agricultural_rate"])
            row.update({f"iou_{item['id']}": item["iou"] for item in stats["classes"]})
            rows.append(row)
            Image.fromarray(prediction).save(predictions_dir / f"{batch['name']}.png")
    report = summarize(total)
    report["barren_area_groups"] = {
        key: dict(image_count=value["count"], **summarize(value["matrix"]))
        for key, value in groups.items()
    }
    report["notes"] = [
        "mIoU is computed from pooled pixel counts, NOT the average of per-image mIoU.",
        "Ignore target 0; predictions of 0 on valid targets count as false negatives.",
        "Barren area denominator is all image pixels, including ignored target pixels.",
        "Confusion row rates are fractions of true-class pixels, not fractions of errors.",
        "Absent-barren group can reveal false positives; its barren recall is undefined.",
        "Area groups are descriptive, pixel-weighted, and may be sparse/confounded by scene.",
        "No test labels: these results cannot establish which test classes caused the score gap.",
        "Checkpoint best_miou is a historical reference; use best.pt for baseline reproduction.",
    ]
    write_json(output / "metrics.json", report)
    write_csv(output / "per_class.csv", report["classes"])
    write_csv(output / "per_image.csv", rows)
    for name, key in [("confusion_counts", "confusion_counts"), ("confusion_row_rates", "confusion_row_rates")]:
        write_csv(output / f"{name}.csv", [dict(target=NAMES[i], **dict(zip(NAMES, values)))
                                          for i, values in enumerate(report[key])])
    write_csv(output / "barren_area_groups.csv", [dict(
        group=key, image_count=value["image_count"],
        barren_pixels=value["classes"][5]["target_pixels"],
        barren_iou=value["classes"][5]["iou"], barren_recall=value["classes"][5]["recall"],
        barren_to_background_rate=value["barren_to_background_rate"],
        barren_to_agricultural_rate=value["barren_to_agricultural_rate"],
    ) for key, value in report["barren_area_groups"].items()])
    worst = sorted((row for row in rows if row["miou"] is not None), key=lambda row: row["miou"])[:args.top_k]
    worst_barren = sorted((row for row in rows if row["barren_recall"] is not None),
                          key=lambda row: row["barren_recall"])[:args.top_k]
    write_json(output / "selected_cases.json", dict(worst_miou=worst, worst_barren_recall=worst_barren))
    cases_dir = output / "cases"
    cases_dir.mkdir()
    by_name = {image.stem: (image, mask) for image, mask in samples}
    for name in sorted({row["name"] for row in worst + worst_barren}):
        render_case(*by_name[name], predictions_dir / f"{name}.png", cases_dir / f"{name}.png")
    provenance.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                      miou=report["miou"], delta_from_checkpoint_best=(
                          report["miou"] - reference_miou if not provenance["partial"]
                          and reference_miou is not None and report["miou"] is not None else None))
    write_json(output / "run.json", provenance)
    print(json.dumps(dict(status="complete", partial=provenance["partial"], images=len(rows),
                          miou=report["miou"], output=str(output.resolve())), ensure_ascii=False))
    return report


if __name__ == "__main__":
    run(parse_args())
