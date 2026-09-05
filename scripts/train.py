from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uavseg.data import (
    SegmentationDataset,
    paired_samples,
    split_samples,
    split_samples_from_file,
)
from uavseg.losses import SegmentationLoss
from uavseg.metrics import confusion_matrix, iou_from_confusion
from uavseg.model import BoundaryAwareSegFormer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one boundary-aware SegFormer model.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--split", default=None, help="split.json from tools/profile_dataset.py")
    parser.add_argument("--class-weights", default=None, help="class_weights.json from profiling")
    parser.add_argument("--output", default="outputs/a00_b2_boundary")
    parser.add_argument("--model", default="nvidia/mit-b2")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--ignore-index", type=int, default=0)
    parser.add_argument("--decoder-channels", type=int, default=192)
    parser.add_argument("--dice-weight", type=float, default=0.3)
    parser.add_argument("--boundary-weight", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--scale-min", type=float, default=0.75)
    parser.add_argument("--scale-max", type=float, default=1.5)
    parser.add_argument("--color-jitter", type=float, default=0.2)
    parser.add_argument("--rare-crop-probability", type=float, default=0.5)
    parser.add_argument("--crop-strategy", choices=("legacy", "targeted"), default="legacy")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load pretrained files from the Hugging Face cache without network checks",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_class_weights(path: str | None, num_classes: int, ignore_index: int) -> torch.Tensor:
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        values = payload["weights"]
    else:
        values = [0.0, 0.7, 0.9, 1.0, 1.2, 1.8, 0.7, 1.3, 2.4]
    if len(values) != num_classes:
        raise ValueError(f"Expected {num_classes} class weights, received {len(values)}")
    values[ignore_index] = 0.0
    return torch.tensor(values, dtype=torch.float32)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    power: float,
) -> LambdaLR:
    warmup_steps = max(1, round(total_steps * warmup_ratio))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return max(1e-6, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress) ** power

    return LambdaLR(optimizer, scale)


def validate_resume_args(current: argparse.Namespace, previous: dict[str, object]) -> None:
    keys = (
        "split",
        "class_weights",
        "epochs",
        "batch_size",
        "gradient_accumulation",
        "crop_size",
        "encoder_lr",
        "decoder_lr",
        "weight_decay",
        "warmup_ratio",
        "power",
        "dice_weight",
        "boundary_weight",
        "label_smoothing",
        "seed",
    )
    mismatches = [
        key
        for key in keys
        if key in previous and getattr(current, key) != previous[key]
    ]
    if getattr(current, "crop_strategy", "legacy") != previous.get("crop_strategy", "legacy"):
        mismatches.append("crop_strategy")
    for key in ("rare_crop_probability", "scale_min", "scale_max", "color_jitter"):
        if key in previous and getattr(current, key) != previous[key]:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "Resume arguments differ from the checkpoint: " + ", ".join(mismatches)
        )


@torch.no_grad()
def validate(
    model: BoundaryAwareSegFormer,
    loader: DataLoader,
    criterion: SegmentationLoss,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    amp: bool,
) -> dict[str, object]:
    model.eval()
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    for batch in tqdm(loader, desc="validate", leave=False):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(pixels)
            loss, _ = criterion(outputs["logits"], outputs["boundary_logits"], labels)
        total_loss += float(loss.detach())
        logits = F.interpolate(
            outputs["logits"], size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        predictions = logits.argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        for prediction, target in zip(predictions, targets):
            matrix += confusion_matrix(prediction, target, num_classes, ignore_index)
    miou, per_class = iou_from_confusion(matrix, ignore_index)
    return {
        "loss": total_loss / max(1, len(loader)),
        "miou": miou,
        "per_class_iou": per_class,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    set_seed(args.seed)
    amp = device.type == "cuda" and not args.no_amp

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    samples = paired_samples(args.images, args.masks)
    if args.split:
        train_samples, val_samples = split_samples_from_file(samples, args.split)
    else:
        print("WARNING: using a random fallback split; formal runs should pass --split")
        train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)
    print(f"samples: train={len(train_samples)}, val={len(val_samples)}")

    train_data = SegmentationDataset(
        train_samples,
        crop_size=args.crop_size,
        training=True,
        num_classes=args.num_classes,
        scale_range=(args.scale_min, args.scale_max),
        color_jitter=args.color_jitter,
        rare_crop_probability=args.rare_crop_probability,
        crop_strategy=args.crop_strategy,
    )
    val_data = SegmentationDataset(
        val_samples,
        crop_size=1024,
        training=False,
        num_classes=args.num_classes,
        rare_crop_probability=0.0,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, **loader_options
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, **loader_options)

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_args(args, checkpoint.get("args", {}))
        model = BoundaryAwareSegFormer.from_model_config(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
    else:
        model = BoundaryAwareSegFormer.from_pretrained(
            args.model,
            num_classes=args.num_classes,
            decoder_channels=args.decoder_channels,
            local_files_only=args.local_files_only,
        )
    checkpointing_enabled = False
    if not args.no_gradient_checkpointing:
        checkpointing_enabled = model.gradient_checkpointing_enable()
        if not checkpointing_enabled:
            print(
                "WARNING: this Transformers SegFormer version does not support gradient "
                "checkpointing; continuing without it"
            )
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    run_info = {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "gradient_checkpointing": checkpointing_enabled,
    }
    (output / "runtime.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_info, ensure_ascii=False))

    encoder_parameters = list(model.encoder.parameters())
    decoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    optimizer = AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": decoder_parameters, "lr": args.decoder_lr},
        ],
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = args.epochs * updates_per_epoch
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_ratio, args.power)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = SegmentationLoss(
        load_class_weights(args.class_weights, args.num_classes, args.ignore_index),
        ignore_index=args.ignore_index,
        dice_weight=args.dice_weight,
        boundary_weight=args.boundary_weight,
        label_smoothing=args.label_smoothing,
    ).to(device)
    start_epoch, best_miou, global_step = 1, -1.0, 0

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint["best_miou"])
        global_step = int(checkpoint.get("global_step", 0))

    history_path = output / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch_index, batch in enumerate(progress, start=1):
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                outputs = model(pixels)
                loss, parts = criterion(outputs["logits"], outputs["boundary_logits"], labels)
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            should_step = (
                batch_index % args.gradient_accumulation == 0 or batch_index == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            running_loss += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", ce=f"{parts['ce']:.4f}")

        metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            args.num_classes,
            args.ignore_index,
            amp,
        )
        metrics.update(
            epoch=epoch,
            train_loss=running_loss / max(1, len(train_loader)),
            encoder_lr=optimizer.param_groups[0]["lr"],
            decoder_lr=optimizer.param_groups[1]["lr"],
        )
        print(json.dumps(metrics, ensure_ascii=False))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_miou": max(best_miou, float(metrics["miou"])),
            "model": model.state_dict(),
            "model_config": model.export_config(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }
        torch.save(state, output / "last.pt")
        if float(metrics["miou"]) > best_miou:
            best_miou = float(metrics["miou"])
            state["best_miou"] = best_miou
            torch.save(state, output / "best.pt")
            print(f"new best mIoU: {best_miou:.6f}")


if __name__ == "__main__":
    main()
