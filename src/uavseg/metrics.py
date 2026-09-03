from __future__ import annotations

import numpy as np


def confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 0,
) -> np.ndarray:
    """Build a target-by-prediction confusion matrix."""
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction={prediction.shape}, target={target.shape}")

    valid = (
        (target != ignore_index)
        & (target >= 0)
        & (target < num_classes)
        & (prediction >= 0)
        & (prediction < num_classes)
    )
    encoded = target[valid] * num_classes + prediction[valid]
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def iou_from_confusion(
    matrix: np.ndarray,
    ignore_index: int = 0,
) -> tuple[float, dict[int, float | None]]:
    """Return mean IoU and per-class IoU, excluding ignore and absent classes."""
    matrix = np.asarray(matrix, dtype=np.float64)
    intersection = np.diag(matrix)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - intersection

    per_class: dict[int, float | None] = {}
    evaluated: list[float] = []
    for class_id in range(matrix.shape[0]):
        if class_id == ignore_index or union[class_id] == 0:
            per_class[class_id] = None
            continue
        value = float(intersection[class_id] / union[class_id])
        per_class[class_id] = value
        evaluated.append(value)

    return (float(np.mean(evaluated)) if evaluated else 0.0), per_class

