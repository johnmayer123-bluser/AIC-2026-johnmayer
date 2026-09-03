from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.check_dataset import analyze_dataset, write_reports
from uavseg.data import SegmentationDataset


class DatasetCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.images = self.root / "images"
        self.masks = self.root / "masks"
        self.images.mkdir()
        self.masks.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_rgb(path: Path, size: tuple[int, int] = (8, 8)) -> None:
        pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(path)

    @staticmethod
    def _write_mask(
        path: Path, values: np.ndarray | None = None, size: tuple[int, int] = (8, 8)
    ) -> None:
        if values is None:
            values = np.zeros((size[1], size[0]), dtype=np.uint8)
        Image.fromarray(values, mode="L").save(path)

    def test_healthy_dataset_and_reports(self) -> None:
        self._write_rgb(self.images / "sample.jpg")
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:, 4:] = 8
        self._write_mask(self.masks / "sample.png", mask)

        summary, records = analyze_dataset(
            self.images, self.masks, expected_size=(8, 8)
        )

        self.assertTrue(summary["healthy"])
        self.assertEqual(summary["valid_pairs"], 1)
        self.assertEqual(records[0].mask_ids, "0,8")
        output = self.root / "reports"
        summary_path, records_path = write_reports(summary, records, output)
        self.assertTrue(summary_path.is_file())
        self.assertTrue(records_path.is_file())

    def test_detects_missing_mask_invalid_id_and_size_mismatch(self) -> None:
        self._write_rgb(self.images / "missing.jpg")
        self._write_rgb(self.images / "bad.jpg")
        invalid_mask = np.full((4, 4), 9, dtype=np.uint8)
        self._write_mask(self.masks / "bad.png", invalid_mask)

        summary, records = analyze_dataset(
            self.images, self.masks, expected_size=(8, 8)
        )
        issues = summary["issue_counts"]

        self.assertFalse(summary["healthy"])
        self.assertEqual(issues["missing_mask"], 1)
        self.assertEqual(issues["unexpected_class_id"], 1)
        self.assertEqual(issues["unexpected_mask_size"], 1)
        self.assertEqual(issues["image_mask_size_mismatch"], 1)
        self.assertEqual(len(records), 2)

    def test_training_dataset_keeps_image_and_mask_aligned(self) -> None:
        self._write_rgb(self.images / "sample.jpg")
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:, 4:] = 8
        self._write_mask(self.masks / "sample.png", mask)
        dataset = SegmentationDataset(
            [(self.images / "sample.jpg", self.masks / "sample.png")],
            crop_size=8,
            training=True,
            scale_range=(1.0, 1.0),
            color_jitter=0.0,
            rare_crop_probability=0.0,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["pixel_values"].shape), (3, 8, 8))
        self.assertEqual(tuple(sample["labels"].shape), (8, 8))
        self.assertEqual(set(sample["labels"].unique().tolist()), {0, 8})


if __name__ == "__main__":
    unittest.main()

