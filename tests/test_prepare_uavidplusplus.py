import json
import sys
import tempfile
import unittest
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.prepare_uavidplusplus import prepare
from src.uavseg.data import paired_samples, split_samples_from_file


def _png_bytes(array: np.ndarray, mode: str) -> bytes:
    import io

    buffer = io.BytesIO()
    image = Image.fromarray(array)
    if image.mode != mode:
        image = image.convert(mode)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class PrepareUavidPlusPlusTests(unittest.TestCase):
    def _archives(self, root: Path, invalid_color: bool = False) -> tuple[Path, Path]:
        rgb_path = root / "rgb.zip"
        labels_path = root / "labels.zip"
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        label = np.zeros_like(image)
        label[:, :2] = (128, 0, 0)
        label[:, 2:4] = (70, 70, 70)
        label[:, 4:6] = (64, 0, 128)
        label[:, 6:] = (128, 255, 255)
        if invalid_color:
            label[0, 0] = (1, 2, 3)
        with zipfile.ZipFile(rgb_path, "w") as archive:
            for split, sequence in (("uavid_train", "seq1"), ("uavid_val", "seq2")):
                archive.writestr(
                    f"UAVid_rgb_only/{split}/{sequence}/Images/000.png",
                    _png_bytes(image, "RGB"),
                )
            archive.writestr(
                "UAVid_rgb_only/uavid_test/seq3/Images/000.png", b"not-opened-by-preparer"
            )
        with zipfile.ZipFile(labels_path, "w") as archive:
            for split, sequence in (("uavid_train", "seq1"), ("uavid_val", "seq2")):
                archive.writestr(
                    f"UAVid++_labels/{split}/{sequence}/Labels/000.png",
                    _png_bytes(label, "RGB"),
                )
            archive.writestr(
                "UAVid++_labels/uavid_test/seq3/Labels/000.png", b"not-opened-by-preparer"
            )
        return rgb_path, labels_path

    def test_prepares_fixed_train_val_tiles_and_ignores_test_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb, labels = self._archives(root)
            output = root / "prepared"
            metadata = prepare(
                rgb,
                labels,
                output,
                tile_size=4,
                workers=2,
                expected_counts={"train": 1, "val": 1, "test": 1},
                approval_reference="unit-test",
                allowed_sizes={(8, 4)},
            )
            self.assertEqual(metadata["tile_count"], {"train": 2, "val": 2})
            self.assertFalse(metadata["test_labels_opened"])
            self.assertEqual(metadata["decoded_partitions"], ["train", "val"])
            split = json.loads((output / "split.json").read_text(encoding="utf-8"))
            train, val = split_samples_from_file(
                paired_samples(output / "images", output / "masks"), output / "split.json"
            )
            self.assertEqual((len(train), len(val)), (2, 2))
            self.assertFalse({path.stem for path, _ in train} & {path.stem for path, _ in val})
            ids = set()
            for path in (output / "masks").glob("*.png"):
                with Image.open(path) as mask:
                    ids.update(int(value) for value in np.unique(mask))
            self.assertEqual(ids, {0, 2, 8})

    def test_rejects_unknown_label_color_without_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb, labels = self._archives(root, invalid_color=True)
            output = root / "prepared"
            with self.assertRaisesRegex(ValueError, "Unexpected UAVid\\+\\+ colors"):
                prepare(
                    rgb,
                    labels,
                    output,
                    tile_size=4,
                    expected_counts={"train": 1, "val": 1, "test": 1},
                    approval_reference="unit-test",
                    allowed_sizes={(8, 4)},
                )
            self.assertFalse(output.exists())

    def test_rejects_archive_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb, labels = self._archives(root)
            with self.assertRaisesRegex(ValueError, "RGB archive SHA256 mismatch"):
                prepare(
                    rgb,
                    labels,
                    root / "prepared",
                    expected_rgb_sha256="0" * 64,
                    approval_reference="unit-test",
                )


if __name__ == "__main__":
    unittest.main()
