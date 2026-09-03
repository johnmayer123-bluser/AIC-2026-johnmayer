from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.make_submission import validate_predictions


class SubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.predictions = self.root / "predictions"
        self.images.mkdir()
        self.predictions.mkdir()
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(
            self.images / "test.png"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_accepts_official_grayscale_mask(self):
        Image.fromarray(np.full((8, 8), 8, dtype=np.uint8), mode="L").save(
            self.predictions / "test.png"
        )
        self.assertEqual(len(validate_predictions(self.images, self.predictions)), 1)

    def test_rejects_palette_mask(self):
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="P").save(
            self.predictions / "test.png"
        )
        with self.assertRaisesRegex(ValueError, "mode L"):
            validate_predictions(self.images, self.predictions)


if __name__ == "__main__":
    unittest.main()
