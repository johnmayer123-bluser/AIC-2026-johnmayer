import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))
from uavseg.data import SegmentationDataset
from train_only_weights import build


class CropTests(unittest.TestCase):
    def pair(self, values):
        return Image.fromarray(np.repeat(values[..., None], 3, axis=2)), Image.fromarray(values)

    def test_legacy_matches_old_draw_order(self):
        values = np.ones((80, 80), dtype=np.uint8)
        values[60:, 60:] = 7
        image, mask = self.pair(values)
        data = SegmentationDataset([], 32, True)
        for seed in range(30):
            random.seed(seed)
            attempts = 8 if random.random() < .5 else 1
            for _ in range(attempts):
                left, top = random.randint(0, 48), random.randint(0, 48)
                box = (left, top, left + 32, top + 32)
                if attempts == 1 or np.isin(np.asarray(mask.crop(box)), (5,7,8)).mean() >= .005:
                    break
            expected_rng = random.getstate()
            random.seed(seed)
            _, _, info = data.crop_with_info(image, mask)
            self.assertEqual(info["box"], box)
            self.assertEqual(random.getstate(), expected_rng)

    def test_single_pixel_at_edges_survives_and_is_aligned(self):
        for y, x in [(0,0), (0,79), (79,0), (79,79), (30,40)]:
            values = np.ones((80,80), dtype=np.uint8)
            values[y,x] = 5
            data = SegmentationDataset([], 32, True, rare_crop_probability=1, crop_strategy="targeted")
            rgb, label, info = data.crop_with_info(*self.pair(values))
            self.assertEqual(info["target_class"], 5)
            self.assertEqual(info["attempts"], 8)
            self.assertFalse(info["threshold_met"])
            self.assertEqual((np.asarray(label) == 5).sum(), 1)
            np.testing.assert_array_equal(np.asarray(rgb)[..., 0], np.asarray(label))

    def test_absent_class_falls_back_and_ordinary_branch_is_kept(self):
        pair = self.pair(np.ones((80,80), dtype=np.uint8))
        for probability, branch in [(1, "no_target_fallback"), (0, "random")]:
            data = SegmentationDataset([], 32, True, rare_crop_probability=probability, crop_strategy="targeted")
            _, mask, info = data.crop_with_info(*pair)
            self.assertEqual(info["branch"], branch)
            self.assertEqual(mask.size, (32,32))

    def test_select_class_not_combined_area_and_best_fallback(self):
        values = np.full((80,80), 7, dtype=np.uint8)
        values[40,40] = 5
        data = SegmentationDataset([], 32, True, rare_crop_probability=1, crop_strategy="targeted")
        with patch("uavseg.data.random.choice", return_value=5):
            _, label, info = data.crop_with_info(*self.pair(values))
        self.assertEqual(info["attempts"], 8)  # Farm pixels cannot satisfy the barren threshold.
        self.assertFalse(info["threshold_met"])
        self.assertTrue((np.asarray(label) == 5).any())

    def test_uniform_class_choice_and_varied_anchor_positions(self):
        values = np.full((80,80), 7, dtype=np.uint8)
        values[40,40] = 5
        data = SegmentationDataset([], 32, True, rare_crop_probability=1, crop_strategy="targeted")
        selected, offsets = [], set()
        random.seed(3407)
        for _ in range(200):
            _, _, info = data.crop_with_info(*self.pair(values))
            selected.append(info["target_class"])
            if info["target_class"] == 5:
                offsets.add((40-info["box"][0],40-info["box"][1]))
        self.assertTrue(70 < selected.count(5) < 130)
        self.assertGreater(len(offsets), 10)

    def test_resume_rejects_strategy_change_and_accepts_old_legacy(self):
        from argparse import Namespace
        from train import validate_resume_args
        validate_resume_args(Namespace(crop_strategy="legacy"), {})
        with self.assertRaisesRegex(ValueError, "crop_strategy"):
            validate_resume_args(Namespace(crop_strategy="targeted"), {})

    def test_targeted_dataset_full_augmentation_alignment_and_validation(self):
        from uavseg.data import IMAGENET_MEAN, IMAGENET_STD
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = np.ones((80,80), dtype=np.uint8)
            values[20:60,20:60] = 5
            image, mask = self.pair(values)
            image.save(root/"image.png")
            mask.save(root/"mask.png")
            pairs = [(root/"image.png", root/"mask.png")]
            dataset = SegmentationDataset(pairs, 32, True, scale_range=(1,1),
                                          rare_crop_probability=1, crop_strategy="targeted")
            with patch.object(dataset, "_photometric", side_effect=lambda image: image):
                result = dataset[0]
            restored = ((result["pixel_values"] * IMAGENET_STD + IMAGENET_MEAN) * 255).round().numpy()
            np.testing.assert_array_equal(restored[0], result["labels"].numpy())
            for strategy in ("legacy", "targeted"):
                val = SegmentationDataset(pairs, 32, False, crop_strategy=strategy)[0]
                np.testing.assert_array_equal(val["labels"].numpy(), values)

    def test_targeted_fallback_keeps_best_candidate_not_last(self):
        values = np.ones((80,80), dtype=np.uint8)
        values[40,40:42] = 5
        data = SegmentationDataset([], 32, True, rare_crop_probability=1, crop_strategy="targeted")
        # Anchor remains (40,40); first box contains both pixels, later boxes only one.
        with patch("uavseg.data.random.randrange", return_value=0), patch(
            "uavseg.data.random.randint", side_effect=[10,10] + [9,10]*7
        ):
            _, label, info = data.crop_with_info(*self.pair(values))
        self.assertEqual(info["box"], (10,10,42,42))
        self.assertEqual(info["target_pixels"], 2)
        self.assertEqual((np.asarray(label) == 5).sum(), 2)

    def test_train_weights_never_read_validation_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub in ("images", "masks"):
                (root/sub).mkdir()
            Image.new("RGB", (1024,1024)).save(root/"images/a.png")
            Image.new("RGB", (1024,1024)).save(root/"images/b.png")
            Image.new("L", (1024,1024), 5).save(root/"masks/a.png")
            # Invalid mask content would fail if validation pixels were read.
            Image.new("L", (1024,1024), 255).save(root/"masks/b.png")
            (root/"split.json").write_text(json.dumps(dict(train=["a"], val=["b"])))
            result = build(root/"images", root/"masks", root/"split.json", root/"out")
            self.assertEqual(result["pixel_counts"][5], 1024**2)
            self.assertEqual(result["weights"][5], 1)
            self.assertEqual(result["weights"][0], 0)
            self.assertEqual(result["train_count"], 1)
            with self.assertRaises(FileExistsError):
                build(root/"images", root/"masks", root/"split.json", root/"out")


if __name__ == "__main__":
    unittest.main()
