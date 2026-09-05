import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_validation import BINS, barren_bin, parse_args, run, summarize
from uavseg.metrics import confusion_matrix


class AnalysisTests(unittest.TestCase):
    def test_ignore_and_background_are_distinct(self):
        matrix = confusion_matrix(np.array([5, 1, 7, 0, 5]), np.array([5, 5, 5, 5, 0]), 9)
        report = summarize(matrix)
        self.assertEqual(report["valid_pixels"], 4)
        self.assertEqual(report["classes"][5]["recall"], .25)
        self.assertEqual(report["classes"][5]["precision"], 1.)
        self.assertEqual(report["classes"][5]["iou"], .25)
        self.assertEqual(report["barren_to_background_rate"], .25)
        self.assertEqual(report["barren_to_agricultural_rate"], .25)
        self.assertEqual(report["confusion_counts"][5][0], 1)
        self.assertIsNone(report["classes"][0]["iou"])

    def test_empty_and_absent_are_not_zero_recall(self):
        report = summarize(np.zeros((9, 9), dtype=np.int64))
        self.assertIsNone(report["miou"])
        self.assertIsNone(report["classes"][5]["recall"])
        self.assertEqual(report["confusion_row_rates"][5], [None] * 9)
        matrix = confusion_matrix(np.array([5]), np.array([1]), 9)
        self.assertEqual(summarize(matrix)["classes"][5]["iou"], 0.)
        self.assertIsNone(summarize(matrix)["classes"][5]["recall"])

    def test_bin_edges(self):
        self.assertEqual([barren_bin(value) for value in [0, .001, .05, .2, .5]], BINS)
        self.assertEqual(barren_bin(1), BINS[-1])
        with self.assertRaises(ValueError):
            barren_bin(-1)

    def test_tiny_checkpoint_end_to_end_and_no_overwrite(self):
        from transformers import SegformerConfig, SegformerModel
        from uavseg.model import BoundaryAwareSegFormer
        config = SegformerConfig(hidden_sizes=[8, 16, 32, 64], depths=[1] * 4,
                                 num_attention_heads=[1, 1, 2, 4], sr_ratios=[8, 4, 2, 1],
                                 mlp_ratios=[2] * 4)
        model = BoundaryAwareSegFormer(SegformerModel(config), decoder_channels=16)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["images", "masks"]:
                (root / name).mkdir()
            for name in ["a", "b"]:
                Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8)).save(root / "images" / f"{name}.png")
                Image.fromarray(np.full((64, 64), 5, dtype=np.uint8)).save(root / "masks" / f"{name}.png")
            (root / "split.json").write_text(json.dumps(dict(train=[], val=["a", "b"])))
            torch.save(dict(model_config=model.export_config(), model=model.state_dict(),
                            epoch=0, best_miou=.1, args=dict(ignore_index=0)), root / "best.pt")
            base = ["--images", str(root / "images"), "--masks", str(root / "masks"),
                    "--split", str(root / "split.json"), "--checkpoint", str(root / "best.pt"),
                    "--device", "cpu", "--top-k", "1"]
            args = parse_args(base + ["--output", str(root / "out"), "--limit", "1"])
            result = run(args)
            self.assertEqual(result["barren_area_groups"][BINS[-1]]["image_count"], 1)
            metadata = json.loads((root / "out/run.json").read_text())
            self.assertTrue(metadata["partial"])
            self.assertIsNone(metadata["delta_from_checkpoint_best"])
            self.assertEqual(metadata["status"], "complete")
            with Image.open(root / "out/cases/a.png") as image:
                self.assertEqual(image.width, 960)
            with self.assertRaises(FileExistsError):
                run(args)
            full = parse_args(base + ["--output", str(root / "full")])
            report = run(full)
            metadata = json.loads((root / "full/run.json").read_text())
            self.assertFalse(metadata["partial"])
            self.assertAlmostEqual(metadata["delta_from_checkpoint_best"], report["miou"] - .1)
            matrix_sum = np.sum([group["confusion_counts"] for group in report["barren_area_groups"].values()], axis=0)
            np.testing.assert_array_equal(matrix_sum, report["confusion_counts"])


if __name__ == "__main__":
    unittest.main()
