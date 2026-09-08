import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train import initialize_compatible_weights
from tools.prepare_loveda import prepare
from tools.train_only_weights import build as build_train_only_weights
from uavseg.data import paired_samples, split_samples_from_file


class TinyTransferModel(nn.Module):
    def __init__(self, num_classes: int, variant: str = "tiny"):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.classifier = nn.Sequential(nn.Identity(), nn.Linear(2, num_classes))
        self.num_classes = num_classes
        self.variant = variant

    def export_config(self):
        return {
            "architecture": "dinov3_boundary",
            "variant": self.variant,
            "feature_blocks": [0, 1, 2, 3],
            "decoder_channels": 16,
            "num_classes": self.num_classes,
        }


class ResearchTransferTests(unittest.TestCase):
    def test_transfer_reinitializes_only_different_class_head(self):
        source = TinyTransferModel(8)
        target = TinyTransferModel(9)
        with torch.no_grad():
            source.encoder.weight.fill_(3.0)
            source.classifier[1].weight.fill_(7.0)
        target_head = target.classifier[1].weight.detach().clone()
        report = initialize_compatible_weights(
            target,
            {"model": source.state_dict(), "model_config": source.export_config()},
        )
        self.assertTrue(torch.equal(target.encoder.weight, source.encoder.weight))
        self.assertTrue(torch.equal(target.classifier[1].weight, target_head))
        self.assertEqual(
            report["reinitialized_parameter_tensors"],
            ["classifier.1.bias", "classifier.1.weight"],
        )
        self.assertEqual(report["source_num_classes"], 8)
        self.assertEqual(report["target_num_classes"], 9)

    def test_transfer_rejects_different_backbone(self):
        source = TinyTransferModel(8, variant="source")
        target = TinyTransferModel(9, variant="target")
        with self.assertRaisesRegex(ValueError, "variant differs"):
            initialize_compatible_weights(
                target,
                {"model": source.state_dict(), "model_config": source.export_config()},
            )

    def test_prepare_loveda_builds_research_only_fixed_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "raw" / "Train"
            val = root / "raw" / "Val"
            for partition, scene, value in ((train, "Urban", 2), (val, "Rural", 7)):
                images = partition / scene / "images_png"
                masks = partition / scene / "masks_png"
                images.mkdir(parents=True)
                masks.mkdir(parents=True)
                Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8)).save(
                    images / "sample.png"
                )
                Image.fromarray(np.full((1024, 1024), value, dtype=np.uint8)).save(
                    masks / "sample.png"
                )
            output = root / "prepared"
            metadata = prepare(train, val, output, copy_files=True)
            self.assertTrue(metadata["research_only"])
            self.assertFalse(metadata["competition_submission_allowed"])
            self.assertEqual(metadata["train_count"], 1)
            self.assertEqual(metadata["val_count"], 1)
            self.assertTrue((output / "DO_NOT_SUBMIT.txt").is_file())
            split = json.loads((output / "split.json").read_text(encoding="utf-8"))
            train_samples, val_samples = split_samples_from_file(
                paired_samples(output / "images", output / "masks"),
                output / "split.json",
            )
            self.assertEqual(len(train_samples), 1)
            self.assertEqual(len(val_samples), 1)
            self.assertNotEqual(split["train"][0], split["val"][0])
            weights = build_train_only_weights(
                output / "images",
                output / "masks",
                output / "split.json",
                root / "weights",
                num_classes=8,
            )
            self.assertEqual(len(weights["weights"]), 8)
            self.assertEqual(len(weights["classes"]), 8)


if __name__ == "__main__":
    unittest.main()
