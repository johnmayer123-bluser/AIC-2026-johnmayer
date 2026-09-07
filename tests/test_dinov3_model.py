import sys
import unittest
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from uavseg.model import (
    DinoV3BoundarySegmenter,
    default_dinov3_blocks,
    parse_dinov3_blocks,
)


class FakeDinoV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 384, kernel_size=16, stride=16)

    def get_intermediate_layers(self, pixels, *, n, reshape, norm):
        self.request = (tuple(n), reshape, norm)
        feature = self.patch(pixels)
        return tuple(feature + float(index) for index in range(len(n)))


class DinoV3ModelTests(unittest.TestCase):
    def test_blocks_have_safe_defaults_and_validation(self):
        self.assertEqual(default_dinov3_blocks("vits16plus"), (2, 5, 8, 11))
        self.assertEqual(parse_dinov3_blocks("1,3,7,10", "vits16plus"), (1, 3, 7, 10))
        self.assertEqual(default_dinov3_blocks("vitl16"), (5, 11, 17, 23))
        with self.assertRaises(ValueError):
            parse_dinov3_blocks("2,5,8", "vits16plus")
        with self.assertRaises(ValueError):
            parse_dinov3_blocks("2,5,8,12", "vits16plus")

    def test_forward_and_portable_checkpoint_config(self):
        encoder = FakeDinoV3()
        model = DinoV3BoundarySegmenter(
            encoder,
            variant="vits16plus",
            decoder_channels=16,
            dinov3_source=None,
            pretrained_sha256="abc",
        )
        output = model(torch.randn(1, 3, 64, 80))
        self.assertEqual(tuple(output["logits"].shape), (1, 9, 16, 20))
        self.assertEqual(tuple(output["boundary_logits"].shape), (1, 1, 16, 20))
        self.assertEqual(encoder.request, ((2, 5, 8, 11), True, True))
        config = model.export_config()
        self.assertEqual(config["architecture"], "dinov3_boundary")
        self.assertEqual(config["variant"], "vits16plus")
        self.assertEqual(config["feature_blocks"], [2, 5, 8, 11])
        self.assertEqual(config["pretrained_sha256"], "abc")
        self.assertFalse(model.gradient_checkpointing_enable())

    def test_rejects_non_patch_aligned_input(self):
        model = DinoV3BoundarySegmenter(
            FakeDinoV3(), variant="vits16plus", decoder_channels=16
        )
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 63, 64))

    def test_resume_accepts_explicit_and_default_equivalent_blocks(self):
        from train import validate_resume_args

        current = Namespace(
            architecture="dinov3",
            dinov3_variant="vits16plus",
            dinov3_blocks=None,
            crop_strategy="legacy",
        )
        validate_resume_args(
            current,
            {
                "architecture": "dinov3",
                "dinov3_variant": "vits16plus",
                "dinov3_blocks": "2,5,8,11",
                "crop_strategy": "legacy",
            },
        )


if __name__ == "__main__":
    unittest.main()
