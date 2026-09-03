import importlib.util
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers is not installed")
class ModelTests(unittest.TestCase):
    def test_small_model_forward_and_checkpoint_config(self):
        from transformers import SegformerConfig, SegformerModel

        from uavseg.model import BoundaryAwareSegFormer

        config = SegformerConfig(
            hidden_sizes=[8, 16, 32, 64],
            depths=[1, 1, 1, 1],
            num_attention_heads=[1, 1, 2, 4],
            sr_ratios=[8, 4, 2, 1],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            mlp_ratios=[2, 2, 2, 2],
        )
        model = BoundaryAwareSegFormer(
            SegformerModel(config), num_classes=9, decoder_channels=16
        )
        outputs = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(outputs["logits"].shape), (1, 9, 16, 16))
        self.assertEqual(tuple(outputs["boundary_logits"].shape), (1, 1, 16, 16))
        restored = BoundaryAwareSegFormer.from_model_config(model.export_config())
        self.assertEqual(restored.num_classes, 9)
        self.assertIsInstance(model.gradient_checkpointing_enable(), bool)


if __name__ == "__main__":
    unittest.main()
