import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uavseg.losses import boundary_target, soft_dice_loss


class LossTests(unittest.TestCase):
    def test_perfect_dice_ignores_zero(self):
        labels = torch.tensor([[[0, 1], [2, 2]]])
        logits = torch.full((1, 3, 2, 2), -20.0)
        logits.scatter_(1, labels.unsqueeze(1), 20.0)
        self.assertLess(float(soft_dice_loss(logits, labels, ignore_index=0)), 1e-5)

    def test_boundary_ignores_transitions_touching_ignore(self):
        labels = torch.tensor([[[0, 1, 1], [0, 1, 2], [0, 1, 2]]])
        edges, valid = boundary_target(labels, ignore_index=0)
        self.assertEqual(tuple(edges.shape), (1, 1, 3, 3))
        self.assertEqual(float(valid[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(edges[0, 0, 1, 2]), 1.0)


if __name__ == "__main__":
    unittest.main()
