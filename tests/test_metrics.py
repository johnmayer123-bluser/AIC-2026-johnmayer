import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uavseg.metrics import confusion_matrix, iou_from_confusion


class MetricTests(unittest.TestCase):
    def test_perfect_prediction_ignores_zero(self):
        target = np.array([[0, 1], [2, 2]])
        matrix = confusion_matrix(target, target, num_classes=3, ignore_index=0)
        miou, per_class = iou_from_confusion(matrix, ignore_index=0)
        self.assertEqual(miou, 1.0)
        self.assertIsNone(per_class[0])
        self.assertEqual(per_class[1], 1.0)
        self.assertEqual(per_class[2], 1.0)

    def test_known_iou(self):
        target = np.array([[1, 1], [2, 2]])
        prediction = np.array([[1, 2], [2, 2]])
        matrix = confusion_matrix(prediction, target, num_classes=3, ignore_index=0)
        miou, per_class = iou_from_confusion(matrix, ignore_index=0)
        self.assertAlmostEqual(per_class[1], 0.5)
        self.assertAlmostEqual(per_class[2], 2 / 3)
        self.assertAlmostEqual(miou, (0.5 + 2 / 3) / 2)


if __name__ == "__main__":
    unittest.main()

