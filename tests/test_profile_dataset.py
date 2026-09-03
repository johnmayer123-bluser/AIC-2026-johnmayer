import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_dataset import choose_split, class_weights


class ProfileDatasetTests(unittest.TestCase):
    def test_split_is_deterministic_and_sized(self):
        descriptors = np.arange(120, dtype=np.float64).reshape(20, 6)
        first, first_score = choose_split(descriptors, val_count=4, trials=20, seed=3407)
        second, second_score = choose_split(descriptors, val_count=4, trials=20, seed=3407)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(first_score, second_score)

    def test_class_weights_ignore_zero_and_favor_rare_class(self):
        counts = np.array([10, 1000, 100, 10], dtype=np.int64)
        weights = class_weights(counts, ignore_index=0)
        self.assertEqual(weights[0], 0.0)
        self.assertGreater(weights[3], weights[2])
        self.assertGreater(weights[2], weights[1])


if __name__ == "__main__":
    unittest.main()
