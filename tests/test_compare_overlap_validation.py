import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.compare_overlap_validation import confusion


class OverlapValidationTests(unittest.TestCase):
    def test_confusion_separates_ignore_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pred=root/'pred';masks=root/'masks';pred.mkdir();masks.mkdir()
            Image.fromarray(np.array([[0,1],[2,2]],np.uint8)).save(masks/'a.png')
            Image.fromarray(np.array([[8,1],[2,1]],np.uint8)).save(pred/'a.png')
            result=confusion({'a'},pred,masks)
            self.assertEqual(result['valid_pixels'],3)
            self.assertAlmostEqual(result['per_class_iou'][1],.5)
            self.assertAlmostEqual(result['per_class_iou'][2],.5)
            self.assertIsNone(result['per_class_iou'][0])
            self.assertAlmostEqual(result['global_miou'],.5)


if __name__=='__main__': unittest.main()
