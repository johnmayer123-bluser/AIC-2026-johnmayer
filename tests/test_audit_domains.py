import sys
import json
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace

import numpy as np
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from audit_domains import extract, hash_distances, nearest, run


class DomainAuditTests(unittest.TestCase):
    def test_hash_distance(self):
        self.assertEqual(hash_distances(np.array([0],np.uint8), np.array([[0],[255],[1]],np.uint8)).tolist(), [0,8,1])

    def test_end_to_end_exact_and_output_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'images').mkdir(); (root/'test').mkdir()
            image=Image.fromarray(np.random.default_rng(3407).integers(0,256,(64,64,3),dtype=np.uint8))
            image.save(root/'images/a.png')
            meta=PngImagePlugin.PngInfo(); meta.add_text('note','different container, same pixels')
            image.save(root/'images/b.png',pnginfo=meta)
            Image.new('RGB',(64,64),'black').save(root/'test/t.png')
            a=extract(('train','a',root/'images/a.png')); b=extract(('val','b',root/'images/b.png'))
            self.assertNotEqual(a['file_sha256'],b['file_sha256'])
            self.assertEqual(a['rgb_sha256'],b['rgb_sha256'])
            n,c=nearest([b],[a]); self.assertEqual(n[0]['phash']['distance'],0); self.assertTrue(c[0]['exact_rgb'])
            (root/'split.json').write_text(json.dumps(dict(train=['a'],val=['b'])))
            args=Namespace(images=str(root/'images'),test_images=str(root/'test'),split=str(root/'split.json'),output=str(root/'out'),workers=1,threshold=8)
            report=run(args)
            self.assertEqual(report['exact_cross_split_groups'],1)
            self.assertIsNone(report['appearance']['test']['nonblack_brightness']['mean'])
            with self.assertRaises(FileExistsError): run(args)
            args.output=str(root/'overlap')
            (root/'split.json').write_text(json.dumps(dict(train=['a','b'],val=['b'])))
            with self.assertRaises(ValueError): run(args)


if __name__ == '__main__':
    unittest.main()
