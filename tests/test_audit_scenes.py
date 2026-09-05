import unittest

import numpy as np

from tools.audit_scenes import components, geometry, grouped_split, cv2
from tools.refine_scene_groups import expanded_pairs


class SceneTests(unittest.TestCase):
    def test_expanded_pairs_no_self_edges_and_monotonic(self):
        bows=np.eye(8,dtype=np.float32)
        small=expanded_pairs(bows,2)
        large=expanded_pairs(bows,5)
        self.assertTrue(small<=large)
        self.assertTrue(all(a<b for a,b in large))
        with self.assertRaises(ValueError):
            expanded_pairs(bows,8)

    def test_transitive_groups_and_singletons(self):
        self.assertEqual(components(['a','b','c','d'],[('b','c'),('a','b')]),[['a','b','c'],['d']])

    def test_group_split_conserves_all_images_and_groups(self):
        groups=[[f'{i:03d}',f'{i+100:03d}'] for i in range(20)]
        counts={n:np.arange(1,10,dtype=np.int64) for g in groups for n in g}
        first,balance=grouped_split(groups,counts,3407,trials=20)
        second,_=grouped_split(groups,counts,3407,trials=20)
        self.assertEqual(first,second)
        self.assertEqual(set(first['train'])|set(first['val']),set(counts))
        self.assertFalse(set(first['train'])&set(first['val']))
        for g in groups:
            self.assertIn(sum(n in first['val'] for n in g),(0,len(g)))
        self.assertEqual(np.array(balance['train_pixels']).tolist(),
                         (np.array(balance['overall_pixels'])-balance['val_pixels']).tolist())

    @unittest.skipIf(cv2 is None,'optional OpenCV audit dependency')
    def test_geometry_rotated_shifted_vs_unrelated(self):
        cv2.setNumThreads(1)
        rng=np.random.default_rng(42)
        image=np.zeros((512,512),np.uint8)
        for _ in range(180):
            x,y=rng.integers(20,492,size=2)
            cv2.circle(image,(int(x),int(y)),int(rng.integers(3,12)),int(rng.integers(40,255)),-1)
        def feature(im):
            points,desc=cv2.SIFT_create(nfeatures=256).detectAndCompute(im,None)
            return dict(xy=np.float32([k.pt for k in points]),desc=desc,gray=cv2.resize(im,(256,256)))
        affine=cv2.getRotationMatrix2D((256,256),15,1)
        affine[:,2]+=np.array([10,-8])
        transformed=cv2.warpAffine(image,affine,(512,512))
        match=geometry(feature(image),feature(transformed))
        self.assertEqual(match['status'],'strong_overlap',match)
        unrelated=rng.integers(0,256,(512,512),dtype=np.uint8)
        self.assertNotEqual(geometry(feature(image),feature(unrelated))['status'],'strong_overlap')


if __name__=='__main__':
    unittest.main()
