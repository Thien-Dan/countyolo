"""CPU protocol/metric tests only; never loads SAM3 weights."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from tools.eval_sam3_detector_only import summary,point_match,run_cases


class DetectorProtocolTests(unittest.TestCase):
    def test_annotation_names_preserve_upstream_case(self):
        from video.eval_common import DATASETS,frame_count_annotation
        expected={'mot20':'MOT20-frame-level-counts-gt.json',
                  'penguins':'penguins-frame-level-counts-gt.json','crystals':'crystals-frame-level-counts-gt.json'}
        for dataset,name in expected.items():
            self.assertEqual(frame_count_annotation(dataset),name)
            folder=Path(__file__).resolve().parents[1]/'data/VideoCount'/DATASETS[dataset][0]/'anno'
            if folder.exists():self.assertIn(name,[p.name for p in folder.iterdir()])

    def test_missing_gt_is_not_zero(self):
        r=summary([dict(status='complete',gt=None,pred=7),dict(status='complete',gt=4,pred=6)],2)
        self.assertEqual(r['scored'],1);self.assertEqual(r['missing_gt'],1)
        self.assertEqual(r['mae'],2);self.assertTrue(r['complete'])

    def test_failure_and_partial_results_not_complete(self):
        self.assertFalse(summary([dict(status='failed',gt=2,pred=None)],1)['complete'])
        self.assertFalse(summary([dict(status='complete',gt=2,pred=2)],2)['complete'])

    def test_one_to_one_duplicate_boxes_cannot_double_match(self):
        r=point_match([[0,0,2,2],[0,0,2,2]],[[1,1]],(640,640),8)
        self.assertEqual(r,dict(tp=1,fp=1,fn=0))

    def test_native_coordinates_scaled_to640(self):
        r=point_match([[0,0,20,20]],[[12,10]],(128,128),8)
        self.assertEqual(r['tp'],0)
        self.assertEqual(point_match([[0,0,20,20]],[[12,10]],(128,128),16)['tp'],1)

    def test_rmse_bias_and_empty(self):
        r=summary([dict(status='complete',gt=10,pred=12),dict(status='complete',gt=20,pred=17)],2)
        self.assertEqual(r['mae'],2.5);self.assertEqual(r['bias'],-.5)
        self.assertAlmostEqual(r['rmse'],(6.5)**.5)
        self.assertIsNone(summary([],0)['mae'])

    def test_detector_only_records_independent_frames_without_tracker(self):
        class Counter:
            def __init__(self):self.calls=[]
            def count_image(self,image,prompt):
                self.calls.append(prompt)
                return dict(count=1,grid=1,boxes=np.array([[0.,0.,2.,2.]]),scores=np.array([.9]),numerics={})
            def count_video(self,*args,**kwargs):raise AssertionError('Tracker must never run')
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);path=output/'frame.png';Image.new('RGB',(10,10)).save(path)
            records=[dict(image=path.name,path=str(path),prompt='human',arm='canonical',gt=None),
                     dict(image=path.name,path=str(path),prompt='person',arm='person_alias',gt=3)]
            counter=Counter();rows=run_cases(records,counter,output,'fixture')
            self.assertEqual(counter.calls,['human','person'])
            self.assertEqual(rows[0]['gt'],None);self.assertEqual(rows[1]['pred'],1)
            self.assertTrue(summary(rows,2)['complete'])
            self.assertEqual(summary(rows,2)['mae'],2)


if __name__=='__main__':unittest.main()
