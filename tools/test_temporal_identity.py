import unittest
from video.identity import TemporalIdentity, IdentityConfig
from video.sam3_policy import DelayedTrackFrames


class IdentityTests(unittest.TestCase):
    def add(self, tracker, frames, raw_id, box=(40,40,50,60), feature=(1,0,0)):
        # Helper batches observations for the joint chronological replay.
        return [(f, raw_id, box, feature) for f in frames]

    def replay(self, observations, config=None):
        tracker=TemporalIdentity(config)
        for frame in sorted(set(o[0] for o in observations)):
            group=[o for o in observations if o[0]==frame]
            tracker.observe(frame,[o[1] for o in group],[o[2] for o in group],
                            [o[3] for o in group],(100,100))
        return tracker

    def test_short_occlusion_stitches_three_confirmations(self):
        rows=self.add(None,range(3),1)+self.add(None,range(5,8),2)
        result=self.replay(rows).finalize()
        self.assertEqual(result['baseline_count'],2);self.assertEqual(result['count'],1)
        self.assertEqual(result['identity_hits'][1],6)
        self.assertEqual(result['links'][0]['kind'],'occlusion')

    def test_reentry_requires_boundaries(self):
        rows=self.add(None,range(3),1,(0,40,10,60))+self.add(None,range(20,23),2,(90,40,100,60))
        result=self.replay(rows).finalize()
        self.assertEqual(result['count'],1);self.assertEqual(result['links'][0]['kind'],'reentry')
        rows=self.add(None,range(3),1)+self.add(None,range(20,23),2)
        self.assertEqual(self.replay(rows).finalize()['count'],2)

    def test_simultaneous_objects_never_merged(self):
        rows=self.add(None,range(5),1)+self.add(None,range(2,7),2)
        self.assertEqual(self.replay(rows).finalize()['count'],2)

    def test_lookalikes_remain_uncertain(self):
        rows=self.add(None,range(3),1)+self.add(None,range(3),2)+self.add(None,range(5,8),3)
        result=self.replay(rows).finalize()
        self.assertEqual(result['count'],3);self.assertEqual(len(result['uncertain']),2)

    def test_different_appearance_and_impossible_motion(self):
        old=self.add(None,range(3),1)
        for new in [self.add(None,range(5,8),2,feature=(0,1,0)),
                    self.add(None,range(5,8),2,box=(90,80,100,100))]:
            self.assertEqual(self.replay(old+new).finalize()['count'],2)

    def test_expired_and_insufficient_observations(self):
        old=self.add(None,range(3),1)
        for new in [self.add(None,range(100,103),2),self.add(None,range(5,7),2)]:
            self.assertEqual(self.replay(old+new).finalize()['count'],2)

    def test_missing_or_unstable_features_refuse_merge(self):
        old=self.add(None,range(3),1)
        new=self.add(None,range(5,8),2,feature=None)
        self.assertEqual(self.replay(old+new).finalize()['count'],2)
        new=[(5,2,(40,40,50,60),(1,0,0)),(6,2,(40,40,50,60),(0,1,0)),(7,2,(40,40,50,60),(1,0,0))]
        self.assertEqual(self.replay(old+new).finalize()['count'],2)

    def test_chained_tracklets_and_min_hits(self):
        rows=sum([self.add(None,range(start,start+3),i) for i,start in [(1,0),(2,5),(3,10)]],[])
        tracker=self.replay(rows)
        self.assertEqual(tracker.finalize(8)['count'],1)
        self.assertEqual(tracker.finalize(10)['count'],0)

    def test_long_reentry_live_lookalike_blocks_merge(self):
        rows=self.add(None,range(3),1,(0,40,10,60))+self.add(None,range(20,23),2,(90,40,100,60))
        rows+=self.add(None,range(10,25),3)
        self.assertEqual(self.replay(rows).finalize()['count'],3)

    def test_delay_filters_features_with_revoked_ids(self):
        q=DelayedTrackFrames(2)
        q.push(dict(frame=0,ids=[1,2],boxes=[[0]*4,[1]*4],identity_features=[[1,0],[0,1]]))
        row=q.push(dict(frame=1,ids=[],boxes=[],identity_features=[]),removed=[1])[0]
        self.assertEqual(row['ids'],[2]);self.assertEqual(row['identity_features'],[[0,1]])

    def test_invalid_input_and_empty(self):
        self.assertEqual(TemporalIdentity().finalize()['count'],0)
        t=TemporalIdentity();t.observe(0,[],[],[],(100,100))
        with self.assertRaises(ValueError):t.observe(0,[],[],[],(100,100))
        with self.assertRaises(ValueError):IdentityConfig(confirmations=1)

    def test_degenerate_mask_box_keeps_count_without_identity_evidence(self):
        t=TemporalIdentity();t.observe(0,[9],[[2,2,2,8]],[[1,0]],(100,100))
        self.assertEqual(t.finalize()['count'],1)
        self.assertIsNone(t.tracks[9][0][2])

    def test_engine_opt_in_and_default_count_share_native_tracks(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import numpy as np
        import torch
        from engines.sam3_counter import SAM3Counter
        rows=self.add(None,range(3),1)+self.add(None,range(5,8),2)
        class Boxes(SimpleNamespace):
            def __len__(self):return len(self.id)
        class Predictor:
            def __init__(self, **kwargs):pass
            def __call__(self, **kwargs):
                for frame in range(8):
                    group=[r for r in rows if r[0]==frame]
                    boxes=Boxes(is_track=True,id=torch.tensor([r[1] for r in group]),
                                          xyxy=torch.tensor([r[2] for r in group],dtype=torch.float32)) if group else None
                    yield SimpleNamespace(boxes=boxes,orig_img=np.zeros((100,100,3),dtype=np.uint8))
        counter=SAM3Counter.__new__(SAM3Counter);counter.conf=.3;counter.half=False;counter.weights='unused'
        with patch('ultralytics.models.sam.SAM3VideoSemanticPredictor',Predictor), \
             patch('video.identity_features.DINOIdentityFeatures') as encoder:
            encoder.return_value.side_effect=lambda rgb,boxes:[[1,0,0] for _ in boxes]
            native=counter.count_video('unused.avi','object')
            encoder.assert_not_called()
            result=counter.count_video('unused.avi','object',identity_mode='dinov3')
        self.assertEqual(native['count'],2);self.assertEqual(result['count'],1)
        self.assertEqual(native['track_hits'],result['track_hits'])
        self.assertEqual(result['baseline_count'],2)


if __name__=='__main__':unittest.main()
