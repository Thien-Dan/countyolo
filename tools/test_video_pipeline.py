"""Small CPU checks; geometry checks also run in Kaggle before model loading."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video.eval_common import atomic_json, summarize


class SAM3LifecycleTests(unittest.TestCase):
    @staticmethod
    def record(frame, ids):
        return dict(frame=frame, ids=ids, boxes=[[i, 0, i+1, 1] for i in ids])

    def test_aligned_job_pins_audited_version_and_exposes_precision(self):
        from launchers.kaggle_video_launch import prepare
        with tempfile.TemporaryDirectory() as d:
            job = prepare('nadthien','sam3_aligned_fp32','penguins',['penguins_1'],out_dir=d)
            spec = job['spec']
            self.assertEqual(spec['tracker_profile'], 'aligned')
            self.assertEqual(spec['precision'], 'fp32')
            self.assertEqual(spec['output_conf'], 0.5)
            script = (Path(d)/'run.py').read_text(encoding='utf-8')
            self.assertIn('ultralytics==8.4.204', script)
            self.assertIn('sam3_checkpoint_sha256', script)
            self.assertLess(len(script.encode()), 1_000_000)

    def test_cross_account_job_preserves_explicit_private_inputs(self):
        from launchers.kaggle_video_launch import prepare
        from tools.video_tasks import refresh_prepared
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            job = prepare('thiend27','sam3_aligned','penguins',['penguins_1'],out_dir=folder/'job',
                          sam3_dataset='nadthien/sam3-weights',data_dataset='nadthien/videocount-benchmark')
            ledger = folder/'tasks.json'
            atomic_json(ledger, dict(tasks=[dict(**job,state='prepared')]))
            refresh_prepared(ledger)
            refreshed = json.loads(ledger.read_text(encoding='utf-8'))['tasks'][0]
            self.assertEqual(refreshed['folder'], job['folder'])
            self.assertEqual(refreshed['spec']['data_dataset'],'nadthien/videocount-benchmark')
            metadata = json.loads((folder/'job/kernel-metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['dataset_sources'], ['nadthien/videocount-benchmark','nadthien/sam3-weights'])
            self.assertTrue(metadata['is_private'])

    def test_object_batching_preserves_all_ids_masks_order_and_existing_states(self):
        from video.sam3_policy import add_object_batches
        ids = list(range(19))
        masks = [f'mask-{i}' for i in ids]
        states = [('existing',)]
        calls = []
        def add(frame, nframes, batch_ids, batch_masks, current):
            calls.append((frame,nframes,list(batch_ids),list(batch_masks)))
            current.append(tuple(batch_ids))
            return current
        result = add_object_batches(add,5,30,ids,masks,states,8)
        self.assertIs(result,states)
        self.assertEqual([len(c[2]) for c in calls],[8,8,3])
        self.assertEqual([i for c in calls for i in c[2]],ids)
        self.assertEqual([m for c in calls for m in c[3]],masks)
        self.assertEqual(result[0],('existing',))
        self.assertTrue(all(c[:2]==(5,30) for c in calls))
        calls.clear()
        add_object_batches(add,5,30,ids,masks,states,0)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][2],ids)
        with self.assertRaises(ValueError):
            add_object_batches(add,5,30,ids,masks[:-1],states,8)

    def test_hotstart_revokes_duplicate_in_pending_frames(self):
        from video.sam3_policy import DelayedTrackFrames
        q = DelayedTrackFrames(3)
        self.assertEqual(q.push(self.record(0, [7, 8])), [])
        self.assertEqual(q.push(self.record(1, [7, 8])), [])
        first = q.push(self.record(2, [7]), removed=[8])
        self.assertEqual(first[0]['ids'], [7])
        self.assertEqual(first[0]['boxes'], [[7, 0, 8, 1]])
        tail = q.finish()
        self.assertEqual([r['frame'] for r in first+tail], [0, 1, 2])
        self.assertEqual([r['ids'] for r in tail], [[7], [7]])
        self.assertEqual(q.finish(), [])

    def test_track_leaving_later_does_not_erase_emitted_sighting(self):
        from video.sam3_policy import DelayedTrackFrames
        q = DelayedTrackFrames(2)
        q.push(self.record(0, [9]))
        emitted = q.push(self.record(1, [9]))
        emitted += q.push(self.record(2, []), removed=[9])
        emitted += q.finish()
        self.assertEqual({i for r in emitted for i in r['ids']}, {9})
        self.assertEqual(emitted[1]['ids'], [])

    def test_short_clip_flush_and_zero_delay(self):
        from video.sam3_policy import DelayedTrackFrames
        q = DelayedTrackFrames(15)
        q.push(self.record(0, [1]))
        q.push(self.record(1, [1, 2]), removed=[1])
        self.assertEqual([r['ids'] for r in q.finish()], [[], [2]])
        q = DelayedTrackFrames(0)
        self.assertEqual(q.push(self.record(0, [3]))[0]['ids'], [3])


class RecordsTests(unittest.TestCase):
    def test_metrics_and_incomplete_are_explicit(self):
        rows = [dict(status='complete', gt=10, pred=12, sec=2, n_frames=4),
                dict(status='complete', gt=20, pred=17, sec=3, n_frames=6)]
        r = summarize(rows, 2)
        self.assertTrue(r['complete'])
        self.assertEqual(r['mae'], 2.5)
        self.assertEqual(r['bias'], -0.5)
        self.assertAlmostEqual(r['wape'], 5/30)
        self.assertEqual(r['fps'], 2)
        self.assertFalse(summarize(rows, 3)['complete'])
        self.assertIsNone(summarize([], 2)['mae'])

    def test_atomic_json_retains_unicode(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'results.json'
            atomic_json(path, {'prompt': 'đếm vật'})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['prompt'], 'đếm vật')
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_countvid_preserves_two_prompts_and_partial_failure(self):
        from countvid_ext import run_videocount as runner
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            cv = folder/'CountVid'
            cv.mkdir()
            script = '''import argparse,json
p=argparse.ArgumentParser()
p.add_argument('--video_dir');p.add_argument('--input_text');p.add_argument('--output_file')
a,_=p.parse_known_args()
prior=json.load(open(a.output_file))
assert isinstance(prior,dict)
if a.input_text=='fail': raise RuntimeError('deliberate failure')
json.dump({a.video_dir:{a.input_text:7}},open(a.output_file,'w'))
'''
            (cv/'count_in_videos.py').write_text(script)
            frames = folder/'frames'/'v1'
            frames.mkdir(parents=True)
            rows = [dict(case_id=text, video='v1', text=text, gt=8, frames=[str(frames/'1.jpg')],
                         frame_sha256='same', frame_manifest=['1.jpg']) for text in ['first', 'second']]
            out = folder/'results.json'
            argv = ['run', '--countvid_dir',str(cv),'--data_root',str(folder),'--dataset','penguins',
                    '--name','test','--results',str(out)]
            with patch.object(sys,'argv',argv), patch.object(runner,'cases',return_value=rows):
                runner.main()
            result = json.loads(out.read_text())['penguins/test']
            self.assertTrue(result['complete'])
            self.assertEqual(len(result['rows']),2)
            rows[1]['text']='fail'
            with patch.object(sys,'argv',argv), patch.object(runner,'cases',return_value=rows):
                with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    runner.main()
            result = json.loads(out.read_text())['penguins/test']
            self.assertFalse(result['complete'])
            self.assertEqual(result['n'],1)
            self.assertEqual(result['failed'],1)
            self.assertEqual(result['rows'][0]['pred'],7)


    def test_packaged_script_respects_limit_and_hashes_disk_bytes(self):
        import hashlib
        from launchers.kaggle_video_launch import prepare
        with tempfile.TemporaryDirectory() as d:
            read_bytes=Path.read_bytes
            def asset_bytes(path):
                if path.name=='density_head_best_vit_exnoise.pth':
                    return b'isolated packaging-test checkpoint'
                return read_bytes(path)
            with patch.object(Path,'read_bytes',asset_bytes):
                job=prepare('testuser','countyolo_fusion','penguins',['penguins_1'],out_dir=d)
            code=(Path(d)/'run.py').read_bytes()
            self.assertLess(len(code),1_000_000)
            self.assertEqual(hashlib.sha256(code).hexdigest(),job['code_sha256'])
            self.assertLessEqual(len(job['ref'].split('/')[1]),50)
            metadata=json.loads((Path(d)/'kernel-metadata.json').read_text())
            self.assertIn('testuser/countyolo-exnoise',metadata['dataset_sources'])
            compile(code,'run.py','exec')

    def test_quota_rejection_keeps_queue_and_stops_additional_submissions(self):
        import hashlib
        from types import SimpleNamespace
        from tools.video_tasks import tick
        import importlib
        watcher=importlib.import_module('kaggle_watch')
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            (folder/'run.py').write_text('print(1)')
            digest=hashlib.sha256((folder/'run.py').read_bytes()).hexdigest()
            tasks=[dict(ref='user/job-'+str(i),folder=str(folder),code_sha256=digest,state='prepared',attempts=0)
                   for i in range(2)]
            ledger=folder/'tasks.json'
            atomic_json(ledger,dict(tasks=tasks,account_errors={}))
            api=SimpleNamespace(kernels_list=lambda **kw: [],
                                kernels_push=unittest.mock.Mock(return_value=SimpleNamespace(error='Maximum batch GPU session count of 2 reached.')))
            with patch.object(watcher,'accounts',return_value={'user':'dummy'}), patch.object(watcher,'api_for',return_value=api):
                result=tick(ledger,max_active=2)
            self.assertEqual(api.kernels_push.call_count,1)
            self.assertTrue(all(t['state']=='prepared' for t in result['tasks']))


    def test_full_stage_rejects_mismatched_pilot_frames(self):
        from tools.video_tasks import expand_full
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            tasks=[]
            for i,engine in enumerate(['baseline','countyolo_fusion']):
                output=folder/engine
                output.mkdir()
                row=dict(case_id='samecase',gt=5,pred=5,frame_sha256=str(i))
                atomic_json(output/'results.json',{f'penguins/{engine}':dict(rows=[row])})
                tasks.append(dict(state='complete',output=str(output),spec=dict(dataset='penguins',engine=engine)))
            path=folder/'tasks.json'
            atomic_json(path,dict(phase='pilot',tasks=tasks))
            with self.assertRaisesRegex(ValueError,'sampling'):
                expand_full(path,folder)
            self.assertEqual(json.loads(path.read_text())['phase'],'pilot')


    def test_queue_lock_releases_after_conflict(self):
        from tools.video_tasks import QueueBusy,queue_lock
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'queue.json'
            with queue_lock(path):
                with self.assertRaises(QueueBusy):
                    with queue_lock(path):
                        pass
            with queue_lock(path):
                pass


    @unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy needed for exact CountVid sampling')
    def test_cached_mot20_is_not_sampled_twice(self):
        import shutil
        from video.eval_common import cases
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'original'
            folder=root/'MOT20-Count'
            (folder/'anno').mkdir(parents=True)
            (folder/'frames/v1').mkdir(parents=True)
            atomic_json(folder/'anno/MOT20-count-gt.json',{'v1':{'human':3}})
            for i in range(17):
                (folder/'frames/v1'/f'{i+1:06d}.jpg').write_bytes(b'unchanged fixture bytes')
            original=cases(root,'mot20')[0]
            cached=Path(d)/'cached'
            target=cached/'MOT20-Count'
            shutil.copytree(folder/'anno',target/'anno')
            (target/'frames/v1').mkdir(parents=True)
            for source in original['frames']:
                shutil.copyfile(source,target/'frames/v1'/Path(source).name)
            atomic_json(cached/'countyolo_sampling.json',dict(protocol='countvid-text-v1',pre_sampled=True,
                videos={'mot20':{'v1':dict(frame_names=original['frame_manifest'],original_downsample=25/3)}}))
            result=cases(cached,'mot20')[0]
            self.assertEqual(result['frame_sha256'],original['frame_sha256'])
            self.assertEqual(len(result['frames']),3)
            self.assertEqual(result['inference_downsample'],1)
            Path(result['frames'][0]).unlink()
            with self.assertRaisesRegex(ValueError,'Cached frame selection changed'):
                cases(cached,'mot20')


    def test_baseline_attaches_verified_checkpoint_cache(self):
        from launchers.kaggle_video_launch import prepare,COUNTGD_BOX_SHA256
        for engine in ('baseline','baseline_batch30'):
            with tempfile.TemporaryDirectory() as d:
                job=prepare('ttdan2006',engine,'mot20',['MOT20-01'],out_dir=d)
                metadata=json.loads((Path(d)/'kernel-metadata.json').read_text())
                self.assertIn('ttdan2006/countvid-weights',metadata['dataset_sources'])
                self.assertNotIn('ttdan2006/sam3-weights',metadata['dataset_sources'])
                self.assertEqual(job['spec']['countgd_checkpoint_sha256'],COUNTGD_BOX_SHA256)
                code=(Path(d)/'run.py').read_text()
                self.assertNotIn("run(['gdown', '1bw",code)
                self.assertIn('CountGD-Box checkpoint checksum mismatch',code)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Geometry checks run in Kaggle (torch unavailable locally)')
class GeometryTests(unittest.TestCase):
    def test_sam3_absence_fallback_does_not_force_density_count_to_zero(self):
        import torch
        from PIL import Image
        from unittest.mock import Mock
        from countvid_ext.countyolo_stage1 import CountYOLOStage1
        stage=CountYOLOStage1.__new__(CountYOLOStage1)
        stage.sam3=None
        stage.calls=0
        stage.present=False
        stage.fusion=False
        stage.model=object()
        stage.device='cpu'
        stage.ttn_thr=0
        stage.ttn_grid=2
        stage.min_mass=0
        stage.protos=None
        proposals=torch.tensor([[1.,1.,9.,9.]])
        stage._tiled=Mock(return_value=(1.,{},proposals,torch.tensor([0.8])))
        image=Image.new('RGB',(20,20))
        stage.absence_policy='block'
        self.assertEqual(len(stage.boxes(image,'crystal')[0]),0)
        stage._tiled.assert_not_called()
        stage.absence_policy='fallback'
        boxes,_,count=stage.boxes(image,'crystal')
        self.assertEqual(len(boxes),1)
        self.assertEqual(count,1.)
        stage._tiled.assert_called_once()

    def test_detector_replaces_agreeing_peak_retains_other_object(self):
        import torch
        from countvid_ext.proposals import fuse_proposals
        peaks=torch.tensor([[0,0,10,10],[40,40,50,50]],dtype=torch.float32)
        det=torch.tensor([[1,1,11,11]],dtype=torch.float32)
        boxes,_=fuse_proposals(peaks,torch.tensor([0.8,0.9]),det,torch.tensor([0.7]))
        self.assertEqual(len(boxes),2)
        self.assertTrue(torch.equal(boxes[0],det[0]))
        self.assertTrue(torch.equal(boxes[1],peaks[1]))

    def test_empty_and_duplicate_detector_boxes(self):
        import torch
        from countvid_ext.proposals import fuse_proposals, decode_yolo
        empty=torch.zeros((0,4))
        boxes,_=fuse_proposals(empty,torch.zeros(0),torch.tensor([[0,0,10,10],[0,0,10,10.]]),torch.tensor([0.7,0.6]))
        self.assertEqual(len(boxes),1)
        boxes,_=fuse_proposals(empty,torch.zeros(0),empty,torch.zeros(0))
        self.assertEqual(tuple(boxes.shape),(0,4))
        d=torch.tensor([[[320.],[320.],[64.],[32.],[0.9]]])
        b,s=decode_yolo(d,1280,640)
        self.assertTrue(torch.equal(b,torch.tensor([[576.,304.,704.,336.]])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
