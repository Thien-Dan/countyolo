"""Submit one private GPU evaluation: SAM3 image detector on FSC147 val + VID frames."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
FILES=['engines/sam3_counter.py','video/eval_common.py','tools/eval_sam3_detector_only.py',
       'tools/test_sam3_detector_eval.py']
RUNNER=r'''
import base64,glob,hashlib,importlib.metadata,json,os,pathlib,subprocess,sys,traceback
W=pathlib.Path('/kaggle/working');os.chdir(W);sys.path.insert(0,str(W))
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
FILES=__FILES__
SPEC=__SPEC__
for name,payload in FILES.items():
    path=W/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(payload))
for name in ['engines','video','tools']:(W/name/'__init__.py').touch()
def status(state,**kw):
    path=W/'job_status.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(state=state,spec=SPEC,**kw),indent=2));os.replace(temp,path)
def run(args):
    print('$',' '.join(map(str,args)),flush=True);subprocess.run(list(map(str,args)),check=True)
def discover(name):return sorted({pathlib.Path(p).resolve() for p in glob.glob('/kaggle/input/**/'+name,recursive=True)})
def digest(path):
    with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
try:
    status('setup');run(['nvidia-smi'])
    run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.204','transformers==4.57.6','numpy<2.3','scipy','opencv-python'])
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Kaggle GPU required')
    weights=discover('sam3.pt');markers=discover('countyolo_sampling.json');splits=discover('Train_Test_Val_FSC_147.json')
    if len(weights)!=1 or len(markers)!=1 or len(splits)!=1:raise RuntimeError('Expected one SAM3, FSC147 and VideoCount input')
    if digest(weights[0])!=SPEC['sam3_sha256']:raise RuntimeError('SAM3 hash mismatch')
    fsc=splits[0].parent;video=markers[0].parent;sampling=json.loads(markers[0].read_text())
    for name,expected in SPEC['fsc_metadata_sha256'].items():
        if digest(fsc/name)!=expected:raise RuntimeError('FSC metadata mismatch: '+name)
    from video.eval_common import DATASETS,cases,atomic_json,frame_count_annotation
    checked=0;coverage=[]
    for dataset,(sub,_,_) in DATASETS.items():
        labels_name=frame_count_annotation(dataset);labels_path=video/sub/'anno'/labels_name
        if digest(labels_path)!=SPEC['video_gt_sha256'][dataset]:raise RuntimeError('Video frame GT mismatch: '+dataset)
        labels=json.loads(labels_path.read_text());scored=0;missing=[]
        for case in cases(video,dataset):
            for filename in case['frames']:
                path=pathlib.Path(filename);relative='VideoCount/'+path.relative_to(video).as_posix()
                if digest(path)!=sampling['files_sha256'][relative]:raise RuntimeError('Corrupt frame: '+relative)
                checked+=1
                if path.name in labels.get(case['video'],{}).get(case['text'],{}):scored+=1
                else:missing.append(dict(video=case['video'],frame=path.name))
        coverage.append(dict(dataset=dataset,scored=scored,missing=missing))
    if checked!=1252:raise RuntimeError('Wrong VideoCount frame coverage')
    run([sys.executable,'-m','unittest','tools.test_sam3_detector_eval'])
    atomic_json(W/'run_manifest.json',dict(spec=SPEC,video_frames_verified=checked,video_gt_coverage=coverage,
        sam3_sha256=digest(weights[0]),sampling_manifest_sha256=digest(markers[0]),
        packages={n:importlib.metadata.version(n) for n in ['torch','torchvision','ultralytics','transformers']},gpu=torch.cuda.get_device_name(),
        source_sha256={n:hashlib.sha256(base64.b64decode(v)).hexdigest() for n,v in FILES.items()}))
    status('running')
    run([sys.executable,'-u',W/'tools/eval_sam3_detector_only.py','--fsc-root',fsc,
         '--video-root',video,'--weights',weights[0],'--output',W/'detector'])
    result=json.loads((W/'detector/results.json').read_text())
    if not result['complete']:raise RuntimeError('Incomplete detector evaluation')
    status('complete')
except BaseException as exc:
    status('failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();raise
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--push',action='store_true')
    p.add_argument('--username',default='nadthien')
    p.add_argument('--retry-error',action='store_true');a=p.parse_args()
    slug='countyolo-sam3-detector-fsc-vid-20261119'
    ledger=ROOT/'experiments/sam3_detector_only_kaggle_20261119.json'
    history=[]
    if ledger.exists():
        old=json.loads(ledger.read_text());history=old.get('attempt_history',[])
        if old.get('state')=='submitted':
            if not a.retry_error:raise RuntimeError('Already submitted; check existing job')
            from tools.kaggle_watch import accounts,api_for
            api=api_for(a.username,accounts()[a.username]);remote=str(api.kernels_status(old['ref']).status).split('.')[-1].lower()
            if remote!='error':raise RuntimeError('Retry requires remote ERROR, got '+remote)
            history.append(dict(code_sha256=old['code_sha256'],submitted_at=old.get('submitted_at'),remote_state=remote,
                                reason='Preserve uppercase MOT20 frame annotation filename on Linux'))
    from video.eval_common import DATASETS,frame_count_annotation
    previous=json.loads((ROOT/'experiments/sam3_identity_kaggle_20261118.json').read_text())
    fsc=ROOT/'data/FSC147'
    spec=dict(purpose='Detector only; no video tracker, association, training, tiling or threshold fitting',
        backend='Ultralytics8.4.204 SAM3SemanticPredictor image',fsc_split='val',fsc_images=1286,
        video_canonical_frames=1252,video_scored_canonical_frames=1251,mot20_person_alias_frames=784,
        conf=.5,precision='fp16',imgsz=1008,max_grid=1,sam3_sha256=previous['spec']['sam3_sha256'],
        fsc_metadata_sha256={n:hashlib.sha256((fsc/n).read_bytes()).hexdigest() for n in
            ['Train_Test_Val_FSC_147.json','annotation_FSC147_384.json','ImageClasses_FSC147.txt']},
        video_gt_sha256={d:hashlib.sha256((ROOT/'data/VideoCount'/sub/'anno'/frame_count_annotation(d)).read_bytes()).hexdigest() for d,(sub,_,_) in DATASETS.items()})
    payload={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in FILES}
    script=RUNNER.replace('__FILES__',repr(payload)).replace('__SPEC__',repr(spec));compile(script,'run.py','exec')
    if len(script.encode())>=1_000_000:raise RuntimeError('Payload too large')
    folder=ROOT/'kaggle_video_run'/a.username/slug;folder.mkdir(parents=True,exist_ok=True)
    (folder/'run.py').write_text(script,encoding='utf-8',newline='\n')
    metadata=dict(id=f'{a.username}/{slug}',title=slug,code_file='run.py',language='python',kernel_type='script',is_private=True,
        enable_gpu=True,enable_internet=True,dataset_sources=['nadthien/sam3-weights','nadthien/videocount-benchmark','xuncngng/fsc147-0'],competition_sources=[],kernel_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    job=dict(ref=metadata['id'],folder=str(folder),state='prepared',spec=spec,attempt_history=history,code_sha256=hashlib.sha256(script.encode()).hexdigest())
    ledger.write_text(json.dumps(job,indent=2),encoding='utf-8')
    if a.push:
        from tools.kaggle_watch import accounts,api_for
        api=api_for(a.username,accounts()[a.username]);response=api.kernels_push(str(folder))
        if getattr(response,'error',None):raise RuntimeError(response.error)
        job.update(state='submitted',submitted_at=datetime.now(timezone.utc).isoformat())
        ledger.write_text(json.dumps(job,indent=2),encoding='utf-8')
        job['remote_state']=str(api.kernels_status(job['ref']).status)
        ledger.write_text(json.dumps(job,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in job.items() if k!='spec'},indent=2),flush=True)


if __name__=='__main__':main()
