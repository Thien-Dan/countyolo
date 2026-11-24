"""Train the counting phase of the paper-aligned YOLO-World replacement."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
FILES=['models/yoloworld_counting.py','models/yoloworld_countvid.py','datasets/countvid_protocol.py',
       'video/countvid_yoloworld.py','tools/train_yoloworld_countvid.py','tools/test_yoloworld_countvid.py',
       'tools/run_countvid_yoloworld.py','outputs/countvid_detect/CountVid/count_in_videos.py',
       'outputs/countvid_detect/CountVid/datasets/transforms.py']
RUNNER=r'''
import base64,glob,hashlib,json,os,pathlib,subprocess,sys,time,traceback
W=pathlib.Path('/kaggle/working');os.chdir(W);sys.path.insert(0,str(W))
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
FILES=__FILES__
SPEC=__SPEC__
for name,payload in FILES.items():
    path=W/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(payload))
for name in ['models','datasets','video','tools']:(W/name/'__init__.py').touch()
def status(state,**more):
    p=W/'job_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(dict(state=state,spec=SPEC,**more),indent=2));q.replace(p)
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def run(args):
    print('$',' '.join(map(str,args)),flush=True);subprocess.run(list(map(str,args)),check=True)
try:
    markers=sorted({pathlib.Path(p).resolve() for p in glob.glob('/kaggle/input/**/countvid_protocol.json',recursive=True)})
    splits=sorted({pathlib.Path(p).resolve() for p in glob.glob('/kaggle/input/**/Train_Test_Val_FSC_147.json',recursive=True)})
    if len(markers)!=1 or len(splits)!=1:raise RuntimeError('Attach exactly one author-protocol dataset and one FSC147')
    protocol=markers[0].parent;data=splits[0].parent
    for name,expected in SPEC['protocol_sha256'].items():
        if digest(protocol/name)!=expected:raise RuntimeError('Author protocol mismatch: '+name)
    os.environ['COUNTVID_PROTOCOL_ROOT']=str(protocol);os.environ['FSC147_ROOT']=str(data)
    status('setup');run(['nvidia-smi'])
    run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel==0.45.1'])
    run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.177','numpy<2.3','scipy','opencv-python'])
    run([sys.executable,'-m','pip','install','--no-build-isolation','git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33'])
    import torch,requests,importlib.metadata
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    weights=W/'yolov8s-worldv2.pt'
    response=requests.get(SPEC['weights_url']+'?download='+str(time.time_ns()),timeout=(10,120));response.raise_for_status();weights.write_bytes(response.content)
    if digest(weights)!=SPEC['weights_sha256']:raise RuntimeError('YOLO weight mismatch')
    (W/'run_manifest.json').write_text(json.dumps(dict(spec=SPEC,gpu=torch.cuda.get_device_name(),
        versions={n:importlib.metadata.version(n) for n in ['torch','torchvision','ultralytics','scipy']},
        source_sha256={n:hashlib.sha256(base64.b64decode(p)).hexdigest() for n,p in FILES.items()}),indent=2))
    status('tests');run([sys.executable,'-m','unittest','tools.test_yoloworld_countvid'])
    common=['--data-root',data,'--protocol-root',protocol,'--weights',weights]
    status('two_phase_gradient_smoke')
    run([sys.executable,'-u','tools/train_yoloworld_countvid.py',*common,'--output',W/'two_phase_smoke','--smoke'])
    if not json.loads((W/'two_phase_smoke/result.json').read_text())['frozen_encoders_unchanged']:raise RuntimeError('Frozen encoder check failed')
    status('counting_phase_training')
    run([sys.executable,'-u','tools/train_yoloworld_countvid.py',*common,'--output',W/'counting_phase','--phase','point','--epochs','30'])
    status('counting_phase_complete',next_stage='Box phase must initialize from counting_phase/point/best.pth; not yet a video-ready box checkpoint')
except BaseException as exc:
    status('failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();raise
'''


def prepare(user='thiend27'):
    slug='countyolo-countvid-yoloworld-point-20261119';folder=ROOT/'kaggle_video_run'/user/slug;folder.mkdir(parents=True,exist_ok=True)
    protocol=ROOT/f'kaggle_adapt_cache/{user}/countvid_protocol'
    spec=dict(purpose='CountVid detector replacement, correct counting warm-start phase before box adaptation',
        architecture='yoloworld-countvid-multimodal-v1',phase='point',epochs=30,
        full_curriculum='counting30 -> counting-best checkpoint -> box30 -> unchanged CountVid/SAM2 video',
        protocol_dataset=f'{user}/countyolo-countvid-protocol-20261119',
        protocol_sha256=json.loads((protocol/'countvid_protocol.json').read_text())['files'],
        weights_url='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt',
        weights_sha256=hashlib.sha256((ROOT/'outputs/yolo_detector_compare/20261119/yolov8s-worldv2.pt').read_bytes()).hexdigest(),
        no_legacy_checkpoint=True,test_used=False,production_promoted=False)
    payload={name:base64.b64encode((ROOT/name).read_bytes()).decode() for name in FILES}
    source=RUNNER.replace('__FILES__',repr(payload)).replace('__SPEC__',repr(spec));compile(source,'run.py','exec')
    if len(source.encode())>1_000_000:raise RuntimeError('Payload too large')
    (folder/'run.py').write_text(source,encoding='utf-8',newline='\n')
    metadata=dict(id=f'{user}/{slug}',title=slug,code_file='run.py',language='python',kernel_type='script',is_private=True,
        enable_gpu=True,enable_internet=True,dataset_sources=['xuncngng/fsc147-0',spec['protocol_dataset']],kernel_sources=[],competition_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    return dict(ref=metadata['id'],folder=str(folder),spec=spec,code_sha256=hashlib.sha256(source.encode()).hexdigest(),state='prepared')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--push',action='store_true');args=p.parse_args()
    ledger=ROOT/'experiments/countvid_yoloworld_paper_aligned_kaggle_20261119.json'
    if ledger.exists() and json.loads(ledger.read_text()).get('state')=='submitted':raise RuntimeError('Already submitted')
    job=prepare();ledger.write_text(json.dumps(job,indent=2))
    if args.push:
        from tools.kaggle_watch import accounts,api_for
        api=api_for('thiend27',accounts()['thiend27']);response=api.kernels_push(job['folder'])
        if response.error:raise RuntimeError(response.error)
        job.update(state='submitted',version_number=response.version_number,submitted_at=datetime.now(timezone.utc).isoformat(),remote_state=str(api.kernels_status(job['ref']).status))
        ledger.write_text(json.dumps(job,indent=2))
    print(json.dumps({k:v for k,v in job.items() if k!='spec'},indent=2))


if __name__=='__main__':main()
