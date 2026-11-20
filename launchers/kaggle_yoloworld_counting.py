"""Package and submit the isolated CountGD-Box-style YOLO-World experiment."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FILES=['models/yoloworld_counting.py','datasets/yoloworld_counting.py',
       'datasets/fsc147_dataset.py','datasets/density_generator.py',
       'tools/train_yoloworld_counting.py','tools/test_yoloworld_counting.py',
       'utils/yoloworld_resume.py','tools/test_yoloworld_resume.py']
RUNNER=r'''
import base64,glob,hashlib,importlib.metadata,json,os,pathlib,subprocess,sys,time,traceback
W=pathlib.Path('/kaggle/working');os.chdir(W);sys.path.insert(0,str(W))
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
FILES=__FILES__
SPEC=__SPEC__
for name,payload in FILES.items():
    path=W/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(payload))
for folder in ['models','datasets','tools','utils']:(W/folder/'__init__.py').touch()
def status(state,**extra):
    path=W/'job_status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state,spec=SPEC,**extra),indent=2));os.replace(tmp,path)
def run(args):
    print('$',' '.join(map(str,args)),flush=True);subprocess.run(list(map(str,args)),check=True)
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
try:
    resume_args=[]
    if SPEC.get('resume_ref'):
        from utils.yoloworld_resume import find_resume_checkpoint
        checkpoint=find_resume_checkpoint('/kaggle/input',SPEC['resume_checkpoint_sha256'],SPEC['resume_best_sha256'])
        resume_args=['--resume',checkpoint]
        status('resume_input_verified',checkpoint=str(checkpoint))
    status('setup');run(['nvidia-smi'])
    run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel==0.45.1'])
    run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.177','numpy<2.3','scipy','opencv-python'])
    run([sys.executable,'-m','pip','install','--no-build-isolation','git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33'])
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    roots=sorted({pathlib.Path(p).resolve().parent for p in glob.glob('/kaggle/input/**/Train_Test_Val_FSC_147.json',recursive=True)})
    if len(roots)!=1:raise RuntimeError('Attach exactly one FSC147 dataset')
    data=roots[0]
    for name,expected in SPEC['fsc_metadata_sha256'].items():
        if digest(data/name)!=expected:raise RuntimeError('Dataset metadata mismatch: '+name)
    import requests
    weights=W/'yolov8s-worldv2.pt'
    with requests.get(SPEC['weights_url']+'?download='+str(time.time_ns()),stream=True,timeout=(15,180)) as response:
        response.raise_for_status()
        with weights.open('wb') as stream:
            for block in response.iter_content(1024*1024):stream.write(block)
    if digest(weights)!=SPEC['weights_sha256']:raise RuntimeError('Checkpoint SHA mismatch')
    (W/'run_manifest.json').write_text(json.dumps(dict(spec=SPEC,gpu=torch.cuda.get_device_name(),
        versions={n:importlib.metadata.version(n) for n in ['torch','torchvision','ultralytics','scipy']},
        source_sha256={n:hashlib.sha256(base64.b64decode(p)).hexdigest() for n,p in FILES.items()}),indent=2))
    status('tests');run([sys.executable,'-m','unittest','tools.test_yoloworld_counting','tools.test_yoloworld_resume'])
    common=['--data-root',data,'--weights',weights,'--exclude-train',W/'inputs/exclude_train.json',
        '--pilot-images',W/'inputs/pilot58.json']
    status('gradient_smoke')
    q_args=['--query-decoder','--num-queries',SPEC.get('num_queries',2200)] if SPEC.get('query_decoder') else []
    run([sys.executable,'-u','tools/train_yoloworld_counting.py',*common,'--output',W/'gradient_smoke','--smoke',*q_args])
    if not json.loads((W/'gradient_smoke/smoke.json').read_text())['passed']:raise RuntimeError('Gradient smoke failed')
    status('training_and_validation')
    run([sys.executable,'-u','tools/train_yoloworld_counting.py',*common,
        '--output',W/'yoloworld_counting','--epochs',SPEC['epochs'],'--accumulate','4',
        '--neg-weight',SPEC.get('neg_weight',0.15),'--neg-topk',SPEC.get('neg_topk',32),
        '--neighbor-sigma',SPEC.get('neighbor_sigma',0.004),'--neck-lr',SPEC.get('neck_lr',2e-5),
        '--stride-radius',SPEC.get('stride_radius',4.0),*q_args,*resume_args])
    status('complete')
except BaseException as exc:
    status('failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();raise
'''


def prepare(user='thiend27',resume_ref=None,query_decoder=False):
    slug='countyolo-yoloworld-prob-detr-20261120' if query_decoder else 'countyolo-yoloworld-countgd-box-20261120'
    if resume_ref:slug+='-resume'
    folder=ROOT/'kaggle_video_run'/user/slug
    folder.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/FSC147'
    weights=ROOT/'outputs/yolo_detector_compare/20261119/yolov8s-worldv2.pt'
    spec=dict(purpose='YOLO-World + PROB Self-Attention Query Decoder + Visual/Self-Exemplar Similarity Guidance (NMS-free 2200 queries + P3/P4 Cosine Similarity + Mahalanobis PROB Objectness + CountGD-Box loss)' if query_decoder else
        'Text-only YOLO-World counting adaptation v4 (nc=1 + Neighborhood Gaussian Relaxation + TopK Contrastive + Spatial Anchor Gating + Balanced Accum4 + FSC147-D)',
        epochs=30,nc_fixed=1,query_decoder=query_decoder,num_queries=2200,visual_self_exemplar_guidance=query_decoder,
        neg_weight=0.15,neg_topk=32,neighbor_sigma=0.0 if query_decoder else 0.004,
        neck_lr=2e-5,head_lr=1e-4,stride_radius=4.0,
        fsc147d_prompts=True,random_size_crop=False,balanced_accumulation4=True,
        weights_url='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt',
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        fsc_metadata_sha256={n:hashlib.sha256((data/n).read_bytes()).hexdigest()
            for n in ['Train_Test_Val_FSC_147.json','ImageClasses_FSC147.txt','annotation_FSC147_384.json']},
        frozen_image_and_text_backbones=True,train_neck_and_detection_heads=True,
        point_matching='one-to-one Hungarian, all decoded candidates',
        loss='5*(center_L1+exemplar_WH_L1)+2*exemplar_GIoU+2*focal',
        no_pseudo_boxes=True,no_density=True,no_dino=True,no_visual_exemplar_inputs=True,
        test_used=False,production_defaults_changed=False,
        stages='unit tests; gradient/freeze smoke; pretrained fullval; 30 epochs with fixed development selection; adapted fullval/pilot/absence')
    if resume_ref:
        previous=ROOT/'outputs/kaggle'/resume_ref.replace('/','__')/'version2/yoloworld_counting'
        history=json.loads((previous/'history.json').read_text())
        spec.update(resume_ref=resume_ref,resume_dataset=f'{user}/countyolo-yoloworld-resume-e10-20261119',resume_epoch=history[-1]['epoch'],
            resume_checkpoint_sha256=hashlib.sha256((previous/'last.pth').read_bytes()).hexdigest(),
            resume_best_sha256=hashlib.sha256((previous/'best.pth').read_bytes()).hexdigest(),
            stages='unit tests; same-shape eval-to-train gradient regression; restore epoch10 model/optimizer/scheduler; fullval epoch10; continue epochs11..30; adapted fullval/pilot/absence',
            runtime_fix='no_grad evaluation and clear decoder shape cache when returning to train; preserve metric definition')
    payload={name:base64.b64encode((ROOT/name).read_bytes()).decode() for name in FILES}
    reviewed=json.loads((ROOT/'outputs/semantic_training/pilot_v1/manifest.json').read_text())
    excluded=[r['image'] for r in reviewed['image_records'] if r['role']=='heldout']
    pilot=json.loads((ROOT/'outputs/fsc147_openworld/v3_pilot/experiment.json').read_text())['images']
    for name,value in [('exclude_train',excluded),('pilot58',pilot)]:
        payload[f'inputs/{name}.json']=base64.b64encode(json.dumps(value).encode()).decode()
    code=RUNNER.replace('__FILES__',repr(payload)).replace('__SPEC__',repr(spec))
    compile(code,'run.py','exec')
    if len(code.encode())>=1_000_000:raise RuntimeError('Payload too large')
    (folder/'run.py').write_text(code,encoding='utf-8',newline='\n')
    metadata=dict(id=f'{user}/{slug}',title=slug,code_file='run.py',language='python',kernel_type='script',
        is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=['xuncngng/fsc147-0']+([spec['resume_dataset']] if resume_ref else []),
        kernel_sources=[],competition_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return dict(ref=metadata['id'],folder=str(folder),state='prepared',spec=spec,
        code_sha256=hashlib.sha256(code.encode()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--user',default='thiend27');p.add_argument('--push',action='store_true')
    p.add_argument('--query-decoder',action='store_true',help='Launch Direction 1: YOLO-World + PROB Self-Attention Query Decoder')
    p.add_argument('--resume-from',help='Original kernel ref; uses verified downloaded version2 checkpoint')
    p.add_argument('--replace',action='store_true',help='Submit a corrected version, preserving the preceding submission record')
    a=p.parse_args();suffix=('_prob_detr' if a.query_decoder else '')+('_resume' if a.resume_from else '')
    ledger=ROOT/f'experiments/yoloworld_counting_kaggle_20261120{suffix}_{a.user}.json'
    previous=json.loads(ledger.read_text()) if ledger.exists() else None
    if previous and previous.get('state')=='submitted' and not a.replace:
        raise RuntimeError('Already submitted; inspect status before another run')
    job=prepare(a.user,a.resume_from,query_decoder=a.query_decoder);save=lambda:(ledger.write_text(json.dumps(job,indent=2),encoding='utf-8'))
    if previous and a.replace:
        job['superseded_submission']=previous
        job['correction']=('Replace failed notebook-output attachment with a private SHA-verified resume dataset; no training loss change' if a.resume_from else
            'Match original GT-count geometry normalization, exemplar target-center replacement/association, and AdamW optimizer; old results invalid for paper-aligned conclusions')
    save()
    if a.push:
        from tools.kaggle_watch import accounts,api_for
        api=api_for(a.user,accounts()[a.user]);response=api.kernels_push(job['folder'])
        if getattr(response,'error',None):raise RuntimeError(response.error)
        job.update(state='submitted',submitted_at=datetime.now(timezone.utc).isoformat(),version_number=response.version_number);save()
        job['remote_state']=str(api.kernels_status(job['ref']).status);save()
    print(json.dumps({k:v for k,v in job.items() if k!='spec'},indent=2),flush=True)


if __name__=='__main__':main()
