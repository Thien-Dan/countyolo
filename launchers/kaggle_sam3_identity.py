"""One private full-VideoCount SAM3/identity paired evaluation, no retraining."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

FILES=['models/density_yolo.py','engines/sam3_counter.py','video/sam3_policy.py',
       'video/identity.py','video/identity_features.py','video/eval_common.py',
       'tools/eval_video_sam3.py','tools/test_temporal_identity.py']

RUNNER=r'''
import base64,glob,hashlib,importlib.metadata,json,os,pathlib,shutil,subprocess,sys,traceback
W=pathlib.Path('/kaggle/working');os.chdir(W);sys.path.insert(0,str(W))
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
FILES=__FILES__
SPEC=__SPEC__
for name,encoded in FILES.items():
    path=W/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(encoded))
for name in ['models','engines','video','tools']:(W/name/'__init__.py').touch()
def status(state,**kw):
    target=W/'job_status.json';tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state,spec=SPEC,**kw),indent=2));os.replace(tmp,target)
def run(args,check=True):
    print('$',' '.join(map(str,args)),flush=True)
    return subprocess.run(list(map(str,args)),check=check).returncode
def discover(name):return sorted({pathlib.Path(p).resolve() for p in glob.glob('/kaggle/input/**/'+name,recursive=True)})
def digest(path):
    with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
try:
    status('setup');run(['nvidia-smi'])
    run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.204','transformers==4.57.6',
         'numpy<2.3','scipy','opencv-python'])
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    markers=discover('countyolo_sampling.json')
    if len(markers)!=1:raise RuntimeError('Expected exactly one sampled VideoCount cache')
    data=markers[0].parent
    caches=discover('dinov3_adapt_cache.json');weights=discover('sam3.pt')
    if len(caches)!=1 or len(weights)!=1:raise RuntimeError('Ambiguous model input datasets')
    source=caches[0].parent;cache=json.loads(caches[0].read_text())
    if cache['dinov3_revision']!=SPEC['dinov3_revision']:raise RuntimeError('Wrong DINOv3 revision')
    target=W/'dinov3_local';target.mkdir(exist_ok=True)
    for name in ['config.json','model.safetensors']:
        filename='dinov3_'+name
        if digest(source/filename)!=SPEC['dinov3_files'][filename]:raise RuntimeError('DINOv3 checksum mismatch')
        shutil.copyfile(source/filename,target/name)
    sam3_hash=digest(weights[0])
    if sam3_hash!=SPEC['sam3_sha256']:raise RuntimeError('SAM3 checksum mismatch')
    # Resolve the authorized private input locally; never embed a Hugging Face token.
    bootstrap="""import sys
from transformers import AutoModel
original=AutoModel.from_pretrained
def local(name,*args,**kwargs):
    if name=='facebook/dinov3-vits16-pretrain-lvd1689m':
        name=__MODEL__;kwargs.pop('revision',None);kwargs['local_files_only']=True
    return original(name,*args,**kwargs)
AutoModel.from_pretrained=staticmethod(local)
from tools.eval_video_sam3 import main
main()
""".replace('__MODEL__',repr(str(target)))
    (W/'identity_bootstrap.py').write_text(bootstrap)
    run([sys.executable,'-m','unittest','tools.test_temporal_identity'])
    from video.eval_common import cases,atomic_json
    inventories={dataset:cases(data,dataset) for dataset in SPEC['datasets']}
    # Check mounted files against the locally reviewed, exact sampling manifest.
    for dataset,items in inventories.items():
        observed=[dict(video=c['video'],text=c['text'],frame_manifest=c['frame_manifest']) for c in items]
        if observed!=SPEC['cases'][dataset]:raise RuntimeError('Video/prompt/frame selection differs: '+dataset)
    manifest=dict(spec=SPEC,sam3_sha256=sam3_hash,dinov3_cache=cache,
        data_sampling_sha256=digest(markers[0]),gpu=torch.cuda.get_device_name(),
        packages={n:importlib.metadata.version(n) for n in ['torch','torchvision','transformers','ultralytics','numpy']},
        source_sha256={n:hashlib.sha256(base64.b64decode(v)).hexdigest() for n,v in FILES.items()})
    atomic_json(W/'run_manifest.json',manifest)
    failures=[]
    for dataset in SPEC['datasets']:
        status('running',dataset=dataset,completed_datasets=SPEC['datasets'][:SPEC['datasets'].index(dataset)])
        code=run([sys.executable,'-u',W/'identity_bootstrap.py','--data-root',data,'--dataset',dataset,
                  '--weights',weights[0],'--results',W/'results.json','--name','sam3_identity',
                  '--conf','0.5','--detection-conf','0.5','--tracker-profile','aligned',
                  '--object-batch-size','8','--precision','fp16','--identity-mode','dinov3',
                  '--identity-device','cpu'],check=False)
        if code:failures.append(dict(dataset=dataset,exit_code=code))
        # Preserve partial paired results even if a later case fails or times out.
        results=json.loads((W/'results.json').read_text()) if (W/'results.json').exists() else {}
        paired={}
        for key,result in results.items():
            rows=[r for r in result['rows'] if r['status']=='complete']; n=len(rows)
            paired[key]=dict(n=n,expected=result['expected'],complete=result['complete'],
                baseline_mae=sum(abs(r['baseline_pred']-r['gt']) for r in rows)/n if n else None,
                identity_mae=sum(abs(r['pred']-r['gt']) for r in rows)/n if n else None,
                accepted_links=sum(len(r['identity']['links']) for r in rows),
                uncertain_links=sum(len(r['identity']['uncertain']) for r in rows))
        atomic_json(W/'paired_summary.json',dict(results=paired,promoted=False,
            note='Same SAM3 pass and emitted tracks. GT counts used for reporting only. No identity GT or threshold tuning.'))
    if failures:raise RuntimeError('Some evaluations failed; inspect results.json and per-case errors: '+str(failures))
    status('complete')
except BaseException as exc:
    status('failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();raise
'''


def prepare(user='nadthien',out_dir=None):
    from video.eval_common import cases
    cache=json.loads((ROOT/'kaggle_adapt_cache/nadthien/dinov3_adapt_cache.json').read_text())
    with (ROOT/'sam3.pt').open('rb') as stream:sam3_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    spec=dict(datasets=['penguins','crystals','mot20'],tracker_profile='aligned',object_batch_size=8,
              identity_mode='dinov3',identity_device='cpu',precision='fp16',
              sam3_sha256=sam3_hash,dinov3_revision=cache['dinov3_revision'],
              dinov3_files={n:cache['files'][n] for n in ['dinov3_config.json','dinov3_model.safetensors']},
              purpose='Paired full-video SAM3 native ID vs temporal counting identity; no retraining or tuning',
              cases={d:[dict(video=c['video'],text=c['text'],frame_manifest=c['frame_manifest'])
                        for c in cases(ROOT/'data/VideoCount',d)] for d in ['penguins','crystals','mot20']})
    payload={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in FILES}
    script=RUNNER.replace('__FILES__',repr(payload)).replace('__SPEC__',repr(spec));compile(script,'run.py','exec')
    if len(script.encode())>=1_000_000:raise ValueError('Kaggle script too large')
    slug='countyolo-sam3-identity-video-20261118'
    folder=Path(out_dir) if out_dir else ROOT/'kaggle_video_run'/user/slug;folder.mkdir(parents=True,exist_ok=True)
    (folder/'run.py').write_text(script,encoding='utf-8',newline='\n')
    metadata=dict(id=f'{user}/{slug}',title=slug,code_file='run.py',
                  language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=True,
                  dataset_sources=['nadthien/videocount-benchmark','nadthien/sam3-weights','nadthien/countyolo-dinov3-adapt-cache'],
                  competition_sources=[],kernel_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return dict(ref=metadata['id'],folder=str(folder.resolve()),spec=spec,state='prepared',
                code_sha256=hashlib.sha256((folder/'run.py').read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--push',action='store_true')
    p.add_argument('--username',default='nadthien')
    p.add_argument('--retry-error',action='store_true',help='Retry only after verifying remote ERROR')
    a=p.parse_args()
    suffix='' if a.username=='nadthien' else '_'+a.username
    ledger=ROOT/'experiments'/('sam3_identity_kaggle_20261118'+suffix+'.json');history=[]
    if ledger.exists():
        old=json.loads(ledger.read_text())
        history=old.get('attempt_history',[])
        if old.get('state') not in ('prepared','failed_before_submission'):
            if not a.retry_error:raise RuntimeError('Existing submitted job; check its status instead of overwriting')
            from tools.kaggle_watch import accounts,api_for
            api=api_for(a.username,accounts()[a.username])
            remote=str(api.kernels_status(old['ref']).status).split('.')[-1].lower()
            if remote!='error':raise RuntimeError('Retry requires remote ERROR, got '+remote)
            history.append(dict(ref=old['ref'],code_sha256=old['code_sha256'],
                submitted_at=old.get('submitted_at'),remote_state=remote,
                reason='Runner import path fix: add /kaggle/working to sys.path'))
    job=prepare(a.username);job['attempt_history']=history
    ledger.write_text(json.dumps(job,indent=2),encoding='utf-8')
    if a.push:
        from tools.kaggle_watch import accounts,api_for
        api=api_for(a.username,accounts()[a.username]);response=api.kernels_push(job['folder'])
        if getattr(response,'error',None):raise RuntimeError(response.error)
        job.update(state='submitted',submitted_at=datetime.now(timezone.utc).isoformat())
        ledger.write_text(json.dumps(job,indent=2),encoding='utf-8')
        job['remote_state']=str(api.kernels_status(job['ref']).status)
        ledger.write_text(json.dumps(job,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in job.items() if k!='spec'},indent=2),flush=True)


if __name__=='__main__':main()
