"""Package and submit the LLMDet (7-layer CountGD-Box deep supervision) experiment to Kaggle."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FILES = [
    'models/yoloworld_counting.py',
    'models/llmdet_counting.py',
    'datasets/yoloworld_counting.py',
    'datasets/fsc147_dataset.py',
    'datasets/density_generator.py',
    'tools/train_yoloworld_counting.py',
    'tools/train_llmdet_counting.py',
    'tools/test_llmdet_counting.py',
]
RUNNER = r'''
import base64,glob,hashlib,importlib.metadata,json,os,pathlib,subprocess,sys,time,traceback
W=pathlib.Path('/kaggle/working');os.chdir(W);sys.path.insert(0,str(W))
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
FILES=__FILES__
SPEC=__SPEC__
for name,payload in FILES.items():
    path=W/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(payload))
for folder in ['models','datasets','tools']:(W/folder/'__init__.py').touch()
def status(state,**extra):
    path=W/'job_status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state,spec=SPEC,**extra),indent=2));os.replace(tmp,path)
def run(args):
    print('$',' '.join(map(str,args)),flush=True);subprocess.run(list(map(str,args)),check=True)
try:
    status('setup');run(['nvidia-smi'])
    run([sys.executable,'-m','pip','install','-q','transformers>=4.49.0','accelerate','scipy','opencv-python'])
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    roots=sorted({pathlib.Path(p).resolve().parent for p in glob.glob('/kaggle/input/**/Train_Test_Val_FSC_147.json',recursive=True)})
    if len(roots)!=1:raise RuntimeError('Attach exactly one FSC147 dataset')
    data=roots[0]
    status('tests');run([sys.executable,'-m','unittest','tools.test_llmdet_counting'])
    common=['--data-root',data,'--model-id',SPEC['model_id'],
        '--exclude-train',W/'inputs/exclude_train.json','--pilot-images',W/'inputs/pilot58.json']
    status('gradient_smoke')
    run([sys.executable,'-u','tools/train_llmdet_counting.py',*common,'--output',W/'gradient_smoke','--smoke'])
    status('training_and_validation')
    run([sys.executable,'-u','tools/train_llmdet_counting.py',*common,
        '--output',W/'llmdet_counting','--epochs',SPEC['epochs'],'--accumulate','4'])
    status('complete')
except BaseException as exc:
    status('failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();raise
'''


def prepare(user='thiend27', model_id='iSEE-Laboratory/llmdet_tiny'):
    slug = 'countyolo-llmdet-countgd-box-20261120'
    folder = ROOT / 'kaggle_video_run' / user / slug
    folder.mkdir(parents=True, exist_ok=True)
    spec = dict(
        purpose='LLMDet (MM-Grounding-DINO) adaptation with 7-layer CountGD-Box deep supervision (1 Encoder + 6 Deformable Decoder stages)',
        model_id=model_id,
        epochs=30,
        accumulate=4,
        lr=1e-4,
        fsc147d_prompts=True,
        nms_free=True,
    )
    payload = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in FILES}
    reviewed = json.loads((ROOT / 'outputs/semantic_training/pilot_v1/manifest.json').read_text())
    excluded = [r['image'] for r in reviewed['image_records'] if r['role'] == 'heldout']
    pilot = json.loads((ROOT / 'outputs/fsc147_openworld/v3_pilot/experiment.json').read_text())['images']
    for name, value in [('exclude_train', excluded), ('pilot58', pilot)]:
        payload[f'inputs/{name}.json'] = base64.b64encode(json.dumps(value).encode()).decode()
    code = RUNNER.replace('__FILES__', repr(payload)).replace('__SPEC__', repr(spec))
    compile(code, 'run.py', 'exec')
    (folder / 'run.py').write_text(code, encoding='utf-8', newline='\n')
    metadata = dict(
        id=f'{user}/{slug}', title=slug, code_file='run.py', language='python', kernel_type='script',
        is_private=True, enable_gpu=True, enable_internet=True, dataset_sources=['xuncngng/fsc147-0'],
        kernel_sources=[], competition_sources=[])
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return dict(ref=metadata['id'], folder=str(folder), state='prepared', spec=spec,
                code_sha256=hashlib.sha256(code.encode()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--user', default='thiend27')
    p.add_argument('--model-id', default='iSEE-Laboratory/llmdet_tiny')
    p.add_argument('--push', action='store_true')
    a = p.parse_args()
    job = prepare(a.user, a.model_id)
    ledger = ROOT / f'experiments/llmdet_counting_kaggle_20261120_{a.user}.json'
    ledger.write_text(json.dumps(job, indent=2), encoding='utf-8')
    if a.push:
        from tools.kaggle_watch import accounts, api_for
        api = api_for(a.user, accounts()[a.user])
        resp = api.kernels_push(job['folder'])
        if getattr(resp, 'error', None):
            raise RuntimeError(resp.error)
        job.update(state='submitted', submitted_at=datetime.now(timezone.utc).isoformat(),
                   version_number=resp.version_number, remote_state=str(api.kernels_status(job['ref']).status))
        ledger.write_text(json.dumps(job, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in job.items() if k != 'spec'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
