"""Package and submit OmDet-Turbo v2 counting experiments (eval / text / multimodal) to Kaggle."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FILES = [
    'models/yoloworld_counting.py',
    'models/llmdet_counting.py',
    'models/omdet_turbo_counting.py',
    'datasets/yoloworld_counting.py',
    'datasets/fsc147_dataset.py',
    'datasets/density_generator.py',
    'tools/train_yoloworld_counting.py',
    'tools/train_llmdet_counting.py',
    'tools/train_omdet_turbo_counting.py',
    'tools/test_omdet_turbo_counting.py',
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
    run([sys.executable,'-m','pip','install','-q','transformers>=4.49.0','timm','accelerate','scipy','opencv-python'])
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    roots=sorted({pathlib.Path(p).resolve().parent for p in glob.glob('/kaggle/input/**/Train_Test_Val_FSC_147.json',recursive=True)})
    if len(roots)!=1:raise RuntimeError('Attach exactly one FSC147 dataset')
    data=roots[0]
    init_args=[]
    if SPEC.get('init_kernel') or SPEC.get('init_dataset'):
        ckpts=sorted(glob.glob('/kaggle/input/**/best.pth',recursive=True))
        if not ckpts:raise RuntimeError('Expected best.pth from attached init_kernel or init_dataset')
        init_args=['--init',ckpts[0]]
        print('Found init checkpoint:',ckpts[0],flush=True)
    status('tests');run([sys.executable,'-m','unittest','tools.test_omdet_turbo_counting'])
    common=['--data-root',data,'--model-id',SPEC['model_id'],
        '--exclude-train',W/'inputs/exclude_train.json','--pilot-images',W/'inputs/pilot58.json',
        '--max-classes',str(SPEC['max_classes']),'--neg-class-weight',str(SPEC['neg_class_weight']),
        '--tile-trigger',str(SPEC['tile_trigger']),*init_args]
    if SPEC.get('use_exemplars'):
        common.append('--use-exemplars')
    if SPEC['mode']=='eval':
        status('evaluating')
        run([sys.executable,'-u','tools/train_omdet_turbo_counting.py',*common,
            '--output',W/'omdet_eval','--eval-only'])
    else:
        status('gradient_smoke')
        run([sys.executable,'-u','tools/train_omdet_turbo_counting.py',*common,
            '--lr',str(SPEC['lr']),'--exemplar-lr',str(SPEC['exemplar_lr']),
            '--output',W/'gradient_smoke','--smoke'])
        status('training_and_validation')
        run([sys.executable,'-u','tools/train_omdet_turbo_counting.py',*common,
            '--lr',str(SPEC['lr']),'--exemplar-lr',str(SPEC['exemplar_lr']),
            '--output',W/'omdet_counting','--epochs',str(SPEC['epochs']),
            '--accumulate',str(SPEC['accumulate'])])
    status('complete')
except BaseException as exc:
    status('failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();raise
'''

MODE_META = {
    'eval': (
        'countyolo-omdet-turbo-eval-20261121',
        'OmDet-Turbo v2 Full-Val (raw + 2x2 tiled), Pilot-58, and Absent evaluation',
        False,
    ),
    'text': (
        'countyolo-omdet-turbo-text-v2-20261121',
        'OmDet-Turbo v2 Text-Only (Frozen CLIP class heads + RT-DETR HybridEncoder + balanced 8-class negatives)',
        False,
    ),
    'multimodal': (
        'countyolo-omdet-turbo-mm-v2-20261121',
        'OmDet-Turbo v2 Class-Agnostic Multimodal (Frozen CLIP heads + Spatial Sim-Map + Direct Visual Cosine Head + 33/33/33 Modality Dropout)',
        True,
    ),
}


def prepare(user='nadthien', mode='multimodal', model_id='omlab/omdet-turbo-swin-tiny-hf',
            epochs=20, lr=3e-5, exemplar_lr=1.5e-4, max_classes=8, neg_class_weight=0.5,
            tile_trigger=600, init_kernel=None, init_dataset=None):
    slug, purpose, use_exemplars = MODE_META[mode]
    folder = ROOT / 'kaggle_video_run' / user / slug
    folder.mkdir(parents=True, exist_ok=True)
    spec = dict(
        mode=mode,
        purpose=purpose,
        model_id=model_id,
        epochs=epochs,
        accumulate=4,
        lr=lr,
        exemplar_lr=exemplar_lr,
        max_classes=max_classes,
        neg_class_weight=neg_class_weight,
        use_exemplars=use_exemplars,
        freeze_class_heads=True,
        tile_trigger=tile_trigger,
        init_kernel=init_kernel,
        init_dataset=init_dataset,
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
    ds_sources = ['xuncngng/fsc147-0']
    if init_dataset:
        ds_sources.append(init_dataset)
    metadata = dict(
        id=f'{user}/{slug}', title=slug, code_file='run.py', language='python', kernel_type='script',
        is_private=True, enable_gpu=True, enable_internet=True, dataset_sources=ds_sources,
        kernel_sources=[init_kernel] if init_kernel else [], competition_sources=[])
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return dict(ref=metadata['id'], folder=str(folder), state='prepared', spec=spec,
                code_sha256=hashlib.sha256(code.encode()).hexdigest())


def push_with_retry(user, job, attempts=5):
    from tools.kaggle_watch import accounts, api_for
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            api = api_for(user, accounts()[user])
            resp = api.kernels_push(job['folder'])
            if getattr(resp, 'error', None):
                raise RuntimeError(resp.error)
            remote_state = str(api.kernels_status(job['ref']).status)
            return resp.version_number, remote_state
        except Exception as exc:
            last_err = exc
            if attempt < attempts:
                time.sleep(3 * attempt)
    raise last_err


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--user', default='nadthien')
    p.add_argument('--mode', choices=sorted(MODE_META), default='multimodal')
    p.add_argument('--model-id', default='omlab/omdet-turbo-swin-tiny-hf')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--exemplar-lr', type=float, default=1.5e-4)
    p.add_argument('--max-classes', type=int, default=8)
    p.add_argument('--neg-class-weight', type=float, default=0.5)
    p.add_argument('--tile-trigger', type=int, default=600)
    p.add_argument('--init-kernel', default=None)
    p.add_argument('--init-dataset', default=None)
    p.add_argument('--push', action='store_true')
    a = p.parse_args()
    job = prepare(
        user=a.user, mode=a.mode, model_id=a.model_id,
        epochs=a.epochs, lr=a.lr, exemplar_lr=a.exemplar_lr,
        max_classes=a.max_classes, neg_class_weight=a.neg_class_weight,
        tile_trigger=a.tile_trigger, init_kernel=a.init_kernel, init_dataset=a.init_dataset,
    )
    ledger = ROOT / f'experiments/omdet_turbo_{a.mode}_v2_kaggle_20261121_{a.user}.json'
    ledger.write_text(json.dumps(job, indent=2), encoding='utf-8')
    if a.push:
        ver, remote_state = push_with_retry(a.user, job)
        job.update(state='submitted', submitted_at=datetime.now(timezone.utc).isoformat(),
                   version_number=ver, remote_state=remote_state)
        ledger.write_text(json.dumps(job, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in job.items() if k != 'spec'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
