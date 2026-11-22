"""Package and submit OV-DQUO (EVA02-CLIP + CLIPSelf + 3-Scale Deformable Attention) experiments to Kaggle."""
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
    'models/ov_dquo_backbone.py',
    'models/ov_dquo_counting.py',
    'datasets/yoloworld_counting.py',
    'datasets/fsc147_dataset.py',
    'datasets/density_generator.py',
    'tools/train_yoloworld_counting.py',
    'tools/train_llmdet_counting.py',
    'tools/train_ov_dquo_counting.py',
    'tools/test_ov_dquo_counting.py',
]

GDRIVE_IDS = {
    'vitb16': '15bVUiHP14ChzLGzU3DA_4J7O1vaox51a',
    'vitl14': '1yQfWsoW9meZtKWhs2e9S9EPwi8lxqJei',
}

RUNNER = r'''
import base64,glob,hashlib,json,os,pathlib,subprocess,sys,time,traceback
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
    run([sys.executable,'-m','pip','install','-q','open_clip_torch','gdown','scipy','opencv-python'])
    import torch, gdown
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    roots=sorted({pathlib.Path(p).resolve().parent for p in glob.glob('/kaggle/input/**/Train_Test_Val_FSC_147.json',recursive=True)})
    if len(roots)!=1:raise RuntimeError('Attach exactly one FSC147 dataset')
    data=roots[0]

    # Download official OV-DQUO checkpoint (or reuse if mounted in /kaggle/input)
    ckpt_path=W/f"ovdquo_{SPEC['variant']}_lvis.pth"
    mounted=sorted(glob.glob('/kaggle/input/**/*OVDQUO*.pth',recursive=True)+glob.glob('/kaggle/input/**/*ovdquo*.pth',recursive=True))
    if mounted:
        ckpt_path=pathlib.Path(mounted[0])
        print('Using mounted OV-DQUO checkpoint:',ckpt_path,flush=True)
    elif not ckpt_path.exists():
        status('downloading_checkpoint')
        url=f"https://drive.google.com/uc?id={SPEC['gdrive_id']}"
        out=gdown.download(url,str(ckpt_path),quiet=False,fuzzy=True)
        if not out or not ckpt_path.exists() or ckpt_path.stat().st_size < 10_000_000:
            raise RuntimeError(f'Failed to download OV-DQUO checkpoint from {url}')
        print(f'Downloaded OV-DQUO checkpoint: {ckpt_path} ({ckpt_path.stat().st_size/1e6:.1f} MB)',flush=True)

    init_args=[]
    if SPEC.get('init_kernel') or SPEC.get('init_dataset'):
        ckpts=sorted(glob.glob('/kaggle/input/**/best.pth',recursive=True))
        if not ckpts:raise RuntimeError('Expected best.pth from attached init_kernel or init_dataset')
        init_args=['--init',ckpts[0]]
        print('Found init checkpoint:',ckpts[0],flush=True)

    status('tests');run([sys.executable,'-m','unittest','tools.test_ov_dquo_counting'])
    common=['--data-root',data,'--variant',SPEC['variant'],'--checkpoint',ckpt_path,
        '--resolution',str(SPEC['resolution']),
        '--exclude-train',W/'inputs/exclude_train.json','--pilot-images',W/'inputs/pilot58.json',
        '--max-classes',str(SPEC['max_classes']),'--neg-class-weight',str(SPEC['neg_class_weight']),
        '--absent-prob',str(SPEC['absent_prob']),'--prior-weight',str(SPEC['prior_weight']),
        '--tile-trigger',str(SPEC['tile_trigger']),*init_args]
    if SPEC.get('use_exemplars'):
        common.append('--use-exemplars')
    if SPEC.get('grad_checkpoint'):
        common.append('--grad-checkpoint')
    if SPEC['mode']=='eval':
        status('evaluating')
        run([sys.executable,'-u','tools/train_ov_dquo_counting.py',*common,
            '--output',W/'ov_dquo_eval','--eval-only'])
    else:
        lr_args=['--backbone-lr',str(SPEC['backbone_lr']),'--lr',str(SPEC['lr']),
                 '--adapter-lr',str(SPEC['adapter_lr']),'--cal-lr',str(SPEC['cal_lr'])]
        status('gradient_smoke')
        run([sys.executable,'-u','tools/train_ov_dquo_counting.py',*common,*lr_args,
            '--output',W/'gradient_smoke','--smoke'])
        status('training_and_validation')
        run([sys.executable,'-u','tools/train_ov_dquo_counting.py',*common,*lr_args,
            '--output',W/'ov_dquo_counting','--epochs',str(SPEC['epochs']),
            '--accumulate',str(SPEC['accumulate'])])
    status('complete')
except BaseException as exc:
    status('failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();raise
'''

MODE_META = {
    'eval': (
        'countyolo-ov-dquo-eval-v3-20261122',
        'OV-DQUO v3 Full-Val (raw + 2x2 scale-matched tiled), Pilot-58, and Absent evaluation',
        False,
    ),
    'text': (
        'countyolo-ov-dquo-text-v3-20261122',
        'OV-DQUO v3 Text-Only Counting (EVA02-CLIP-L/14 + Interleaved Mid/Post Fusion + Stride-8 Region-Text Residual + Absent Reg)',
        False,
    ),
    'multimodal': (
        'countyolo-ov-dquo-mm-v3-20261122',
        'OV-DQUO v3 Multimodal Counting (EVA02-CLIP-L/14 + Single-Matcher + Dual-Tower Stride-8 Text/Vis + Interleaved Fusion + 2x Tiling)',
        True,
    ),
}


def prepare(user='ttdan2006', mode='multimodal', variant='vitl14', resolution=672,
            epochs=10, backbone_lr=1e-5, lr=2.0e-5, adapter_lr=1.5e-4, cal_lr=1.5e-3,
            max_classes=8, neg_class_weight=0.5, absent_prob=0.05, prior_weight=0.10,
            tile_trigger=400, grad_checkpoint=True, init_kernel=None, init_dataset=None):
    slug, purpose, use_exemplars = MODE_META[mode]
    folder = ROOT / 'kaggle_video_run' / user / slug
    folder.mkdir(parents=True, exist_ok=True)
    spec = dict(
        mode=mode,
        purpose=purpose,
        variant=variant,
        gdrive_id=GDRIVE_IDS[variant],
        resolution=resolution,
        epochs=epochs,
        accumulate=4,
        backbone_lr=backbone_lr,
        lr=lr,
        adapter_lr=adapter_lr,
        cal_lr=cal_lr,
        max_classes=max_classes,
        neg_class_weight=neg_class_weight,
        absent_prob=absent_prob,
        prior_weight=prior_weight,
        use_exemplars=use_exemplars,
        grad_checkpoint=grad_checkpoint,
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
    p.add_argument('--user', default='ttdan2006')
    p.add_argument('--mode', choices=sorted(MODE_META), default='multimodal')
    p.add_argument('--variant', choices=sorted(GDRIVE_IDS), default='vitl14')
    p.add_argument('--resolution', type=int, default=672)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--lr', type=float, default=2.0e-5)
    p.add_argument('--adapter-lr', type=float, default=1.5e-4)
    p.add_argument('--cal-lr', type=float, default=1.5e-3)
    p.add_argument('--max-classes', type=int, default=8)
    p.add_argument('--neg-class-weight', type=float, default=0.5)
    p.add_argument('--absent-prob', type=float, default=0.05)
    p.add_argument('--prior-weight', type=float, default=0.10)
    p.add_argument('--tile-trigger', type=int, default=400)
    p.add_argument('--no-grad-checkpoint', action='store_true')
    p.add_argument('--init-kernel', default=None)
    p.add_argument('--init-dataset', default=None)
    p.add_argument('--push', action='store_true')
    a = p.parse_args()
    job = prepare(
        user=a.user, mode=a.mode, variant=a.variant, resolution=a.resolution,
        epochs=a.epochs, backbone_lr=a.backbone_lr, lr=a.lr,
        adapter_lr=a.adapter_lr, cal_lr=a.cal_lr,
        max_classes=a.max_classes, neg_class_weight=a.neg_class_weight,
        absent_prob=a.absent_prob, prior_weight=a.prior_weight,
        tile_trigger=a.tile_trigger, grad_checkpoint=not a.no_grad_checkpoint,
        init_kernel=a.init_kernel, init_dataset=a.init_dataset,
    )
    ledger = ROOT / f'experiments/ov_dquo_{a.mode}_v3_kaggle_20261122_{a.user}.json'
    ledger.write_text(json.dumps(job, indent=2), encoding='utf-8')
    if a.push:
        ver, remote_state = push_with_retry(a.user, job)
        job.update(state='submitted', submitted_at=datetime.now(timezone.utc).isoformat(),
                   version_number=ver, remote_state=remote_state)
        ledger.write_text(json.dumps(job, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in job.items() if k != 'spec'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
