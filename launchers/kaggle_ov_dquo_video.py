"""Package and submit OV-DQUO (v1 / v2 / v3.1) + SAM 2.1 Large (FP16) + Tracking + TemporalIdentity VideoCount eval on Kaggle."""
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
    'models/ov_dquo_backbone.py',
    'models/ov_dquo_counting.py',
    'models/ov_dquo_v1_infer.py',
    'models/ov_dquo_video_detector.py',
    'video/count_sam2.py',
    'video/identity.py',
    'video/eval_common.py',
    'video/ov_dquo_sam2_pipeline.py',
    'tools/eval_ov_dquo_video.py',
    'tools/test_ov_dquo_video.py',
    'checkpoints/huggingface.json',
]

GDRIVE_IDS = {
    'vitb16': ('ovdquo_vitb16_lvis.pth', '15bVUiHP14ChzLGzU3DA_4J7O1vaox51a'),
    'vitl14': ('ovdquo_vitl14_lvis.pth', '1yQfWsoW9meZtKWhs2e9S9EPwi8lxqJei'),
}

RUNNER = r'''
import base64,glob,hashlib,json,os,pathlib,shutil,subprocess,sys,traceback,zipfile
W=pathlib.Path('/kaggle/working');os.chdir(W);sys.path.insert(0,str(W))
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
FILES=__FILES__
SPEC=__SPEC__
for name,payload in FILES.items():
    p=W/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(payload))
for folder in ['models','video','tools']:(W/folder/'__init__.py').touch()
def status(state,**extra):
    p=W/'job_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state,spec=SPEC,**extra),indent=2));os.replace(tmp,p)
def run(args,env=None):
    print('$',' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),check=True,env=env)
try:
    status('setup');run(['nvidia-smi'])
    run([sys.executable,'-m','pip','install','-q','open_clip_torch','gdown','scipy','opencv-python','huggingface_hub','hydra-core','iopath'])
    run([sys.executable,'-m','pip','install','-q','--no-build-isolation',
         'git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'],
        env=dict(os.environ,SAM2_BUILD_CUDA='0'))
    import torch, gdown
    if not torch.cuda.is_available():raise RuntimeError('GPU required')

    # 1. Locate mounted base OV-DQUO checkpoint (or download from Google Drive if not in kernel_sources)
    base_name=SPEC['base_filename']
    base_ckpts=sorted(glob.glob(f'/kaggle/input/**/{base_name}',recursive=True))
    if base_ckpts:
        base_ckpt=pathlib.Path(base_ckpts[0])
    else:
        base_ckpt=W/base_name
        url=f"https://drive.google.com/uc?id={SPEC['gdrive_id']}"
        out=gdown.download(url,str(base_ckpt),quiet=False,fuzzy=True)
        if not out or not base_ckpt.exists() or base_ckpt.stat().st_size < 10_000_000:
            raise RuntimeError(f'Failed downloading base OV-DQUO ckpt from {url}')

    # 2. Locate mounted fine-tuned OV-DQUO checkpoint (kernel output `best.pth` or private dataset `.pth`)
    tuned_ckpts=sorted(glob.glob('/kaggle/input/**/ov_dquo_counting/best.pth',recursive=True))
    if not tuned_ckpts:
        tuned_ckpts=sorted(glob.glob('/kaggle/input/**/ov_dquo_*.pth',recursive=True))
    if not tuned_ckpts:
        tuned_ckpts=sorted(glob.glob('/kaggle/input/**/best.pth',recursive=True))
    if not tuned_ckpts:
        raise RuntimeError(f'Missing mounted tuned OV-DQUO checkpoint in /kaggle/input')
    tuned_ckpt=pathlib.Path(tuned_ckpts[0])
    print('Base OV-DQUO ckpt:',base_ckpt,'Tuned ckpt:',tuned_ckpt,flush=True)

    # 3. Download pinned SAM 2.1 Large checkpoint with SHA-256 verification
    from huggingface_hub import hf_hub_download
    manifest=json.loads((W/'checkpoints/huggingface.json').read_text())
    entry=next(f for f in manifest['files'] if f['path']=='checkpoints/sam2.1_hiera_large.pt')
    sam_src=hf_hub_download(repo_id=manifest['repo_id'],revision=manifest['revision'],filename=entry.get('filename','sam2.1_hiera_large.pt'))
    sam_ckpt=W/'checkpoints/sam2.1_hiera_large.pt';sam_ckpt.parent.mkdir(exist_ok=True)
    shutil.copyfile(sam_src,sam_ckpt)
    sam_sha=hashlib.file_digest(sam_ckpt.open('rb'),'sha256').hexdigest()
    if sam_sha!=entry['sha256']:raise ValueError(f'SAM2 SHA256 mismatch: {sam_sha}')
    print('SAM 2.1 Large verified:',sam_sha[:16],flush=True)

    # 4. Locate pre-sampled VideoCount benchmark
    markers=list(pathlib.Path('/kaggle/input').rglob('countyolo_sampling.json'))
    if len(markers)==1:
        data_root=markers[0].parent
    else:
        zips=list(pathlib.Path('/kaggle/input').rglob('videocount-benchmark.zip'))
        if len(zips)!=1:raise RuntimeError(f'Expected 1 VideoCount cache, found markers={markers} zips={zips}')
        with zipfile.ZipFile(zips[0]) as zf:zf.extractall(W/'data')
        data_root=W/'data/VideoCount'
    print('VideoCount data_root:',data_root,flush=True)

    # 5. Run pipeline unit tests before inference
    status('tests')
    run([sys.executable,'-m','unittest','-v','tools.test_ov_dquo_video'])

    # 6. Evaluate complete 4-stage OV-DQUO + SAM2 (FP16) + Tracking + TemporalIdentity on VideoCount
    status('evaluating')
    cmd=[
        sys.executable,'-u','tools/eval_ov_dquo_video.py',
        '--data-root',str(data_root),
        '--datasets',*SPEC['datasets'],
        '--base-checkpoint',str(base_ckpt),
        '--tuned-checkpoint',str(tuned_ckpt),
        '--sam-checkpoint',str(sam_ckpt),
        '--output-dir',str(W/'ov_dquo_video'),
        '--threshold',str(SPEC['threshold']),
        '--tile-trigger',str(SPEC['tile_trigger']),
        '--window',str(SPEC['window']),
        '--batch-size',str(SPEC['batch_size']),
        '--box-chunk',str(SPEC['box_chunk']),
        '--save-previews',
    ]
    run(cmd)
    status('complete')
except BaseException as exc:
    status('failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();raise
finally:
    for cleanup in [W/'checkpoints/sam2.1_hiera_large.pt', W/SPEC['base_filename']]:
        cleanup.unlink(missing_ok=True)
'''


def prepare(user='nadthien', version='v2', variant='vitl14',
            datasets=('penguins', 'crystals', 'mot20'),
            threshold=0.26, tile_trigger=400, window=3, batch_size=30, box_chunk=64,
            ckpt_kernel=None, ckpt_dataset='nadthien/ov-dquo-v2-ckpt', video_dataset=None):
    slug = f'countyolo-ov-dquo-{version}-sam2-video-20261123'
    folder = ROOT / 'kaggle_video_run' / user / slug
    folder.mkdir(parents=True, exist_ok=True)
    ds_ref = video_dataset or f'{user}/videocount-benchmark'
    base_filename, gdrive_id = GDRIVE_IDS[variant]
    spec = dict(
        user=user,
        version=version,
        variant=variant,
        base_filename=base_filename,
        gdrive_id=gdrive_id,
        purpose=f'OV-DQUO {version} ({variant}) + SAM 2.1 Large (FP16) + Tracking + TemporalIdentity VideoCount benchmark',
        datasets=list(datasets),
        threshold=float(threshold),
        tile_trigger=int(tile_trigger),
        window=int(window),
        batch_size=int(batch_size),
        box_chunk=int(box_chunk),
        ckpt_kernel=ckpt_kernel,
        ckpt_dataset=ckpt_dataset,
        video_dataset=ds_ref,
    )
    payload = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in FILES}
    code = RUNNER.replace('__FILES__', repr(payload)).replace('__SPEC__', repr(spec))
    compile(code, 'run.py', 'exec')
    if len(code.encode('utf-8')) >= 1_000_000:
        raise ValueError(f'Kaggle script exceeds 1 MB: {len(code.encode("utf-8"))} bytes')
    (folder / 'run.py').write_text(code, encoding='utf-8', newline='\n')
    dataset_sources = [ds_ref]
    if ckpt_dataset:
        dataset_sources.append(ckpt_dataset)
    metadata = dict(
        id=f'{user}/{slug}',
        title=slug,
        code_file='run.py',
        language='python',
        kernel_type='script',
        is_private=True,
        enable_gpu=True,
        enable_internet=True,
        dataset_sources=dataset_sources,
        kernel_sources=[ckpt_kernel] if ckpt_kernel else [],
        competition_sources=[],
    )
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return dict(
        ref=metadata['id'],
        folder=str(folder),
        state='prepared',
        spec=spec,
        code_sha256=hashlib.sha256(code.encode()).hexdigest(),
    )


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
    p.add_argument('--version', choices=['v1', 'v2', 'v3'], default='v2')
    p.add_argument('--variant', choices=sorted(GDRIVE_IDS), default='vitl14')
    p.add_argument('--datasets', nargs='+', choices=['penguins', 'crystals', 'mot20'],
                   default=['penguins', 'crystals', 'mot20'])
    p.add_argument('--threshold', type=float, default=0.26)
    p.add_argument('--tile-trigger', type=int, default=400)
    p.add_argument('--window', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=30)
    p.add_argument('--box-chunk', type=int, default=64)
    p.add_argument('--ckpt-kernel', default=None)
    p.add_argument('--ckpt-dataset', default='nadthien/ov-dquo-v2-ckpt')
    p.add_argument('--push', action='store_true')
    a = p.parse_args()
    job = prepare(
        user=a.user, version=a.version, variant=a.variant, datasets=tuple(a.datasets),
        threshold=a.threshold, tile_trigger=a.tile_trigger,
        window=a.window, batch_size=a.batch_size, box_chunk=a.box_chunk,
        ckpt_kernel=a.ckpt_kernel, ckpt_dataset=a.ckpt_dataset,
    )
    ledger = ROOT / f'experiments/ov_dquo_{a.version}_sam2_video_kaggle_20261123_{a.user}.json'
    ledger.write_text(json.dumps(job, indent=2), encoding='utf-8')
    if a.push:
        ver, remote_state = push_with_retry(a.user, job)
        job.update(
            state='submitted',
            submitted_at=datetime.now(timezone.utc).isoformat(),
            version_number=ver,
            remote_state=remote_state,
        )
        ledger.write_text(json.dumps(job, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in job.items() if k != 'spec'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
