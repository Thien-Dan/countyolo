"""Build isolated Kaggle video evaluations. Heavy work runs only in Kaggle."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
COUNTGD_BOX_SHA256 = 'dc550ecb35dec9bb6608c53523a250eac84a6208bce00ab444cadb4a0903cfcd'
SOURCES = [
    'countvid_ext/tiled_countgd.py', 'countvid_ext/patch_countvid.py', 'countvid_ext/run_videocount.py',
    'countvid_ext/countyolo_stage1.py', 'countvid_ext/proposals.py', 'models/density_yolo.py',
    'utils/scale_zoom.py', 'engines/sam3_counter.py', 'video/sam3_policy.py', 'video/eval_common.py', 'tools/eval_video_sam3.py',
    'tools/test_video_pipeline.py', 'tools/sam3_image_probe.py', 'tools/video_tasks.py', 'tools/kaggle_watch.py', 'tools/compare_video_runs.py',
    'launchers/kaggle_video_launch.py',
]
CONFIGS = {
    'baseline': '',
    'baseline_batch30': '--obj_batch_size 30',
    'countyolo_fallback': '--stage1_countyolo @CY@ --countyolo_ckpt @CK@ --countyolo_sam3 @S3@ --countyolo_kf_every 30 --countyolo_absence_policy fallback',
    'countyolo_old': '--stage1_countyolo @CY@ --countyolo_ckpt @CK@ --countyolo_sam3 @S3@ --countyolo_kf_every 30',
    'countyolo_exnoise': '--stage1_countyolo @CY@ --countyolo_ckpt @CK@ --countyolo_sam3 @S3@ --countyolo_kf_every 30',
    'countyolo_fusion': '--stage1_countyolo @CY@ --countyolo_ckpt @CK@ --countyolo_sam3 @S3@ --countyolo_kf_every 5 --countyolo_fusion --countyolo_min_mass 0.2',
    # Ablation distinguishes proposal fusion from a more frequent SAM 3 teacher.
    'countyolo_kf5': '--stage1_countyolo @CY@ --countyolo_ckpt @CK@ --countyolo_sam3 @S3@ --countyolo_kf_every 5 --countyolo_min_mass 0.2',
    'sam3': None,
    'sam3_native': None,
    'sam3_aligned': None,
    'sam3_aligned_batch8': None,
    'sam3_aligned_fp32': None,
}

RUNNER = r'''
import base64, glob, hashlib, importlib.metadata, json, os, pathlib, shutil, subprocess, sys, time, traceback
W = pathlib.Path('/kaggle/working')
os.chdir(W)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
FILES = __FILES__
SPEC = __SPEC__
for relative, encoded in FILES.items():
    p = W / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(encoded))

def run(args, cwd=W, env=None):
    print('$', ' '.join(map(str,args)), flush=True)
    subprocess.run(list(map(str,args)), cwd=cwd, env=env, check=True)

def write_status(state, **extra):
    (W/'job_status.json').write_text(json.dumps(dict(state=state, spec=SPEC, **extra),indent=2))

write_status('setup')
try:
    run(['nvidia-smi'])
    run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.204', 'transformers>=4.40,<5',
         'numpy<2.3', 'scipy', 'addict', 'yapf==0.40.1', 'termcolor', 'pycocotools', 'gdown', 'timm'])
    run([sys.executable, 'tools/test_video_pipeline.py'])
    sam3 = glob.glob('/kaggle/input/**/sam3.pt', recursive=True)
    if SPEC['engine'] not in ('baseline','baseline_batch30') and not sam3:
        raise FileNotFoundError('Attach a sam3-weights Kaggle dataset containing sam3.pt')
    # Fingerprint the actual SAM3 weights used; never infer identity from a filename.
    sam3_sha256 = None
    if sam3 and SPEC['engine'] not in ('baseline','baseline_batch30'):
        with open(sam3[0], 'rb') as weight_file:
            sam3_sha256 = hashlib.file_digest(weight_file,'sha256').hexdigest()
    # Input cache contains the exact original sampled JPEGs; never redownload Drive per job.
    markers=glob.glob('/kaggle/input/**/countyolo_sampling.json',recursive=True)
    if len(markers)==1:
        data=pathlib.Path(markers[0]).parent
    else:
        archives=glob.glob('/kaggle/input/**/videocount-benchmark.zip',recursive=True)
        if len(archives)!=1:
            raise RuntimeError('Attach exactly one videocount-benchmark dataset/cache')
        run(['unzip','-q',archives[0],'-d',W/'data'])
        data=W/'data/VideoCount'
    if not (data/'countyolo_sampling.json').exists():
        raise RuntimeError('Cache sampling manifest is missing')
    manifest = dict(spec=SPEC, sam3_checkpoint_sha256=sam3_sha256, data_sampling_sha256=hashlib.sha256((data/'countyolo_sampling.json').read_bytes()).hexdigest(), packages={k:importlib.metadata.version(k) for k in
                    ['torch', 'torchvision', 'ultralytics', 'transformers', 'numpy']},
                    payload_sha256={p:hashlib.sha256(base64.b64decode(v)).hexdigest() for p,v in FILES.items()})
    if not SPEC['engine'].startswith('sam3'):
        cv = W/'CountVid'
        run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/niki-amini-naieni/CountVid.git', cv])
        manifest['countvid_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=cv,text=True).strip()
        env = dict(os.environ, SAM2_BUILD_CUDA='0')
        run([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/facebookresearch/sam2.git'], env=env)
        # Known-compatible PyTorch fallback avoids rebuilding legacy CUDA ops.
        f = cv/'models/GroundingDINO/ms_deform_attn.py'
        s = f.read_text().replace('\nimport MultiScaleDeformableAttention as _C\n', '\n_C = None\n', 1)
        s = s.replace('if torch.cuda.is_available() and value.is_cuda:', 'if _C is not None and value.is_cuda:')
        f.write_text(s)
        f = cv/'count_in_videos.py'
        s = f.read_text().replace('torch.load(args.pretrain_model_path, map_location="cpu")',
                                 'torch.load(args.pretrain_model_path, map_location="cpu", weights_only=False)')
        f.write_text(s)
        ck = cv/'checkpoints'
        ck.mkdir(exist_ok=True)
        if SPEC['engine'] in ('baseline','baseline_batch30'):
            run([sys.executable, 'download_bert.py'], cwd=cv)
            weights=[p for p in glob.glob('/kaggle/input/**/countgd_box.pth',recursive=True)
                     if '/countvid-weights/' in p]
            if len(weights)!=1:
                raise RuntimeError('Attach exactly one private countvid-weights dataset')
            with open(weights[0],'rb') as weight_file:
                manifest['countgd_checkpoint_sha256']=hashlib.file_digest(weight_file,'sha256').hexdigest()
            if manifest['countgd_checkpoint_sha256']!=SPEC['countgd_checkpoint_sha256']:
                raise RuntimeError('CountGD-Box checkpoint checksum mismatch')
            shutil.copyfile(weights[0],ck/'countgd_box.pth')
        run(['wget', '-q', '-O', ck/'sam2.1_hiera_large.pt',
             'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'])
        run([sys.executable, W/'countvid_ext/patch_countvid.py', cv])
    if SPEC['engine'].startswith('countyolo'):
        checkpoint = W/SPEC['checkpoint']
        checkpoint.parent.mkdir(exist_ok=True)
        if SPEC['engine'] in ('countyolo_old','countyolo_fallback'):
            run(['wget','-q','-O',checkpoint,
                 'https://raw.githubusercontent.com/Thien-Dan/countyolo/45cfbd6f42ca02a6df7b3fd4c992d5c3dac5342d/checkpoints/density_head_best_vit_seed3_ep19_valmae21.44.pth'])
        else:
            candidates = [p for p in glob.glob('/kaggle/input/**/*.pth',recursive=True)
                          if '/countyolo-exnoise/' in p and pathlib.Path(p).name == 'density_head_best.pth']
            if len(candidates) != 1:
                raise RuntimeError(f'Expected one exnoise checkpoint in attached dataset, found {len(candidates)}')
            shutil.copyfile(candidates[0],checkpoint)
        manifest['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if manifest['checkpoint_sha256'] != SPEC['checkpoint_sha256']:
            raise RuntimeError('Attached checkpoint differs from the declared checkpoint')
    (W/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    write_status('running')
    videos = ['--videos'] + SPEC['videos'] if SPEC['videos'] else []
    results = W/'results.json'
    if SPEC['engine'] == 'sam3_aligned_fp32':
        run([sys.executable, W/'tools/sam3_image_probe.py', '--data-root', data,
             '--weights', sam3[0], '--output', W/'sam3_image_probe.json'])
    if SPEC['engine'].startswith('sam3'):
        run([sys.executable, W/'tools/eval_video_sam3.py', '--data-root', data, '--dataset', SPEC['dataset'],
             '--weights', sam3[0], '--results', results, '--name', SPEC['engine'],
             '--conf', str(SPEC.get('output_conf',0.3)),
             '--detection-conf',str(SPEC['detection_conf']),
             '--tracker-profile', SPEC.get('tracker_profile','legacy'),
             '--precision', SPEC.get('precision','fp16'),
             '--object-batch-size', str(SPEC.get('object_batch_size',0))] + videos)
    else:
        extra = SPEC['extra'].replace('@CY@',str(W)).replace('@CK@',str(W/SPEC['checkpoint']))
        if '@S3@' in extra:
            extra = extra.replace('@S3@',sam3[0])
        run([sys.executable, W/'countvid_ext/run_videocount.py', '--countvid_dir',cv,'--data_root',data,
             '--checkpoint_dir',ck,'--dataset',SPEC['dataset'],'--name',SPEC['engine'],
             '--extra='+extra,'--results',results] + videos)
    result = json.loads(results.read_text())
    if not result or not all(r['complete'] for r in result.values()):
        raise RuntimeError('Evaluation incomplete; inspect per-case logs')
    write_status('complete')
except BaseException as e:
    write_status('failed', error=f'{type(e).__name__}: {e}')
    traceback.print_exc()
    raise
finally:
    # Keep small logs, traces, manifest and results; drop large downloaded inputs.
    for name in ('data', 'CountVid'):
        target = W/name
        if target.exists() and target.resolve().parent == W.resolve():
            shutil.rmtree(target)
'''


def prepare(username, engine, dataset, videos=None, out_dir=None, sam3_dataset=None, data_dataset=None):
    if engine not in CONFIGS:
        raise ValueError(engine)
    videos = videos or []
    checkpoint = ('checkpoints/density_head_best_vit_seed3_ep19_valmae21.44.pth' if engine in ('countyolo_old','countyolo_fallback')
                  else 'checkpoints/density_head_best_vit_exnoise.pth')
    sources = SOURCES
    files = {p: base64.b64encode((ROOT/p).read_bytes()).decode() for p in sources}
    tag = '-'.join(videos).lower().replace('_','-') if videos else 'all'
    # Unique kernel per configuration AND selected video; no overwrite of a running task.
    slug = f'countyolo-video-{engine.replace("_","-")}-{dataset}-{tag}'
    if len(slug) > 50:
        slug = slug[:39] + '-' + hashlib.sha256(slug.encode()).hexdigest()[:10]
    spec = dict(engine=engine, dataset=dataset, videos=videos, checkpoint=checkpoint,
                data_dataset=data_dataset or f'{username}/videocount-benchmark',
                sam3_dataset=sam3_dataset or f'{username}/sam3-weights',
                countgd_checkpoint_sha256=COUNTGD_BOX_SHA256 if engine in ('baseline','baseline_batch30') else None,
                checkpoint_sha256=hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest() if engine.startswith('countyolo') else None,
                tracker_profile='aligned' if engine.startswith('sam3_aligned') else 'legacy',
                precision='fp32' if engine.endswith('_fp32') else 'fp16',
                output_conf=0.5 if engine.startswith('sam3_aligned') else 0.3,
                object_batch_size=8 if engine == 'sam3_aligned_batch8' else 0,
                extra=CONFIGS[engine], detection_conf=0.3 if engine == 'sam3' else 0.5, protocol='countvid-text-v1')
    runner = RUNNER.replace('__FILES__', repr(files)).replace('__SPEC__', repr(spec))
    compile(runner, 'run.py', 'exec')
    if len(runner.encode('utf-8')) >= 1_000_000:
        raise ValueError('Kaggle script exceeds 1 MB; attach large artifacts as datasets')
    folder = Path(out_dir) if out_dir else ROOT/'kaggle_video_run'/slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'run.py').write_text(runner, encoding='utf-8', newline='\n')
    meta = dict(id=f'{username}/{slug}', title=slug, code_file='run.py', language='python',
                kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True,
                dataset_sources=([spec['data_dataset']] + ([f'{username}/countvid-weights'] if engine in ('baseline','baseline_batch30') else []) + ([spec['sam3_dataset']] if engine not in ('baseline','baseline_batch30') else [])
                                 + ([f'{username}/countyolo-exnoise'] if engine.startswith('countyolo') and engine not in ('countyolo_old','countyolo_fallback') else [])),
                competition_sources=[], kernel_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return dict(ref=meta['id'], folder=str(folder.resolve()), spec=spec,
                code_sha256=hashlib.sha256((folder/'run.py').read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--username', required=True)
    p.add_argument('--engine', choices=CONFIGS, required=True)
    p.add_argument('--dataset', choices=['crystals','penguins','mot20'], required=True)
    p.add_argument('--videos', nargs='+')
    p.add_argument('--sam3-dataset')
    p.add_argument('--push', action='store_true')
    a = p.parse_args()
    job = prepare(a.username,a.engine,a.dataset,a.videos,sam3_dataset=a.sam3_dataset)
    print(json.dumps(job,indent=2))
    if a.push:
        sys.path.insert(0,str(ROOT/'tools'))
        from kaggle_watch import accounts, api_for
        api = api_for(a.username, accounts()[a.username])
        api.kernels_push(job['folder'])


if __name__ == '__main__':
    main()
