"""Run patched CountVid; checkpoint every video/prompt and preserve full logs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video.eval_common import DATASETS, atomic_json, cases, save_result

FLAGS = {
    'crystals': '--obj_batch_size 30 --img_batch_size 4 --downsample_factor 1 --temporal_filter --w 3 --convert_to_rgb',
    'penguins': '--obj_batch_size 30 --img_batch_size 4 --downsample_factor 1 --temporal_filter --w 3',
    'mot20': f'--obj_batch_size 10 --img_batch_size 4 --downsample_factor {25/3}',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--countvid_dir', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--checkpoint_dir', default='checkpoints')
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--extra', default='')
    p.add_argument('--videos', nargs='*')
    p.add_argument('--results', required=True)
    p.add_argument('--timeout', type=int, default=14400)
    a = p.parse_args()
    selected = cases(a.data_root, a.dataset, a.videos)
    rows, key = [], f'{a.dataset}/{a.name}'
    metadata = dict(extra=a.extra, protocol='countvid-text-v1', schema_version=2)
    save_result(a.results, key, rows, len(selected), **metadata)
    ckpt = Path(a.checkpoint_dir).resolve()
    logs = Path(a.results).resolve().parent / 'logs'
    logs.mkdir(exist_ok=True)
    for c in selected:
        row = {k: v for k, v in c.items() if k != 'frames'}
        tag = hashlib.sha256(c['case_id'].encode()).hexdigest()[:12]
        out_file = logs / f'pred_{a.dataset}_{a.name}_{tag}.json'
        log_file = logs / f'{a.dataset}_{a.name}_{tag}.log'
        # Resolve once; this task owns exactly this temporary directory.
        temp_dir = (Path(a.results).resolve().parent / f'temp_{a.dataset}_{a.name}_{tag}').resolve()
        video_dir = str(Path(c['frames'][0]).parent)
        flags=shlex.split(FLAGS[a.dataset])
        flags[flags.index('--downsample_factor')+1]=str(c.get('inference_downsample',c.get('downsample',DATASETS[a.dataset][2])))
        cmd = [sys.executable, 'count_in_videos.py', '--video_dir', video_dir, '--input_text', c['text'],
               '--sam_checkpoint', str(ckpt / 'sam2.1_hiera_large.pt'),
               '--sam_model_cfg', 'configs/sam2.1/sam2.1_hiera_l.yaml',
               '--pretrain_model_path', str(ckpt / 'countgd_box.pth'), '--output_file', str(out_file),
               '--temp_dir', str(temp_dir)] + flags + shlex.split(a.extra)
        t0 = time.perf_counter()
        try:
            # Upstream CountVid updates an existing JSON object at the end.
            atomic_json(out_file,{})
            with log_file.open('w', encoding='utf-8') as log:
                subprocess.run(cmd, cwd=a.countvid_dir, stdout=log, stderr=subprocess.STDOUT,
                               timeout=a.timeout, check=True)
            preds = json.loads(out_file.read_text())
            matches = [v[c['text']] for k, v in preds.items()
                       if (k.replace('\\', '/').endswith('/' + c['video']) or k == c['video']) and c['text'] in v]
            if len(matches) != 1:
                raise ValueError(f'Expected one prediction, found {len(matches)}')
            pred = matches[0]
            if not isinstance(pred, (int, float)) or not math.isfinite(pred) or pred < 0:
                raise ValueError('Invalid prediction')
            row.update(status='complete', pred=pred, sec=time.perf_counter()-t0, n_frames=len(c['frames']), log=str(log_file))
            print(f"{key} {c['video']} {c['text']!r}: pred={pred} gt={c['gt']} ({row['sec']:.0f}s)", flush=True)
            for line in log_file.read_text(encoding='utf-8', errors='replace').splitlines():
                if line.startswith(('time stage', 'tiling:', 'sam3 keyframe:')):
                    print('  ' + line, flush=True)
        except Exception as e:
            row.update(status='failed', error=f'{type(e).__name__}: {e}', sec=time.perf_counter()-t0, log=str(log_file))
            traceback.print_exc()
            if log_file.exists():
                print(log_file.read_text(encoding='utf-8', errors='replace')[-4000:], flush=True)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        rows.append(row)
        save_result(a.results, key, rows, len(selected), **metadata)
    if any(r['status'] != 'complete' for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
