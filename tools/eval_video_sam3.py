"""Evaluate native SAM 3 video tracking on the exact CountVid video/prompt cases."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video.eval_common import DATASETS, atomic_json, cases, save_result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--conf', type=float, default=0.3)
    p.add_argument('--detection-conf', type=float, default=0.5)
    p.add_argument('--tracker-profile', choices=['legacy','aligned'], default='legacy')
    p.add_argument('--object-batch-size',type=int,default=0)
    p.add_argument('--precision', choices=['fp16','fp32'], default='fp16')
    p.add_argument('--videos', nargs='+')
    p.add_argument('--results', required=True)
    p.add_argument('--name', default='sam3')
    a = p.parse_args()
    from engines.sam3_counter import SAM3Counter
    import torch
    selected = cases(a.data_root, a.dataset, a.videos)
    rows = []
    key = f'{a.dataset}/{a.name}'
    metadata = dict(engine='SAM3VideoSemanticPredictor', conf=a.conf, detection_conf=a.detection_conf, protocol='countvid-text-v1',
                    count_rule='union of emitted track IDs; persistence>=3 supplementary',
                    tracker_profile=a.tracker_profile, precision=a.precision, object_batch_size=a.object_batch_size, schema_version=3)
    save_result(a.results, key, rows, len(selected), **metadata)
    for c in selected:
        row = {k: v for k, v in c.items() if k != 'frames'}
        tag = hashlib.sha256(c['case_id'].encode()).hexdigest()[:12]
        trace = str(Path(a.results).parent / 'traces' / f'{a.dataset}_{a.name}_{tag}.jsonl')
        t0 = time.perf_counter()
        counter = None
        try:
            counter = SAM3Counter(a.weights, conf=a.conf, half=a.precision == "fp16")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            r = counter.count_video(c['frames'], c['text'], min_hits=1, trace_path=trace, detection_conf=a.detection_conf, tracker_profile=a.tracker_profile, object_batch_size=a.object_batch_size)
            filtered = sum(n >= 3 for n in r['track_hits'].values())
            row.update(status='complete', pred=r['count'], pred_min_hits3=filtered, n_frames=r['frames'],
                       raw_pred=r['raw_count'], tracker_settings=r['tracker_settings'],
                       weight_load_reports=r['weight_load_reports'],
                       peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0,
                       frame_counts=r['frame_counts'], track_hits=r['track_hits'], encode_sec=r['encode_sec'],
                       trace=trace, sec=time.perf_counter() - t0)
            print(f"{key} {c['video']} {c['text']!r}: pred={r['count']} persistent3={filtered} gt={c['gt']}", flush=True)
        except Exception as e:
            row.update(status='failed', error=f'{type(e).__name__}: {e}', sec=time.perf_counter()-t0)
            traceback.print_exc()
        finally:
            del counter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(row)
        save_result(a.results, key, rows, len(selected), **metadata)
    if any(r['status'] != 'complete' for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
