"""Small FP16/FP32 image diagnostic on unchanged cached video frames, Kaggle only."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video.eval_common import cases, atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    import torch
    from engines.sam3_counter import SAM3Counter
    selected = cases(a.data_root, 'penguins', ['penguins_1'])
    selected += cases(a.data_root, 'crystals', ['ma2035_023', 'ma2035_039'])
    rows = []
    # Ground truth is neither consulted nor used to select a threshold.
    for precision in ('fp16', 'fp32'):
        counter = SAM3Counter(a.weights, conf=0.3, half=precision=='fp16', max_grid=1)
        for c in selected:
            row = dict(case_id=c['case_id'], precision=precision, frame=c['frame_manifest'][0],
                       text=c['text'], conf=0.3)
            t0 = time.perf_counter()
            try:
                r = counter.count_image(c['frames'][0], c['text'])
                row.update(status='complete', count=r['count'], numerics=r['numerics'],
                           scores=r['scores'].tolist(), boxes=r['boxes'].tolist())
            except Exception as e:
                row.update(status='failed', error=f'{type(e).__name__}: {e}')
                traceback.print_exc()
            row['sec'] = time.perf_counter()-t0
            rows.append(row)
            atomic_json(a.output, dict(purpose='numerical setup diagnostic, not video counting metric',rows=rows))
            print('SAM3 image probe:', {k:v for k,v in row.items() if k not in ('boxes','scores')}, flush=True)
        del counter
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
