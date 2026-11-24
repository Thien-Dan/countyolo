"""Evaluate complete OV-DQUO v1 + SAM 2.1 + Long-Term Tracking + TemporalIdentity pipeline on VideoCount."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from models.ov_dquo_video_detector import load_ov_dquo_video_detector
from video.eval_common import atomic_json, cases, summarize
from video.ov_dquo_sam2_pipeline import build_sam2_predictors, count_video_ov_dquo_sam2


def summarize_stitched(rows, expected):
    """Exact mirror of `eval_common.summarize` evaluated on `stitched_pred`."""
    ok = [r for r in rows if r.get('status') == 'complete']
    errors = [r['stitched_pred'] - r['gt'] for r in ok]
    n = len(errors)
    seconds = sum(r.get('sec', 0.0) for r in ok)
    return dict(
        complete=len(rows) == expected and n == expected,
        expected=expected,
        n=n,
        failed=sum(r.get('status') == 'failed' for r in rows),
        mae=sum(map(abs, errors)) / n if n else None,
        rmse=math.sqrt(sum(e * e for e in errors) / n) if n else None,
        bias=sum(errors) / n if n else None,
        wape=sum(map(abs, errors)) / max(sum(r['gt'] for r in ok), 1) if n else None,
        sec=seconds,
        fps=sum(r.get('n_frames', 0) for r in ok) / seconds if seconds else None,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', choices=['penguins', 'crystals', 'mot20'],
                   default=['penguins', 'crystals', 'mot20'])
    p.add_argument('--videos', nargs='*', default=None)
    p.add_argument('--base-checkpoint', type=Path, required=True, help='Official OVDQUO_ViT-B16/L14_LVIS.pth')
    p.add_argument('--tuned-checkpoint', type=Path, required=True, help='Fine-tuned OV-DQUO v1/v2/v3.1 best.pth')
    p.add_argument('--sam-checkpoint', type=Path, required=True, help='SAM 2.1 sam2.1_hiera_large.pt')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resolution', type=int, default=None)
    p.add_argument('--threshold', type=float, default=0.26)
    p.add_argument('--tile-trigger', type=int, default=400)
    p.add_argument('--window', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=30)
    p.add_argument('--box-chunk', type=int, default=64)
    p.add_argument('--save-previews', action='store_true')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INIT] Auto-loading OV-DQUO detector from {a.tuned_checkpoint.name} (thr={a.threshold}) on {device}...', flush=True)
    detector = load_ov_dquo_video_detector(
        base_checkpoint_path=str(a.base_checkpoint),
        tuned_checkpoint_path=str(a.tuned_checkpoint),
        device=device,
        resolution=a.resolution,
    )
    print(f'[INIT] Loading SAM 2.1 Large from {a.sam_checkpoint}...', flush=True)
    sam_predictors = build_sam2_predictors(a.sam_checkpoint, torch.device(device))

    overall_summary = dict(
        architecture=detector.ARCHITECTURE,
        threshold=a.threshold,
        tile_trigger=a.tile_trigger,
        window=a.window,
        batch_size=a.batch_size,
        datasets={},
    )
    all_rows = []
    total_expected = 0

    for ds in a.datasets:
        ds_cases = cases(a.data_root, ds, videos=a.videos if len(a.datasets) == 1 else None)
        # Run lighter videos first inside each dataset so partial results appear quickly
        ds_cases = sorted(ds_cases, key=lambda c: (len(c['frames']), c['gt'], c['video']))
        total_expected += len(ds_cases)
        ds_rows = []
        ds_out = a.output_dir / f'results_{ds}.json'
        preview_dir = (a.output_dir / 'previews' / ds) if a.save_previews else None

        for idx, case in enumerate(ds_cases, 1):
            vid, prompt, gt = case['video'], case['text'], case['gt']
            frames = case['frames']
            print(f'[{ds} {idx}/{len(ds_cases)}] START {vid} prompt="{prompt}" frames={len(frames)} gt={gt}', flush=True)
            try:
                out = count_video_ov_dquo_sam2(
                    detector=detector,
                    frames=frames,
                    text=prompt,
                    device=device,
                    threshold=a.threshold,
                    tile_trigger=a.tile_trigger,
                    window=a.window,
                    batch_size=a.batch_size,
                    box_chunk=a.box_chunk,
                    sam_predictors=sam_predictors,
                    preview_dir=preview_dir,
                    preview_prefix=f'{ds}_{vid}',
                )
                row = dict(
                    status='complete',
                    dataset=ds,
                    video=vid,
                    text=prompt,
                    gt=gt,
                    pred=out['count'],
                    stitched_pred=out['stitched_count'],
                    identity_links_count=out['identity_links_count'],
                    identity_uncertain_count=out['identity_uncertain_count'],
                    n_frames=len(frames),
                    sec=round(out['sec'], 2),
                    fps=round(out['fps'], 3),
                    mean_proposals=round(sum(out['proposal_counts']) / max(1, len(out['proposal_counts'])), 2),
                    mean_independent=round(sum(out['independent_counts']) / max(1, len(out['independent_counts'])), 2),
                    mean_filtered=round(sum(out['filtered_counts']) / max(1, len(out['filtered_counts'])), 2),
                    tiled_frames=out['tiled_frames'],
                    stage_times={k: round(v, 2) for k, v in out['stage_times'].items()},
                    frame_counts=out['frame_counts'],
                    frame_sha256=case['frame_sha256'],
                    preview_files=out['preview_files'],
                )
                print(
                    f'[{ds} {idx}/{len(ds_cases)}] DONE  {vid} | GT={gt} '
                    f'| sam2={row["pred"]} (err={row["pred"]-gt:+d}) '
                    f'| stitched={row["stitched_pred"]} (err={row["stitched_pred"]-gt:+d}, links={row["identity_links_count"]}) '
                    f'| props={row["mean_proposals"]:.1f} filt={row["mean_filtered"]:.1f} '
                    f'| sec={row["sec"]:.1f}s ({row["fps"]:.2f} fps)',
                    flush=True,
                )
            except Exception as exc:
                traceback.print_exc()
                row = dict(
                    status='failed',
                    dataset=ds,
                    video=vid,
                    text=prompt,
                    gt=gt,
                    n_frames=len(frames),
                    error=f'{type(exc).__name__}: {exc}',
                )
                print(f'[{ds} {idx}/{len(ds_cases)}] FAIL  {vid}: {row["error"]}', flush=True)

            ds_rows.append(row)
            all_rows.append(row)
            ds_payload = dict(
                dataset=ds,
                sam2=summarize(ds_rows, len(ds_cases)),
                stitched=summarize_stitched(ds_rows, len(ds_cases)),
            )
            atomic_json(ds_out, ds_payload)
            overall_summary['datasets'][ds] = dict(
                sam2={k: v for k, v in ds_payload['sam2'].items() if k != 'rows'},
                stitched=ds_payload['stitched'],
            )
            overall_summary['overall_sam2'] = {k: v for k, v in summarize(all_rows, total_expected).items() if k != 'rows'}
            overall_summary['overall_stitched'] = summarize_stitched(all_rows, total_expected)
            atomic_json(a.output_dir / 'summary.json', overall_summary)

        s_sam = overall_summary['datasets'][ds]['sam2']
        s_sti = overall_summary['datasets'][ds]['stitched']
        print(
            f'=== DATASET {ds.upper()} COMPLETE ({s_sam["n"]}/{s_sam["expected"]}) === '
            f'SAM2 MAE={s_sam["mae"]:.3f} RMSE={s_sam["rmse"]:.3f} | '
            f'STITCHED MAE={s_sti["mae"]:.3f} RMSE={s_sti["rmse"]:.3f}',
            flush=True,
        )

    print('=== OVERALL SUMMARY ===', flush=True)
    print(json.dumps(overall_summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
