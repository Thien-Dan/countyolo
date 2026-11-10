"""Shared VideoCount protocol and atomic, per-prompt evaluation records."""
import hashlib
import json
import math
import os
from pathlib import Path
import re

DATASETS = {
    'crystals': ('Crystals', 'crystals-count-gt.json', 1.0),
    'penguins': ('Penguins', 'penguins-count-gt.json', 1.0),
    'mot20': ('MOT20-Count', 'MOT20-count-gt.json', 25 / 3),
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def frame_paths(folder, downsample=1):
    import numpy as np
    names = [p for p in Path(folder).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg')]
    names.sort(key=lambda p: int(p.stem[re.search(r'\d', p.stem).start():]))
    if not names:
        raise ValueError(f'No JPEG frames in {folder}')
    # Exact CountVid sampling: round(linspace(..., ceil(n/factor))).
    idx = np.round(np.linspace(0, len(names) - 1, int(np.ceil(len(names) / downsample)))).astype(int)
    return [str(names[i]) for i in idx]


def cases(data_root, dataset, videos=None):
    sub, annotation, downsample = DATASETS[dataset]
    folder = Path(data_root) / sub
    gt = json.loads((folder / 'anno' / annotation).read_text(encoding='utf-8'))
    unknown = set(videos or []) - set(gt)
    if unknown:
        raise ValueError(f'Unknown videos: {sorted(unknown)}')
    marker_path=Path(data_root)/'countyolo_sampling.json'
    cached=json.loads(marker_path.read_text(encoding='utf-8')) if marker_path.exists() else None
    if cached is not None and (cached.get('protocol')!='countvid-text-v1' or cached.get('pre_sampled') is not True):
        raise ValueError('Unrecognized cached sampling protocol')
    rows = []
    for video, prompts in gt.items():
        if videos and video not in videos:
            continue
        inference_downsample=1.0 if cached else downsample
        frames = frame_paths(folder / 'frames' / video, inference_downsample)
        manifest = [Path(f).name for f in frames]
        if cached:
            saved=cached['videos'][dataset][video]
            if manifest!=saved['frame_names'] or saved['original_downsample']!=downsample:
                raise ValueError(f'Cached frame selection changed for {dataset}/{video}')
        digest = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
        for text, count in prompts.items():
            rows.append(dict(case_id=json.dumps([video, text], ensure_ascii=False), video=video, text=text,
                             gt=len(count) if isinstance(count, list) else count, frames=frames,
                             frame_manifest=manifest, frame_sha256=digest, downsample=downsample,
                             inference_downsample=inference_downsample))
    if not rows:
        raise ValueError('No selected video/prompt cases')
    return rows


def summarize(rows, expected):
    ok = [r for r in rows if r.get('status') == 'complete']
    errors = [r['pred'] - r['gt'] for r in ok]
    n = len(errors)
    seconds = sum(r.get('sec', 0) for r in ok)
    return dict(complete=len(rows) == expected and n == expected, expected=expected, n=n,
                failed=sum(r.get('status') == 'failed' for r in rows),
                mae=sum(map(abs, errors)) / n if n else None,
                rmse=math.sqrt(sum(e * e for e in errors) / n) if n else None,
                bias=sum(errors) / n if n else None,
                wape=sum(map(abs, errors)) / max(sum(r['gt'] for r in ok), 1) if n else None,
                sec=seconds, fps=sum(r.get('n_frames', 0) for r in ok) / seconds if seconds else None,
                rows=rows)


def save_result(path, key, rows, expected, **metadata):
    results = json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else {}
    results[key] = dict(summarize(rows, expected), **metadata)
    atomic_json(path, results)
