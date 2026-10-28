"""
Chọn ngưỡng min_mass (khối lượng density 3x3 quanh đỉnh) cho việc biến density thành điểm/box, trên FSC-147 VAL:
ghép đỉnh với điểm GT (Hungarian, bán kính = max(8px, nửa kích thước exemplar)), chọn ngưỡng có F1 cao nhất.
Điểm/box dùng làm prompt khi đếm video, nên precision quan trọng: mỗi đỉnh giả thành một track giả.

    python tools/calibrate_peaks.py --checkpoint <ckpt> --data-dir data/FSC147 --out calib.json
"""
# Chạy được từ thư mục gốc dự án: python tools/<script>.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from datasets.fsc147_dataset import FSC147Dataset
from models.density_yolo import DensityYOLOWorld
from utils.scale_zoom import _peak_boxes

THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def match(pred, gt, radius):
    """Số cặp (pred, gt) ghép được trong bán kính radius (Hungarian trên khoảng cách)."""
    if len(pred) == 0 or len(gt) == 0:
        return 0
    dist = np.linalg.norm(pred[:, None] - gt[None], axis=2)
    r, c = linear_sum_assignment(np.where(dist <= radius, dist, 1e6))
    return int((dist[r, c] <= radius).sum())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data/FSC147")
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="calib.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DensityYOLOWorld("yolov8s-world.pt")
    model.load_head(a.checkpoint, map_location="cpu")
    model.to(device).eval_mode()
    ds = FSC147Dataset(a.data_dir, split=a.split)
    n = min(a.limit, len(ds)) if a.limit else len(ds)

    stats = {t: {"tp": 0, "fp": 0, "fn": 0, "abs_err": 0.0} for t in THRESHOLDS}
    for i in range(n):
        s = ds[i]
        _, dens, aux = model(s['image'][None].to(device), [s['class_name']], return_aux=True)
        d = dens[0, 0].float()
        ls = aux['log_size'][0, 0].float() if aux['log_size'] is not None else None
        gt = s['points'].numpy()
        radius = max(8.0, 0.5 * 2 ** s['size_map'][0, 0, 0].item())
        for t in THRESHOLDS:
            boxes, _ = _peak_boxes(d, ls, float(d.sum()), min_mass=t)
            pred = ((boxes[:, :2] + boxes[:, 2:]) / 2).numpy()
            m = match(pred, gt, radius)
            st = stats[t]
            st["tp"] += m
            st["fp"] += len(pred) - m
            st["fn"] += len(gt) - m
            st["abs_err"] += abs(len(pred) - len(gt))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    res = {}
    for t, st in stats.items():
        prec = st["tp"] / max(st["tp"] + st["fp"], 1)
        rec = st["tp"] / max(st["tp"] + st["fn"], 1)
        res[str(t)] = {"precision": round(prec, 4), "recall": round(rec, 4),
                       "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
                       "point_count_mae": round(st["abs_err"] / n, 2)}
    best = max(res, key=lambda k: res[k]["f1"])
    print(f"\n=== Ngưỡng khối lượng đỉnh trên {a.split} ({n} ảnh) ===")
    for t, r in res.items():
        print(f"  min_mass {t:>4s}: P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f} "
              f"| MAE số điểm {r['point_count_mae']:.2f}" + ("  <- chọn" if t == best else ""))
    json.dump({"results": res, "best_min_mass": float(best)}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
