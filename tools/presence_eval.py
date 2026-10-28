"""
Cổng text dùng ở mức "có / không" (vitgate): điểm presence = trung bình cổng có trọng số density. Nếu presence >= ngưỡng
thì giữ nguyên density của phần đếm (không nhân cổng), ngược lại đếm 0. Ngưỡng chọn trên VAL theo
    MAE (hỏi đúng lớp) + số đếm TB (hỏi sai lớp)
rồi báo cáo TEST: MAE, hỏi sai lớp, mosaic 2 lớp (cùng cách dựng với tools/prompt_eval.py).

    python tools/presence_eval.py --checkpoint <ckpt vitgate> --data-dir data/FSC147 --out presence_eval.json
"""
# Chạy được từ thư mục gốc dự án: python tools/<script>.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from datasets.fsc147_dataset import FSC147Dataset  # noqa: E402
from models.density_yolo import DensityYOLOWorld  # noqa: E402


@torch.no_grad()
def score(model, image, cls, device):
    """(presence, số đếm KHÔNG nhân cổng) của một ảnh (3, S, S) với một prompt."""
    _, d, aux = model(image[None].to(device), [cls], return_aux=True)
    return float(aux["presence"][0]), float(d.float().sum())


def per_image(model, ds, device, rng, classes):
    pos, neg = [], []
    for i in range(len(ds)):
        s = ds[i]
        wrong = rng.choice([c for c in classes if c != s['class_name']])
        pos.append((*score(model, s['image'], s['class_name'], device), s['count']))
        neg.append((*score(model, s['image'], wrong, device), s['count']))
    return np.array(pos), np.array(neg)


def apply(rows, t):
    """rows (N, 3) = presence, count, gt -> số đếm sau ngưỡng."""
    return rows[:, 1] * (rows[:, 0] >= t)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data/FSC147")
    p.add_argument("--mosaics", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="presence_eval.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DensityYOLOWorld("yolov8s-world.pt")
    model.load_head(a.checkpoint, map_location="cpu")
    model.to(device).eval_mode()
    if model.gate is None:
        sys.exit("Checkpoint không có cổng text (cần arch vitgate)")
    model.presence_thr = -1.0  # luôn giữ: lấy density chưa nhân cổng + điểm presence

    # 1. VAL: chọn ngưỡng
    val = FSC147Dataset(a.data_dir, split="val")
    rng = random.Random(a.seed)
    vp, vn = per_image(model, val, device, rng, sorted(val.class_to_idx))
    cands = np.unique(np.quantile(np.concatenate([vp[:, 0], vn[:, 0]]), np.linspace(0, 1, 101)))
    best, best_obj = 0.0, float("inf")
    for t in np.concatenate([[0.0], cands]):
        obj = np.abs(apply(vp, t) - vp[:, 2]).mean() + apply(vn, t).mean()
        if obj < best_obj:
            best, best_obj = float(t), obj
    print(f"=== VAL: ngưỡng presence {best:.4f} | MAE đúng lớp {np.abs(apply(vp, best) - vp[:, 2]).mean():.2f} "
          f"(không ngưỡng {np.abs(vp[:, 1] - vp[:, 2]).mean():.2f}) | đếm TB sai lớp {apply(vn, best).mean():.2f} "
          f"(không ngưỡng {vn[:, 1].mean():.2f}) | presence TB đúng {vp[:, 0].mean():.3f} / sai {vn[:, 0].mean():.3f}")

    # 2. TEST
    test = FSC147Dataset(a.data_dir, split="test")
    classes = sorted(test.class_to_idx)
    tp, tn = per_image(model, test, device, rng, classes)
    res = {"threshold": best,
           "test_mae": float(np.abs(apply(tp, best) - tp[:, 2]).mean()),
           "test_mae_no_gate": float(np.abs(tp[:, 1] - tp[:, 2]).mean()),
           "wrong_prompt_mean_count": float(apply(tn, best).mean()),
           "wrong_prompt_median_ratio_to_gt": float(np.median(apply(tn, best) / np.maximum(tn[:, 2], 1))),
           "correct_suppressed_pct": float((tp[:, 0] < best).mean() * 100)}

    # 3. mosaic 2 lớp (như tools/prompt_eval.py)
    mos_err, mos_rel = [], []
    for _ in range(a.mosaics):
        ca, cb = rng.sample([c for c in classes if len(test.class_to_idx[c]) >= 2], 2)
        tiles = [(ca, j) for j in rng.sample(test.class_to_idx[ca], 2)] + \
                [(cb, j) for j in rng.sample(test.class_to_idx[cb], 2)]
        rng.shuffle(tiles)
        half = test.image_size // 2
        image = torch.zeros(3, test.image_size, test.image_size)
        gt = {ca: 0.0, cb: 0.0}
        for k, (c, j) in enumerate(tiles):
            s = test._load(j)
            oy, ox = divmod(k, 2)
            image[:, oy * half:(oy + 1) * half, ox * half:(ox + 1) * half] = \
                F.interpolate(s['image'][None], size=(half, half), mode='area')[0]
            gt[c] += s['count']
        for c in (ca, cb):
            pr, cnt = score(model, image, c, device)
            pred = cnt * (pr >= best)
            mos_err.append(abs(pred - gt[c]))
            mos_rel.append(abs(pred - gt[c]) / max(gt[c], 1))
    res.update(mosaic_mae=float(np.mean(mos_err)), mosaic_median_rel_err=float(np.median(mos_rel)))
    print(f"=== TEST (ngưỡng {best:.4f}) ===\n  Đúng lớp: MAE {res['test_mae']:.2f} (không cổng {res['test_mae_no_gate']:.2f}) "
          f"| bị chặn nhầm {res['correct_suppressed_pct']:.1f}% ảnh\n  Sai lớp: đếm TB {res['wrong_prompt_mean_count']:.2f} "
          f"| trung vị đếm/GT {res['wrong_prompt_median_ratio_to_gt']:.2f}\n  Mosaic 2 lớp: MAE {res['mosaic_mae']:.2f} "
          f"| sai số tương đối trung vị {res['mosaic_median_rel_err']:.2f}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
