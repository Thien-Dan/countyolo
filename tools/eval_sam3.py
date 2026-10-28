"""
Đánh giá engine SAM 3 trên FSC-147 theo cùng giao thức với CountYOLO:
  1. chọn ngưỡng conf trên VAL (một lượt, không chia ô, lấy mọi score rồi đếm theo từng ngưỡng)
  2. TEST với ngưỡng đã chọn (có chia ô khi gần trần 200 query): MAE / RMSE / theo nhóm số vật
  3. mức nghe theo text như tools/prompt_eval.py: hỏi sai lớp + mosaic 2 lớp

    python tools/eval_sam3.py --data-dir data/FSC147 --out outputs/eval_sam3.json [--limit-val 300]
"""
# Chạy được từ thư mục gốc dự án: python tools/<script>.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from datasets.fsc147_dataset import load_class_names  # noqa: E402
from engines.sam3_counter import DEFAULT_WEIGHTS, SAM3Counter  # noqa: E402

THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/FSC147")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--limit-val", type=int, default=300, help="Số ảnh val dùng chọn ngưỡng")
    p.add_argument("--mosaics", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/eval_sam3.json")
    a = p.parse_args()
    split = json.load(open(os.path.join(a.data_dir, "Train_Test_Val_FSC_147.json")))
    anno = json.load(open(os.path.join(a.data_dir, "annotation_FSC147_384.json")))
    cls = load_class_names(a.data_dir)
    img = lambda n: Image.open(os.path.join(a.data_dir, "images_384_VarV2", n)).convert("RGB")
    rng = random.Random(a.seed)

    # 1. chọn ngưỡng trên val: một lượt với conf thấp, đếm theo từng ngưỡng
    low = SAM3Counter(a.weights, conf=min(THRESHOLDS))
    val = rng.sample(split["val"], min(a.limit_val, len(split["val"])))
    err = {t: [] for t in THRESHOLDS}
    for n in val:
        rgb = np.array(img(n))
        _, s = low._detect(rgb, cls[n])
        gt = len(anno[n]["points"])
        for t in THRESHOLDS:
            err[t].append(abs(int((s >= t).sum()) - gt))
    val_mae = {t: float(np.mean(e)) for t, e in err.items()}
    best = min(val_mae, key=val_mae.get)
    print("=== VAL (chọn ngưỡng, không chia ô) ===")
    for t, m in val_mae.items():
        print(f"  conf {t:.2f}: MAE {m:.2f}" + ("  <- chọn" if t == best else ""))

    # 2. test với ngưỡng đã chọn, có chia ô
    c = SAM3Counter(a.weights, conf=best)
    rows, t0 = [], time.perf_counter()
    for n in split["test"]:
        r = c.count_image(img(n), cls[n])
        rows.append((len(anno[n]["points"]), r["count"], r["grid"]))
    sec = time.perf_counter() - t0
    g = np.array([x[0] for x in rows], float)
    pr = np.array([x[1] for x in rows], float)
    e = pr - g
    bins = {}
    for lo, hi in [(0, 50), (50, 200), (200, 500), (500, 1e9)]:
        m = (g >= lo) & (g < hi)
        bins[f"{lo}-{hi if hi < 1e9 else 'inf'}"] = {"n": int(m.sum()), "mae": float(np.abs(e[m]).mean()),
                                                      "bias": float(e[m].mean())}
    test = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
            "tiled": int(sum(x[2] > 1 for x in rows)), "sec_per_image": sec / len(rows), "bins": bins}
    print(f"=== TEST (conf {best}, chia ô khi gần trần) ===\n  MAE {test['mae']:.2f} RMSE {test['rmse']:.2f} "
          f"bias {test['bias']:+.2f} | chia ô {test['tiled']}/{len(rows)} ảnh | {test['sec_per_image']:.2f} s/ảnh")
    for k, v in bins.items():
        print(f"  gt[{k}]: n={v['n']:4d} MAE {v['mae']:8.2f} bias {v['bias']:+8.2f}")

    # 3. nghe theo text
    classes = sorted(set(cls[n] for n in split["test"]))
    neg, ratio = [], []
    for n in split["test"]:
        wrong = rng.choice([x for x in classes if x != cls[n]])
        k = c.count_image(img(n), wrong)["count"]
        neg.append(k)
        ratio.append(k / max(len(anno[n]["points"]), 1))
    by_cls = {}
    for n in split["test"]:
        by_cls.setdefault(cls[n], []).append(n)
    mos_err, mos_rel = [], []
    for _ in range(a.mosaics):
        ca, cb = rng.sample([x for x in by_cls if len(by_cls[x]) >= 2], 2)
        tiles = [(ca, n) for n in rng.sample(by_cls[ca], 2)] + [(cb, n) for n in rng.sample(by_cls[cb], 2)]
        rng.shuffle(tiles)
        canvas = Image.new("RGB", (768, 768))
        gt = {ca: 0, cb: 0}
        for k, (cc, n) in enumerate(tiles):
            canvas.paste(img(n).resize((384, 384)), ((k % 2) * 384, (k // 2) * 384))
            gt[cc] += len(anno[n]["points"])
        for cc in (ca, cb):
            k = c.count_image(canvas, cc)["count"]
            mos_err.append(abs(k - gt[cc]))
            mos_rel.append(abs(k - gt[cc]) / max(gt[cc], 1))
    prompt = {"wrong_prompt_mean_count": float(np.mean(neg)), "wrong_prompt_median_ratio_to_gt": float(np.median(ratio)),
              "mosaic_mae": float(np.mean(mos_err)), "mosaic_median_rel_err": float(np.median(mos_rel))}
    print(f"=== Nghe theo text ===\n  Sai lớp: đếm TB {prompt['wrong_prompt_mean_count']:.1f} | trung vị đếm/GT "
          f"{prompt['wrong_prompt_median_ratio_to_gt']:.2f}\n  Mosaic 2 lớp: MAE {prompt['mosaic_mae']:.2f} | sai số "
          f"tương đối trung vị {prompt['mosaic_median_rel_err']:.2f} ({a.mosaics} mosaic x 2)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"val_mae": val_mae, "best_conf": best, "test": test, "prompt": prompt}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
