"""
Đo mức mô hình "nghe theo text" (tương tự ý tưởng benchmark PrACo) trên FSC-147:
  1. đúng lớp     : MAE như thường lệ
  2. sai lớp      : hỏi một lớp không có trong ảnh -> đếm lý tưởng = 0 (báo cáo số đếm TB và tỉ lệ so với GT)
  3. mosaic 2 lớp : ghép 2 ô lớp A + 2 ô lớp B, hỏi A rồi hỏi B -> MAE so với đúng số vật của lớp được hỏi

    python tools/prompt_eval.py --checkpoint <ckpt> --data-dir data/FSC147 --split test --out prompt_eval.json
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
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data/FSC147")
    p.add_argument("--split", default="test")
    p.add_argument("--mosaics", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--oracle-exemplars", action="store_true",
                   help="Mosaic: đưa exemplar = điểm GT của lớp được hỏi (giới hạn trên khi có exemplar đúng lớp, vd SAM 3)")
    p.add_argument("--out", default="prompt_eval.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DensityYOLOWorld("yolov8s-world.pt")
    ckpt = model.load_head(a.checkpoint, map_location="cpu")
    model.to(device).eval_mode()
    ds = FSC147Dataset(a.data_dir, split=a.split)
    rng = random.Random(a.seed)
    classes = sorted(ds.class_to_idx)
    count = lambda img, cls, ex=None: float(model(img[None].to(device), [cls], exemplar_points=ex)[1].sum())

    # 1 + 2: đúng lớp / sai lớp trên từng ảnh
    pos_err, neg_cnt, neg_ratio = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        wrong = rng.choice([c for c in classes if c != s['class_name']])
        c_pos, c_neg = count(s['image'], s['class_name']), count(s['image'], wrong)
        pos_err.append(abs(c_pos - s['count']))
        neg_cnt.append(c_neg)
        neg_ratio.append(c_neg / max(s['count'], 1))

    # 3: mosaic 2x2 từ 2 lớp khác nhau (2 ô mỗi lớp, vị trí ngẫu nhiên)
    mos_err, mos_rel = [], []
    for _ in range(a.mosaics):
        ca, cb = rng.sample([c for c in classes if len(ds.class_to_idx[c]) >= 2], 2)
        tiles = [(ca, j) for j in rng.sample(ds.class_to_idx[ca], 2)] + \
                [(cb, j) for j in rng.sample(ds.class_to_idx[cb], 2)]
        rng.shuffle(tiles)
        half = ds.image_size // 2
        image = torch.zeros(3, ds.image_size, ds.image_size)
        gt = {ca: 0.0, cb: 0.0}
        pts = {ca: [], cb: []}
        for k, (c, j) in enumerate(tiles):
            s = ds._load(j)
            oy, ox = divmod(k, 2)
            image[:, oy * half:(oy + 1) * half, ox * half:(ox + 1) * half] = \
                F.interpolate(s['image'][None], size=(half, half), mode='area')[0]
            gt[c] += s['count']
            pts[c].append(s['points'] / 2 + torch.tensor([ox * half, oy * half], dtype=torch.float32))
        for c in (ca, cb):
            ex = [torch.cat(pts[c])] if a.oracle_exemplars and len(torch.cat(pts[c])) else None
            pred = count(image, c, ex)
            mos_err.append(abs(pred - gt[c]))
            mos_rel.append(abs(pred - gt[c]) / max(gt[c], 1))

    res = {"checkpoint": os.path.basename(a.checkpoint), "arch": ckpt.get("arch"), "split": a.split,
           "correct_prompt_mae": float(np.mean(pos_err)),
           "wrong_prompt_mean_count": float(np.mean(neg_cnt)),
           "wrong_prompt_median_ratio_to_gt": float(np.median(neg_ratio)),
           "mosaic_mae": float(np.mean(mos_err)), "mosaic_median_rel_err": float(np.median(mos_rel))}
    print(f"\n=== Prompt-awareness [{res['checkpoint']}, arch={res['arch']}, {a.split}] ===")
    print(f"  Đúng lớp      : MAE {res['correct_prompt_mae']:.2f}")
    print(f"  Sai lớp       : đếm TB {res['wrong_prompt_mean_count']:.1f} | trung vị (đếm / GT lớp thật) "
          f"{res['wrong_prompt_median_ratio_to_gt']:.2f}  (lý tưởng 0)")
    print(f"  Mosaic 2 lớp  : MAE {res['mosaic_mae']:.2f} | sai số tương đối trung vị "
          f"{res['mosaic_median_rel_err']:.2f}  ({a.mosaics} mosaic x 2 câu hỏi)")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
