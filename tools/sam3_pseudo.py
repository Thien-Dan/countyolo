"""
Sinh nhãn giả "chỗ nào là khái niệm được hỏi" bằng SAM 3 cho FSC-147, để chưng cất khả năng phân biệt lớp của SAM 3
vào cổng text nhẹ của CountYOLO (arch vitgate). Mỗi ảnh: prompt = lớp của ảnh + K lớp khác lấy từ tên lớp FSC-147 và
COCO (lớp COCO hay có mặt thật trong ảnh -> negative khó). Mỗi (ảnh, prompt) -> hợp các mask instance của SAM 3, thu về
lưới 80x80 (tỉ lệ diện tích, uint8 0..255) + số instance.

    python tools/sam3_pseudo.py --split train --out outputs/sam3_pseudo_train.npz
"""
# Chạy được từ thư mục gốc dự án: python tools/<script>.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from datasets.fsc147_dataset import load_class_names  # noqa: E402
from engines.sam3_counter import DEFAULT_WEIGHTS  # noqa: E402

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]
GRID = 80
# Lớp hay xuất hiện thật trong ảnh: hỏi SAM 3 trên mọi ảnh, giữ lớp nào có vật -> negative khó (vật khác có thật)
PROBE = ["person", "car", "chair", "bottle", "cup", "book", "dining table", "bowl", "potted plant", "bird", "dog",
         "boat", "umbrella", "handbag", "clock", "vase", "bench", "truck", "tv", "box"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/FSC147")
    p.add_argument("--split", default="train")
    p.add_argument("--neg", type=int, default=2, help="Số prompt lớp khác mỗi ảnh")
    p.add_argument("--conf", type=float, default=0.2, help="Ngưỡng SAM 3 (chọn trên val ở tools/eval_sam3.py)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from ultralytics.models.sam import SAM3SemanticPredictor
    pred = SAM3SemanticPredictor(overrides=dict(conf=a.conf, task="segment", mode="predict", model=DEFAULT_WEIGHTS,
                                                quantize=16, save=False, verbose=False, imgsz=1008))
    names = json.load(open(os.path.join(a.data_dir, "Train_Test_Val_FSC_147.json")))[a.split][:a.limit]
    cls = load_class_names(a.data_dir)
    vocab = sorted(set(cls.values()) | set(COCO))
    rng = random.Random(a.seed)
    keys, masks, counts = [], [], []
    t0 = time.perf_counter()
    for i, n in enumerate(names):
        rgb = np.array(Image.open(os.path.join(a.data_dir, "images_384_VarV2", n)).convert("RGB"))
        pred.set_image(rgb[..., ::-1].copy())
        prompts = [cls[n]] + rng.sample([v for v in vocab if v != cls[n]], a.neg) +             [q for q in PROBE if q != cls[n]]
        for j, prompt in enumerate(prompts):
            r = pred(text=[prompt])[0]
            if r.masks is None or len(r.masks) == 0:
                m, c = np.zeros((GRID, GRID), np.uint8), 0
            else:
                u = r.masks.data.any(dim=0).float().cpu().numpy()
                m = (cv2.resize(u, (GRID, GRID), interpolation=cv2.INTER_AREA) * 255).round().astype(np.uint8)
                c = len(r.masks)
            if j > a.neg and c == 0:
                continue  # lớp probe không có trong ảnh: bỏ (đã có a.neg negative dễ ngẫu nhiên)
            keys.append(f"{n}|{prompt}")
            masks.append(m)
            counts.append(c)
        if (i + 1) % 200 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{len(names)} ảnh | {el / (i + 1):.2f} s/ảnh | còn ~{el / (i + 1) * (len(names) - i - 1) / 60:.0f} phút",
                  flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, keys=np.array(keys), masks=np.stack(masks), counts=np.array(counts, np.int32))
    pos = [c for k, c in zip(keys, counts) if k.split("|")[1] == cls[k.split("|")[0]]]
    neg = [c for k, c in zip(keys, counts) if k.split("|")[1] != cls[k.split("|")[0]]]
    print(f"Xong {len(names)} ảnh, {len(keys)} cặp -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB) | "
          f"instance TB: đúng lớp {np.mean(pos):.1f}, lớp khác {np.mean(neg):.2f} "
          f"({np.mean([c > 0 for c in neg]) * 100:.0f}% prompt lớp khác có vật thật trong ảnh)")


if __name__ == "__main__":
    main()
