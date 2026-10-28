"""
Vẽ kết quả "detect" của DensityYOLOWorld lên ảnh:
  1. Box thô của YOLO-World (cùng lần forward) lọc bằng Density-Guided Soft-NMS (như eval --mode density-nms)
  2. Điểm vật = các đỉnh cục bộ của density cuối; lấy round(tổng density) đỉnh mạnh nhất nên số điểm = số đếm
  3. Điểm GT

    python tools/visualize_detections.py --checkpoint <ckpt> --images 6248.jpg 7426.jpg --out outputs/detections.png
"""
# Chạy được từ thư mục gốc dự án: python tools/<script>.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F

from datasets.fsc147_dataset import FSC147Dataset
from eval_fsc147 import FEATURE_STRIDE, raw_detections
from models.density_yolo import DensityYOLOWorld
from utils.custom_nms import density_guided_soft_nms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def density_points(density, count):
    """round(count) đỉnh cục bộ mạnh nhất (max-pool 3x3) của density (H, W) -> tâm ô [x, y] (pixel ảnh 640)."""
    d = density[None, None]
    peaks = torch.where(d == F.max_pool2d(d, 3, stride=1, padding=1), d, torch.zeros_like(d))[0, 0]
    k = min(int(round(count)), int((peaks > 0).sum()))
    idx = peaks.flatten().topk(k).indices
    W = density.shape[-1]
    return torch.stack([idx % W, idx // W], 1).float() * FEATURE_STRIDE + FEATURE_STRIDE / 2


def main(args):
    model = DensityYOLOWorld("yolov8s-world.pt")
    ckpt = model.load_head(args.checkpoint, map_location="cpu")
    model.to(DEVICE).eval_mode()
    print(f"arch={model.arch}, epoch {ckpt.get('epoch')}, val MAE {ckpt.get('val_mae')}")

    dataset = FSC147Dataset(args.data_dir, split=args.split)
    name_to_idx = {n: i for i, n in enumerate(dataset.image_names)}
    fig, axes = plt.subplots(len(args.images), 3, figsize=(18, 6.2 * len(args.images)), squeeze=False)
    for row, name in enumerate(args.images):
        sample = dataset[name_to_idx[name]]
        with torch.no_grad():
            dets, density = model(sample['image'].unsqueeze(0).to(DEVICE), [sample['class_name']])
        count = density.sum().item()
        boxes, scores = raw_detections(dets, args.raw_conf)
        kept = density_guided_soft_nms(boxes, scores, density, stride=FEATURE_STRIDE,
                                       score_threshold=args.score_threshold)
        boxes = boxes[kept].cpu()
        pts = density_points(density[0, 0].float(), count).cpu()
        gt = sample['points']
        img = sample['image'].permute(1, 2, 0).numpy()

        panels = [(f"Box YOLO-World + Density-Guided Soft-NMS: {len(boxes)} box", "box"),
                  (f"Điểm từ đỉnh density: {len(pts)} điểm (ĐẾM = {count:.1f})", "pred"),
                  (f"GT: {sample['count']:.0f} điểm", "gt")]
        for c, (title, kind) in enumerate(panels):
            ax = axes[row, c]
            ax.imshow(img)
            if kind == "box":
                for x1, y1, x2, y2 in boxes.tolist():
                    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec='lime', lw=1.2))
            elif kind == "pred":
                ax.scatter(pts[:, 0], pts[:, 1], s=18, c='yellow', edgecolors='black', linewidths=0.5)
            else:
                ax.scatter(gt[:, 0], gt[:, 1], s=18, c='red', edgecolors='white', linewidths=0.5)
            prefix = f"{name} \"{sample['class_name']}\"\n" if c == 0 else "\n"
            ax.set_title(prefix + title, fontsize=11)
            ax.axis('off')
        print(f"{name} ({sample['class_name']}): GT {sample['count']:.0f} | đếm (density) {count:.1f} | "
              f"box sau NMS {len(boxes)} | điểm {len(pts)}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=110)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Vẽ box / điểm dự đoán của DensityYOLOWorld lên ảnh FSC-147")
    p.add_argument("--data-dir", default="data/FSC147")
    p.add_argument("--checkpoint", default="checkpoints/density_head_best.pth")
    p.add_argument("--split", default="test")
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--raw-conf", type=float, default=0.05, help="Như eval_fsc147.py")
    p.add_argument("--score-threshold", type=float, default=0.1, help="Như eval_fsc147.py")
    p.add_argument("--out", default="outputs/detections.png")
    main(p.parse_args())
