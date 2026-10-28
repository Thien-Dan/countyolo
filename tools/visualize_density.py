# Chạy được từ thư mục gốc dự án: python tools/<script>.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.density_yolo import DensityYOLOWorld
from datasets.fsc147_dataset import FSC147Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    print(f"Loading model on {DEVICE}...")
    model = DensityYOLOWorld("yolov8s-world.pt")
    ckpt = model.load_head(args.checkpoint, map_location="cpu")
    model.to(DEVICE).eval_mode()
    print(f"arch={model.arch}, epoch {ckpt.get('epoch')}, val MAE {ckpt.get('val_mae')}")

    dataset = FSC147Dataset(args.data_dir, split=args.split)
    if args.images:
        name_to_idx = {n: i for i, n in enumerate(dataset.image_names)}
        indices = [name_to_idx[n] for n in args.images]
    else:
        indices = np.random.default_rng(args.seed).choice(len(dataset), args.num, replace=False)

    # Mỗi hàng đi theo pipeline: ảnh -> Text Map -> similarity P3 / ViT -> density thô -> density cuối | GT
    cols = ["Ảnh + điểm GT (đỏ) + exemplar tự chọn (vàng)", "Text Map (YOLO-World)", "Similarity P3 (max)",
            "Similarity DINOv2 (max)", "Density thô (bước 1)", "Density cuối (Refine)", "Density GT"]
    fig, axes = plt.subplots(len(indices), len(cols), figsize=(3.4 * len(cols), 3.8 * len(indices)), squeeze=False)
    for row, idx in enumerate(indices):
        sample = dataset[int(idx)]
        with torch.no_grad():
            _, pred, aux = model(sample['image'].unsqueeze(0).to(DEVICE), [sample['class_name']], return_aux=True)
        to_np = lambda t: t[0, -1].float().cpu().numpy() if t is not None else None  # kênh cuối = max
        grid = pred.shape[-1]
        loc = aux['exemplar_locations'][0].cpu().numpy()
        ex_xy = np.stack([loc % grid, loc // grid], 1) * (640 / grid) + 640 / grid / 2
        pts = sample['points'].numpy()
        gt_map = sample['density_map'].squeeze().numpy()
        maps = [to_np(aux['text_map']), to_np(aux['sim_p3']), to_np(aux['sim_vit']),
                aux['coarse'][0, 0].float().cpu().numpy(), pred[0, 0].float().cpu().numpy(), gt_map]

        ax = axes[row, 0]
        ax.imshow(sample['image'].permute(1, 2, 0).numpy())
        ax.scatter(pts[:, 0], pts[:, 1], s=3, c='red')
        ax.scatter(ex_xy[:, 0], ex_xy[:, 1], s=120, marker='*', c='yellow', edgecolors='black')
        ax.set_title(f"{sample['image_name']} \"{sample['class_name']}\"\nGT = {sample['count']:.0f}", fontsize=10)
        for c, (m, title) in enumerate(zip(maps, cols[1:]), start=1):
            ax = axes[row, c]
            if m is None:
                ax.text(0.5, 0.5, "không có (arch không phải vit)", ha='center', va='center')
            else:
                ax.imshow(m, cmap='jet')
            extra = {4: f"\nsum = {maps[3].sum():.1f}", 5: f"\nĐẾM = {maps[4].sum():.1f}",
                     6: f"\nsum = {gt_map.sum():.1f}"}.get(c, "")
            ax.set_title(title + extra, fontsize=10)
        for ax in axes[row]:
            ax.axis('off')
        print(f"{sample['image_name']} ({sample['class_name']}): GT {sample['count']:.0f} | "
              f"density thô {maps[3].sum():.1f} | dự đoán {maps[4].sum():.1f}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=110)
    print(f"Visualization saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trực quan hóa pipeline: bản đồ trung gian + density dự đoán vs GT")
    parser.add_argument("--data-dir", default="data/FSC147")
    parser.add_argument("--checkpoint", default="checkpoints/density_head_best.pth")
    parser.add_argument("--split", default="test")
    parser.add_argument("--images", nargs="+", default=None, help="Tên ảnh cụ thể (vd 6248.jpg 7426.jpg)")
    parser.add_argument("--num", type=int, default=3, help="Số ảnh ngẫu nhiên khi không truyền --images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/density_visualization.jpg")
    main(parser.parse_args())
