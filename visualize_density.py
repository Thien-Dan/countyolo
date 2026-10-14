import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.density_yolo import DensityYOLOWorld
from datasets.fsc147_dataset import FSC147Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    print(f"Loading model on {DEVICE}...")
    model = DensityYOLOWorld("yolov8s-world.pt")
    model.load_head(args.checkpoint, map_location="cpu")
    model.to(DEVICE).eval_mode()

    dataset = FSC147Dataset(args.data_dir, split=args.split)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), args.num, replace=False)

    fig, axes = plt.subplots(args.num, 3, figsize=(15, 5 * args.num), squeeze=False)
    for row, idx in enumerate(indices):
        sample = dataset[int(idx)]
        with torch.no_grad():
            _, pred_density = model(sample['image'].unsqueeze(0).to(DEVICE), [sample['class_name']])
        pred_map = pred_density.squeeze().cpu().numpy()
        gt_map = sample['density_map'].squeeze().numpy()
        pts = sample['points'].numpy()

        axes[row, 0].imshow(sample['image'].permute(1, 2, 0).numpy())
        axes[row, 0].scatter(pts[:, 0], pts[:, 1], s=4, c='red')
        axes[row, 0].set_title(f"{sample['image_name']} — \"{sample['class_name']}\"\nGT Count: {sample['count']:.0f}")
        axes[row, 1].imshow(gt_map, cmap='jet')
        axes[row, 1].set_title(f"GT Density (80x80)\nSum: {gt_map.sum():.1f}")
        axes[row, 2].imshow(pred_map, cmap='jet')
        axes[row, 2].set_title(f"Pred Density (80x80)\nPred Count: {pred_map.sum():.1f}")
        for ax in axes[row]:
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out)
    print(f"Visualization saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trực quan hóa density map dự đoán vs GT")
    parser.add_argument("--data-dir", default="data/FSC147")
    parser.add_argument("--checkpoint", default="checkpoints/density_head_best.pth")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/density_visualization.jpg")
    main(parser.parse_args())
