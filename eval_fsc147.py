import sys
import os
import csv
import torch
from tqdm import tqdm
import argparse
from typing import Optional

import numpy as np
from PIL import Image

try:
    from ultralytics import YOLOWorld
except ImportError:
    print("Vui lòng cài đặt ultralytics: pip install ultralytics")
    sys.exit(1)

from datasets.fsc147_dataset import FSC147Dataset
from utils.scale_zoom import canonical_scale_count, density_tiled_count, weighted_log_size

CHECKPOINT_DIR: str = "checkpoints"
FEATURE_STRIDE: float = 8.0  # P3


def load_density_model(device: torch.device, checkpoint_path: str) -> Optional[object]:
    """Tải DensityYOLOWorld + trọng số Density Head đã train."""
    try:
        from models.density_yolo import DensityYOLOWorld
        wrapper = DensityYOLOWorld("yolov8s-world.pt")
        ckpt = wrapper.load_head(checkpoint_path, map_location="cpu")
        wrapper.to(device)
        wrapper.eval_mode()
        print(f"  Đã tải Density Head từ: {checkpoint_path} (epoch {ckpt.get('epoch')}, "
              f"val MAE {ckpt.get('val_mae')})")
        return wrapper
    except Exception as e:
        print(f"  [LỖI] Không tải được Density Head: {e}")
        return None


def raw_detections(dets: torch.Tensor, raw_conf: float, max_det: int = 3000):
    """
    dets: (1, 4 + 1, N) đầu ra đã decode của YOLO-World (xywh, cùng hệ tọa độ ảnh 640x640 đưa vào model).
    Trả về boxes xyxy (M, 4) và scores (M,) trước NMS.
    """
    d = dets[0].float()
    xywh, scores = d[:4].T, d[4]
    keep = scores >= raw_conf
    xywh, scores = xywh[keep], scores[keep]
    if scores.numel() > max_det:
        scores, idx = scores.topk(max_det)
        xywh = xywh[idx]
    boxes = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], dim=1)
    return boxes, scores


def eval_fsc147(
    split: str = "test",
    limit: Optional[int] = None,
    conf: float = 0.1,
    iou: float = 0.7,
    mode: str = "zero-shot",
    checkpoint: str = "density_head_best.pth",
    raw_conf: float = 0.05,
    score_threshold: float = 0.1,
    data_dir: str = "data/FSC147",
    exemplars: str = "self",
    zoom: float = 0.0,
    zoom_max: int = 6,
    dump: Optional[str] = None,
    ttn_thr: float = 0.0,
    ttn_grid: int = 2,
    fast: bool = False,
    tta: bool = False,
) -> Optional[dict]:
    """
    Đánh giá đếm trên FSC-147.

    mode:
        zero-shot     : YOLO-World + NMS mặc định (baseline, ảnh gốc qua letterbox của Ultralytics)
        density-count : SUM(density map)
        density-nms   : Density-Guided Soft-NMS trên box thô của cùng một lần forward
        soft-nms      : Soft-NMS thường trên cùng box thô (baseline ablation cho density-nms)

    exemplars (Self-Exemplar): self = đỉnh density thô (zero-shot), gt = tâm 3 exemplar box của FSC-147 (few-shot)
    zoom: kích thước vật chuẩn (px) cho Canonical-Scale Zoom ở density-count, 0 = tắt (cần Scale Head)
    ttn_thr / ttn_grid: test-time tiling theo số đếm ở density-count — ảnh có count >= ttn_thr được chia
        ttn_grid x ttn_grid ô và đếm lại; 0 = tắt
    Trả về {"mae", "rmse", "n"} (None nếu không chạy được).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    dataset = FSC147Dataset(data_dir, split=split)
    n_images = min(limit, len(dataset)) if limit else len(dataset)
    print(f"Chế độ: {mode} | split={split} | {n_images} ảnh")

    if mode == "zero-shot":
        yolo_model = YOLOWorld("yolov8s-world.pt")
        yolo_model.to(device)
    else:
        from utils.custom_nms import density_guided_soft_nms
        ckpt_path = checkpoint if os.path.exists(checkpoint) else os.path.join(CHECKPOINT_DIR, checkpoint)
        wrapper = load_density_model(device, ckpt_path)
        if wrapper is not None and fast:
            # Suy luận nhanh: DINOv2 fp16 + torch.compile (T4: ~43 ms -> ~23 ms / ảnh)
            wrapper.optimize_inference(half_vit=True, compile_vit=True)
        if wrapper is None:
            return
        if zoom and wrapper.scale_head is None:
            print("  [CẢNH BÁO] Checkpoint không có Scale Head -> tắt Canonical-Scale Zoom")
            zoom = 0.0
        if zoom and exemplars != "self":
            print("  [CẢNH BÁO] Canonical-Scale Zoom chỉ hỗ trợ exemplars=self -> tắt zoom")
            zoom = 0.0
        if ttn_thr and (zoom or exemplars != "self"):
            print("  [CẢNH BÁO] Test-time tiling chỉ dùng với exemplars=self và không zoom -> tắt")
            ttn_thr = 0.0

    abs_err, sq_err, coarse_err = 0.0, 0.0, 0.0
    errors, gt_sizes, est_sizes, n_zoomed = [], [], [], 0
    rows = []  # --dump: kết quả từng ảnh
    for i in tqdm(range(n_images)):
        sample = dataset[i]
        gt_count = sample['count']
        class_name = sample['class_name']

        if mode == "zero-shot":
            yolo_model.set_classes([class_name])
            img_path = os.path.join(dataset.img_dir, sample['image_name'])
            results = yolo_model.predict(img_path, conf=conf, iou=iou, max_det=3000, verbose=False)
            pred_count = float(len(results[0].boxes))
        else:
            img = sample['image'].unsqueeze(0).to(device)
            ex_points = None
            if exemplars == "gt":
                b = sample['boxes']
                ex_points = [(b[:, :2] + b[:, 2:]) / 2]
            with torch.no_grad():
                dets, density_map, aux = wrapper(img, [class_name], exemplar_points=ex_points, return_aux=True)
                if tta and ex_points is None:
                    # TTA lật ngang: trung bình density của ảnh gốc và ảnh lật (lật lại về đúng vị trí)
                    _, dflip = wrapper(img.flip(-1), [class_name])
                    density_map = (density_map + dflip.flip(-1)) / 2
            coarse_err += abs(aux['coarse'].sum().item() - gt_count)
            gt_sizes.append(sample['size_map'][0, 0, 0].item())
            if aux['log_size'] is not None:
                est_sizes.append(weighted_log_size(aux['log_size'][0, 0].float(), density_map[0, 0].float(),
                                                   fallback=float('nan')))
            if mode == "density-count":
                pred_count = density_map.sum().item()
                if zoom:
                    image = Image.open(os.path.join(dataset.img_dir, sample['image_name'])).convert("RGB")
                    with torch.no_grad():
                        pred_count, info = canonical_scale_count(wrapper, image, class_name, device,
                                                                 target=zoom, max_zoom=zoom_max)
                    n_zoomed += info["zoomed"] > 0
                elif ttn_thr:
                    image = Image.open(os.path.join(dataset.img_dir, sample['image_name'])).convert("RGB")
                    with torch.no_grad():
                        pred_count, info = density_tiled_count(wrapper, image, class_name, device,
                                                               count_thr=ttn_thr, grid=ttn_grid)
                    n_zoomed += info["tiled"]
            else:
                boxes, scores = raw_detections(dets, raw_conf)
                kept = density_guided_soft_nms(
                    boxes, scores, density_map if mode == "density-nms" else None,
                    stride=FEATURE_STRIDE, score_threshold=score_threshold,
                )
                pred_count = float(len(kept))

        if dump and mode != "zero-shot":
            rows.append({"image": sample['image_name'], "class": class_name, "gt": gt_count,
                         "pred": round(pred_count, 2), "coarse": round(aux['coarse'].sum().item(), 2),
                         "size_px": round(2 ** gt_sizes[-1], 1),
                         "max_cell": round(density_map.max().item(), 4)})
        abs_err += abs(pred_count - gt_count)
        sq_err += (pred_count - gt_count) ** 2
        errors.append(pred_count - gt_count)

    if rows:
        with open(dump, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"  Đã ghi kết quả từng ảnh: {dump}")

    if n_images > 0:
        tag = f"{mode}, {split}" + (f", exemplars={exemplars}" if mode != "zero-shot" else "")
        if zoom:
            tag += f", zoom={zoom:g}px"
        if ttn_thr:
            tag += f", ttn>={ttn_thr:g} grid={ttn_grid}"
        if fast:
            tag += ", fast (fp16 + compile)"
        if tta:
            tag += ", TTA lật ngang"
        print(f"\n=== KẾT QUẢ BENCHMARK FSC-147 [{tag}] ===")
        print(f"  Tổng số ảnh đánh giá : {n_images}")
        print(f"  MAE : {abs_err / n_images:.2f}")
        print(f"  RMSE: {(sq_err / n_images) ** 0.5:.2f}")
        if mode != "zero-shot":
            e, g = np.array(errors), 2 ** np.array(gt_sizes)
            print(f"  Bias: {e.mean():+.2f}")
            for lo, hi in [(0, 24), (24, 40), (40, float('inf'))]:
                m = (g >= lo) & (g < hi)
                if m.any():
                    print(f"  Vật {lo:g}-{hi:g}px (n={m.sum()}): MAE {np.abs(e[m]).mean():.2f} bias {e[m].mean():+.2f}")
        if mode == "density-count":
            print(f"  MAE density thô (bước 1): {coarse_err / n_images:.2f}")
        if est_sizes:
            est, gl = np.array(est_sizes), np.array(gt_sizes)
            ok = ~np.isnan(est)
            rank = lambda a: np.argsort(np.argsort(a))
            print(f"  Scale Head: |log2| trung vị {np.median(np.abs(est[ok] - gl[ok])):.3f} | "
                  f"Spearman {np.corrcoef(rank(est[ok]), rank(gl[ok]))[0, 1]:.3f}")
        if zoom or ttn_thr:
            print(f"  Ảnh được zoom/chia ô: {n_zoomed}/{n_images}")
        return {"mae": abs_err / n_images, "rmse": (sq_err / n_images) ** 0.5, "n": n_images}
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate counting on FSC-147")
    parser.add_argument("--mode", choices=["zero-shot", "density-count", "density-nms", "soft-nms"],
                        default="zero-shot")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số ảnh đánh giá")
    parser.add_argument("--data-dir", default="data/FSC147")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold (zero-shot)")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (zero-shot)")
    parser.add_argument("--raw-conf", type=float, default=0.05, help="Ngưỡng lọc box thô trước Soft-NMS")
    parser.add_argument("--score-threshold", type=float, default=0.1,
                        help="Ngưỡng score sau suy giảm để một box được tính (soft-nms/density-nms)")
    parser.add_argument("--checkpoint", type=str, default="density_head_best.pth",
                        help="Đường dẫn checkpoint hoặc tên file trong checkpoints/")
    parser.add_argument("--exemplars", choices=["self", "gt"], default="self",
                        help="Vị trí exemplar cho Self-Exemplar: đỉnh density thô (zero-shot) hoặc exemplar box GT")
    parser.add_argument("--zoom", type=float, default=0.0,
                        help="Canonical-Scale Zoom: kích thước vật chuẩn (px), 0 = tắt")
    parser.add_argument("--zoom-max", type=int, default=6, help="Số ô tối đa mỗi chiều khi zoom")
    parser.add_argument("--ttn-thr", type=float, default=0.0,
                        help="Test-time tiling: ảnh có count >= ngưỡng này được chia ô và đếm lại (0 = tắt)")
    parser.add_argument("--ttn-grid", type=int, default=2, help="Số ô mỗi chiều khi chia")
    parser.add_argument("--fast", action="store_true", help="DINOv2 fp16 + torch.compile (chỉ suy luận)")
    parser.add_argument("--tta", action="store_true", help="TTA lật ngang (x2 thời gian)")
    parser.add_argument("--dump", default=None, help="Ghi CSV kết quả từng ảnh (gt, pred, kích thước vật...)")
    args = parser.parse_args()

    eval_fsc147(
        split=args.split,
        limit=args.limit,
        conf=args.conf,
        iou=args.iou,
        mode=args.mode,
        checkpoint=args.checkpoint,
        raw_conf=args.raw_conf,
        score_threshold=args.score_threshold,
        data_dir=args.data_dir,
        exemplars=args.exemplars,
        zoom=args.zoom,
        zoom_max=args.zoom_max,
        dump=args.dump,
        ttn_thr=args.ttn_thr,
        ttn_grid=args.ttn_grid,
        fast=args.fast,
        tta=args.tta,
    )
