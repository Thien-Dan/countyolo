import sys
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, List, Optional

try:
    from ultralytics import YOLOWorld
except ImportError:
    print("Vui lòng cài đặt ultralytics: pip install ultralytics")
    sys.exit(1)

CHECKPOINT_DIR: str = "checkpoints"


def load_density_head(device: torch.device, checkpoint_path: str) -> Optional[object]:
    """
    Tải Density Head đã được fine-tune từ file checkpoint.

    Args:
        device: Thiết bị chạy mô hình (cuda/cpu)
        checkpoint_path: Đường dẫn file .pth

    Returns:
        DensityYOLOWorld wrapper đã tải trọng số, hoặc None nếu có lỗi
    """
    try:
        from models.density_yolo import DensityYOLOWorld
        wrapper = DensityYOLOWorld("yolov8s-world.pt")
        # Load checkpoint trước khi move sang device
        wrapper.density_head.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        # Move toàn bộ model (cả YOLO backbone + CLIP text encoder + Density Head) sang cùng device
        wrapper.yolo.to(device)
        wrapper.density_head.to(device)
        wrapper.eval_mode()
        print(f"  Đã tải Density Head từ: {checkpoint_path}")
        return wrapper
    except Exception as e:
        print(f"  [LỖI] Không tải được Density Head: {e}")
        return None


def predict_with_density_count(
    wrapper,
    img_path: str,
    class_name: str,
    device: torch.device,
) -> float:
    """
    Dự đoán số lượng vật thể bằng cách lấy tổng (sum) của Density Map.
    Đây là nguyên lý cốt lõi của Density Estimation Counting.
    Không cần NMS, không cần Bounding Box.

    Args:
        wrapper: Model DensityYOLOWorld đã tải checkpoint
        img_path: Đường dẫn ảnh đầu vào
        class_name: Tên class (dùng để set_classes cho YOLO)
        device: Thiết bị tính toán

    Returns:
        Số lượng vật thể dự đoán dưới dạng float
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    image = Image.open(img_path).convert("RGB").resize((640, 640))
    img_tensor: torch.Tensor = TF.to_tensor(image).unsqueeze(0).to(device)  # (1,3,640,640)

    wrapper.yolo.set_classes([class_name])
    with torch.no_grad():
        _, density_map = wrapper(img_tensor)  # density_map: (1, 1, 40, 40)

    # Tổng của Density Map = số lượng vật thể ước lượng
    pred_count: float = density_map.sum().item()
    return pred_count


def predict_with_density_nms(
    wrapper,
    img_path: str,
    class_name: str,
    device: torch.device,
    raw_conf: float = 0.01,
    score_threshold: float = 0.01,
) -> int:
    """
    Chạy inference với Density-Guided Soft-NMS.

    Args:
        wrapper: Model DensityYOLOWorld đã tải checkpoint
        img_path: Đường dẫn ảnh đầu vào
        class_name: Tên class (dùng làm text prompt cho YOLO-World)
        device: Thiết bị tính toán
        raw_conf: Ngưỡng confidence thấp (lấy RAW detections trước khi custom NMS)
        score_threshold: Ngưỡng score tối thiểu trong Soft-NMS

    Returns:
        Số lượng vật thể dự đoán (int)
    """
    from utils.custom_nms import density_guided_soft_nms
    from PIL import Image
    import torchvision.transforms.functional as TF

    # 1. Chuẩn bị tensor ảnh resize về 640x640
    image = Image.open(img_path).convert("RGB").resize((640, 640))
    img_tensor: torch.Tensor = TF.to_tensor(image).unsqueeze(0).to(device)  # (1,3,640,640)

    # 2. Đặt text prompt -> set_classes sẽ dùng CLIP text encoder (phải cùng device)
    wrapper.yolo.set_classes([class_name])
    with torch.no_grad():
        # Lấy raw boxes với conf cực thấp + iou=0.99 để bypass NMS của YOLO
        raw_results = wrapper.yolo.predict(
            img_path, conf=raw_conf, iou=0.99, max_det=3000,
            device=device, verbose=False
        )
        # Forward qua wrapper để kích hoạt hook -> lấy density_map
        _, density_map = wrapper(img_tensor)  # density_map: (1, 1, 40, 40)

    # Boxes và scores đã ở đúng device vì predict được chỉ định device=device
    boxes_xyxy: torch.Tensor = raw_results[0].boxes.xyxy  # (N, 4)
    scores: torch.Tensor = raw_results[0].boxes.conf       # (N,)

    if boxes_xyxy.numel() == 0:
        return 0

    # 3. Chạy Density-Guided Soft-NMS
    kept_indices = density_guided_soft_nms(
        boxes_xyxy, scores, density_map,
        stride=16.0,
        score_threshold=score_threshold,
    )
    return len(kept_indices)


def eval_fsc147(
    limit: Optional[int] = None,
    conf: float = 0.1,
    iou: float = 0.7,
    use_density_nms: bool = False,
    use_density_count: bool = False,
    checkpoint: str = "density_head_best.pth",
) -> None:
    """
    Đánh giá hiệu năng trên tập test của FSC-147.

    Args:
        limit: Số ảnh tối đa cần đánh giá (None = toàn bộ)
        conf: Confidence threshold cho chế độ Zero-Shot
        iou: IoU threshold cho NMS ở chế độ Zero-Shot
        use_density_nms: Nếu True, dùng Density-Guided Soft-NMS thay vì NMS mặc định
        use_density_count: Nếu True, dùng SUM(Density Map) để đếm trực tiếp (không cần NMS)
        checkpoint: Tên file checkpoint (bên trong thư mục checkpoints/)
    """
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    data_dir: str = "data/FSC147"
    img_dir: str = os.path.join(data_dir, "images_384_VarV2")
    split_path: str = os.path.join(data_dir, "Train_Test_Val_FSC_147.json")
    class_path: str = os.path.join(data_dir, "ImageClasses_FSC147.txt")
    anno_path: str = os.path.join(data_dir, "annotation_FSC147_384.json")

    if not all(os.path.exists(p) for p in [split_path, class_path, anno_path, img_dir]):
        print("Không tìm thấy đủ file dữ liệu FSC-147.")
        return

    # Load dữ liệu metadata
    with open(split_path, 'r') as f:
        test_images: List[str] = json.load(f).get("test", [])

    if limit:
        test_images = test_images[:limit]
        print(f"Chỉ chạy đánh giá trên {limit} ảnh đầu tiên.")
    else:
        print(f"Chạy đánh giá trên toàn bộ {len(test_images)} ảnh tập Test.")

    img_to_class: Dict[str, str] = {}
    with open(class_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_to_class[parts[0]] = parts[1]

    with open(anno_path, 'r') as f:
        annos: Dict = json.load(f)

    total_abs_error: float = 0.0
    valid_images: int = 0

    # ---- Chế độ 0: Density Sum Count (dùng tổng Density Map để đếm trực tiếp) ----
    if use_density_count:
        ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint)
        if not os.path.exists(ckpt_path):
            print(f"[LỖI] Không tìm thấy checkpoint: {ckpt_path}")
            return

        print(f"\nChế độ: Density Sum Count (SUM của Heatmap, checkpoint: {ckpt_path})")
        wrapper = load_density_head(device, ckpt_path)
        if wrapper is None:
            return

        for img_name in tqdm(test_images):
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            class_name = img_to_class.get(img_name, "object")
            gt_count: int = len(annos.get(img_name, {}).get("points", []))
            pred_count_f: float = predict_with_density_count(wrapper, img_path, class_name, device)

            total_abs_error += abs(pred_count_f - gt_count)
            valid_images += 1

    # ---- Chế độ 1: Density-Guided NMS (dùng Density Head đã train) ----
    elif use_density_nms:
        ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint)
        if not os.path.exists(ckpt_path):
            print(f"[LỖI] Không tìm thấy checkpoint: {ckpt_path}")
            return

        print(f"\nChế độ: Density-Guided Soft-NMS (checkpoint: {ckpt_path})")
        wrapper = load_density_head(device, ckpt_path)
        if wrapper is None:
            return

        for img_name in tqdm(test_images):
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            class_name = img_to_class.get(img_name, "object")
            gt_count: int = len(annos.get(img_name, {}).get("points", []))
            pred_count: int = predict_with_density_nms(wrapper, img_path, class_name, device)

            total_abs_error += abs(pred_count - gt_count)
            valid_images += 1

    # ---- Chế độ 2: Zero-Shot YOLO-World (NMS mặc định của Ultralytics) ----
    else:
        print("\nChế độ: Zero-Shot YOLO-World (NMS mặc định)")
        yolo_model = YOLOWorld("yolov8s-world.pt")
        yolo_model.to(device)

        for img_name in tqdm(test_images):
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            class_name = img_to_class.get(img_name, "object")
            gt_count = len(annos.get(img_name, {}).get("points", []))

            yolo_model.set_classes([class_name])
            results = yolo_model.predict(img_path, conf=conf, iou=iou, verbose=False)
            pred_count = len(results[0].boxes)

            total_abs_error += abs(pred_count - gt_count)
            valid_images += 1

    # ---- In kết quả ----
    if valid_images > 0:
        mae: float = total_abs_error / valid_images
        if use_density_count:
            mode_str = "Density Sum Count"
        elif use_density_nms:
            mode_str = "Density-Guided NMS"
        else:
            mode_str = "Zero-Shot"
        print(f"\n=== KẾT QUẢ BENCHMARK FSC-147 [{mode_str}] ===")
        print(f"  Tổng số ảnh đánh giá : {valid_images}")
        print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    else:
        print("Không có ảnh nào được đánh giá thành công.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO-World on FSC-147")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số ảnh đánh giá")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold (Zero-Shot mode)")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (Zero-Shot mode)")
    parser.add_argument("--density-nms", action="store_true", help="Dùng Density-Guided Soft-NMS")
    parser.add_argument("--density-count", action="store_true",
                        help="Dùng SUM(Density Map) để đếm trực tiếp (không NMS)")
    parser.add_argument("--checkpoint", type=str, default="density_head_best.pth",
                        help="Tên file checkpoint trong thư mục checkpoints/")
    args = parser.parse_args()

    eval_fsc147(
        limit=args.limit,
        conf=args.conf,
        iou=args.iou,
        use_density_nms=args.density_nms,
        use_density_count=args.density_count,
        checkpoint=args.checkpoint,
    )
