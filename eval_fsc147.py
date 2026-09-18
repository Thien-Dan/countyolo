import sys
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse

try:
    from ultralytics import YOLOWorld
except ImportError:
    print("Vui lòng cài đặt ultralytics: pip install ultralytics")
    sys.exit(1)

def eval_fsc147(limit=None, conf=0.1, iou=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    data_dir = "data/FSC147"
    img_dir = os.path.join(data_dir, "images_384_VarV2")
    split_path = os.path.join(data_dir, "Train_Test_Val_FSC_147.json")
    class_path = os.path.join(data_dir, "ImageClasses_FSC147.txt")
    anno_path = os.path.join(data_dir, "annotation_FSC147_384.json")
    
    if not all(os.path.exists(p) for p in [split_path, class_path, anno_path, img_dir]):
        print("Không tìm thấy đủ file dữ liệu FSC-147.")
        return

    # Load splits
    with open(split_path, 'r') as f:
        splits = json.load(f)
    test_images = splits.get("test", [])
    
    if limit:
        test_images = test_images[:limit]
        print(f"Chỉ chạy đánh giá trên {limit} ảnh đầu tiên.")
    else:
        print(f"Chạy đánh giá trên toàn bộ {len(test_images)} ảnh tập Test.")

    # Load classes
    img_to_class = {}
    with open(class_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_to_class[parts[0]] = parts[1]

    # Load annotations
    with open(anno_path, 'r') as f:
        annos = json.load(f)

    # Khởi tạo Models
    print("Đang tải YOLO-World...")
    yolo_model = YOLOWorld("yolov8s-world.pt")
    yolo_model.to(device)
    
    total_abs_error = 0
    valid_images = 0
    
    # Evaluate
    print("Bắt đầu đánh giá Zero-Shot YOLO-World...")
    for img_name in tqdm(test_images):
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue
            
        class_name = img_to_class.get(img_name, "object")
        gt_points = annos.get(img_name, {}).get("points", [])
        gt_count = len(gt_points)
        
        # YOLO-World prediction
        yolo_model.set_classes([class_name])
        results = yolo_model.predict(img_path, conf=conf, iou=iou, verbose=False)
        pred_count = len(results[0].boxes)
        
        total_abs_error += abs(pred_count - gt_count)
        valid_images += 1
        
    if valid_images > 0:
        mae = total_abs_error / valid_images
        print(f"\n=== KẾT QUẢ BENCHMARK ZERO-SHOT FSC-147 ===")
        print(f"  Tổng số ảnh đánh giá : {valid_images}")
        print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    else:
        print("Không có ảnh nào được đánh giá thành công.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO-World on FSC-147")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to evaluate")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    args = parser.parse_args()
    
    eval_fsc147(limit=args.limit, conf=args.conf, iou=args.iou)
