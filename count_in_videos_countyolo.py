import sys
sys.stdout = open("pipeline_log.txt", "w", encoding="utf-8")
sys.stderr = sys.stdout

import torch
# IMPORT CountYOLO and SAM2 FIRST to avoid DLL conflicts with cv2/pyarrow on Windows!
from models.count_yolo import CountYOLO
from utils.router import DualModeRouter
try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("Vui lòng cài đặt thư viện sam2: pip install git+https://github.com/facebookresearch/sam2.git")
    build_sam2_video_predictor = None

print("---- BẮT ĐẦU CHẠY SCRIPT ----", flush=True)

import os
import cv2
import json
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

def get_frames(video_dir):
    """Lấy danh sách các frame jpg trong thư mục, sắp xếp theo tên."""
    frame_names = [p for p in os.listdir(video_dir) if p.endswith('.jpg') or p.endswith('.png')]
    frame_names.sort()
    return [os.path.join(video_dir, p) for p in frame_names]

def extract_prompts(model, router, frame_path, text_prompt="human", device="cuda"):
    """
    Chạy CountYOLO trên frame để lấy dự đoán boxes và density map,
    sau đó dùng DualModeRouter để chuyển thành Point/Box Prompts cho SAM.
    """
    # 1. Load và preprocess image
    image = Image.open(frame_path).convert("RGB")
    W, H = image.size
    
    # Resize pad về bội số của 32 (cho CountYOLO)
    new_W = ((W - 1) // 32 + 1) * 32
    new_H = ((H - 1) // 32 + 1) * 32
    image_padded = Image.new("RGB", (new_W, new_H))
    image_padded.paste(image, (0, 0))
    
    img_tensor = TF.to_tensor(image_padded).unsqueeze(0).to(device)
    
    # 2. Forward CountYOLO
    model.eval()
    with torch.no_grad():
        # Gọi mô hình với text prompt
        outputs = model(img_tensor, texts=[text_prompt])
        
        density_map = outputs['density_map'] # (1, 1, H, W)
        cls_scores = outputs['cls_scores']   # (1, 1, H/4, W/4)
        box_preds = outputs['box_preds']     # (1, 4, H/4, W/4)
        
        from utils.box_utils import decode_fcos_boxes
        
        # 3. Trích xuất các box dự đoán
        # Giải mã box từ (l,t,r,b) sang tọa độ ảnh gốc tuyệt đối
        decoded_boxes = decode_fcos_boxes(box_preds, stride=4.0) # (1, N_preds, 4)
        
        cls_probs = torch.sigmoid(cls_scores[0, 0]) # (H/4, W/4)
        flat_probs = cls_probs.view(-1)
        
        # Áp dụng ngưỡng tự tin (confidence threshold) thay vì hardcode 5 vật thể
        CONF_THRESH = 0.3
        # Lấy tối đa 300 vật thể có xác suất cao nhất (tránh SAM 2 bị OOM), sau đó lọc qua ngưỡng
        topk_vals, topk_indices = torch.topk(flat_probs, k=min(300, flat_probs.numel()))
        
        mask = topk_vals > CONF_THRESH
        topk_indices = topk_indices[mask]
        
        # Lấy tọa độ Box đã giải mã
        selected_boxes = decoded_boxes[0, topk_indices] # [x1, y1, x2, y2] ở scale ảnh gốc
        
        # 4. Đẩy qua Router
        points, point_labels, large_boxes, idx_p, idx_b = router.route(selected_boxes, density_map)
        
    return points, point_labels, large_boxes

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # Cấu hình đường dẫn
    video_dir = "data/VideoCount/MOT20-Count/frames/MOT20-01-small"
    text_prompt = "human"
    
    if not os.path.exists(video_dir):
        print(f"Không tìm thấy thư mục frame: {video_dir}")
        return

    # 1. Khởi tạo Models
    print("Đang tải CountYOLO...")
    countyolo = CountYOLO(use_clip=True).to(device)
    
    print("Đang nạp trọng số fine-tune...")
    ckpt_path = "checkpoints/checkpoint_best.pth"
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        countyolo.load_state_dict(state_dict, strict=False)
        print(f"  -> Đã nạp thành công {ckpt_path}")
    else:
        print(f"  -> Không tìm thấy {ckpt_path}, dùng weights mặc định.")
    
    print("Đang khởi tạo Router...")
    router = DualModeRouter(size_threshold=15)
    
    if build_sam2_video_predictor is None:
        print("SAM2 chưa được cài đặt. Dừng pipeline.")
        return
        
    print("Đang tải SAM 2.1 Video Predictor...")
    # Cần file weights của sam2, giả định đã tải sam2.1_hiera_tiny.pt
    sam2_checkpoint = "sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    if not os.path.exists(sam2_checkpoint):
        print(f"Thiếu file trọng số {sam2_checkpoint}. Hãy chạy wget tải về.")
        return
        
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    # 2. Xử lý Video Pipeline
    print(f"Bắt đầu xử lý Video: {video_dir}")
    frames = get_frames(video_dir)
    if len(frames) == 0:
        print("Không có frames nào.")
        return
        
    inference_state = predictor.init_state(video_path=video_dir)
    
    # Chạy CountYOLO trên Frame 0 để lấy Prompts
    print("Trích xuất prompts từ CountYOLO (Frame 0)...")
    points, point_labels, large_boxes = extract_prompts(countyolo, router, frames[0], text_prompt, device)
    
    print(f"  -> Sinh ra {len(points)} Point Prompts và {len(large_boxes)} Box Prompts.")
    
    # 3. Add Prompts vào SAM 2
    obj_id_counter = 1
    
    # Thêm Box Prompts
    for box in large_boxes:
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id_counter,
            box=box.cpu().numpy()
        )
        obj_id_counter += 1
        
    # Thêm Point Prompts
    for i in range(len(points)):
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id_counter,
            points=points[i:i+1].cpu().numpy(),
            labels=point_labels[i:i+1].cpu().numpy()
        )
        obj_id_counter += 1
        
    # 4. Propagate (Tracking)
    print("Bắt đầu SAM 2 Video Propagation...")
    video_segments = {}  # {frame_idx: {obj_id: mask}}
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
    print(f"Hoàn thành! Đã track {obj_id_counter-1} đối tượng qua {len(video_segments)} frames.")
    
    # 5. Đánh giá Benchmark MOT20
    gt_path = "data/VideoCount/MOT20-Count/anno/MOT20-frame-level-counts-gt.json"
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
            
        mot20_01_gt = gt_data.get("MOT20-01", {}).get("human", {})
        
        total_abs_error = 0
        valid_frames = 0
        
        for frame_idx, masks_dict in video_segments.items():
            # Tên frame theo format MOT20 (ví dụ: 000001.jpg)
            frame_name = f"{frame_idx + 1:06d}.jpg"
            
            if frame_name in mot20_01_gt:
                gt_count = mot20_01_gt[frame_name]
                # Đếm số lượng object có pixel xuất hiện trong frame
                pred_count = sum(1 for mask in masks_dict.values() if mask.any())
                
                total_abs_error += abs(pred_count - gt_count)
                valid_frames += 1
                
        if valid_frames > 0:
            mae = total_abs_error / valid_frames
            print(f"\\n=== KẾT QUẢ BENCHMARK ZERO-SHOT MOT20-01 ===")
            print(f"  Frames đánh giá : {valid_frames}")
            print(f"  Mean Absolute Error (MAE): {mae:.2f}")
        else:
            print("Không tìm thấy frame khớp với Ground Truth.")

if __name__ == "__main__":
    main()
