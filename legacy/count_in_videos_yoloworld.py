import sys
import os
import json
import torch
import numpy as np

# Set stdout to log file
sys.stdout = open("pipeline_log.txt", "w", encoding="utf-8")
sys.stderr = sys.stdout

print("---- BẮT ĐẦU CHẠY SCRIPT YOLO-WORLD + SAM 2 ----", flush=True)

try:
    from ultralytics import YOLOWorld
except ImportError:
    print("Vui lòng cài đặt ultralytics: pip install ultralytics")
    sys.exit(1)

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("Vui lòng cài đặt thư viện sam2: pip install git+https://github.com/facebookresearch/sam2.git")
    build_sam2_video_predictor = None

def get_frames(video_dir):
    """Lấy danh sách các frame jpg trong thư mục, sắp xếp theo tên."""
    frame_names = [p for p in os.listdir(video_dir) if p.endswith('.jpg') or p.endswith('.png')]
    frame_names.sort()
    return [os.path.join(video_dir, p) for p in frame_names]

def extract_prompts_yolo_world(model, frame_path, text_prompt="human", conf=0.1):
    """
    Chạy YOLO-World trên frame để lấy dự đoán boxes.
    """
    print(f"Detecting '{text_prompt}' on {frame_path}...")
    # Thiết lập class cho YOLO-World
    model.set_classes([text_prompt])
    
    # Dự đoán
    results = model.predict(frame_path, conf=conf, verbose=False)
    
    # Trích xuất bounding boxes (N, 4) định dạng xyxy
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    print(f"  -> Found {len(boxes)} boxes.")
    return boxes

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # Cấu hình đường dẫn
    video_dir = "data/VideoCount/MOT20-Count/frames/MOT20-01-small"
    text_prompt = "person"
    
    if not os.path.exists(video_dir):
        print(f"Không tìm thấy thư mục frame: {video_dir}")
        return

    # 1. Khởi tạo Models
    print("Đang tải YOLO-World...")
    yolo_model = YOLOWorld("yolov8s-world.pt")
    yolo_model.to(device)
    
    if build_sam2_video_predictor is None:
        print("SAM2 chưa được cài đặt. Dừng pipeline.")
        return
        
    print("Đang tải SAM 2.1 Video Predictor...")
    sam2_checkpoint = "sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    if not os.path.exists(sam2_checkpoint):
        print(f"Thiếu file trọng số {sam2_checkpoint}. Hãy chạy tải về.")
        return
        
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    # 2. Xử lý Video Pipeline
    print(f"Bắt đầu xử lý Video: {video_dir}")
    frames = get_frames(video_dir)
    if len(frames) == 0:
        print("Không có frames nào.")
        return
        
    inference_state = predictor.init_state(video_path=video_dir)
    
    # Chạy YOLO-World trên Frame 0 để lấy Bounding Box Prompts
    print("Trích xuất prompts từ YOLO-World (Frame 0)...")
    boxes = extract_prompts_yolo_world(yolo_model, frames[0], text_prompt, conf=0.1)
    
    # 3. Add Prompts vào SAM 2
    obj_id_counter = 1
    
    # Thêm Box Prompts vào SAM 2
    for box in boxes:
        # box là [x1, y1, x2, y2]
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id_counter,
            box=box
        )
        obj_id_counter += 1
        
    # 4. Propagate (Tracking)
    print("Bắt đầu SAM 2 Video Propagation...")
    video_segments = {}  # {frame_idx: {obj_id: mask}}
    
    if len(boxes) > 0:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    else:
        print("Không có box nào được tìm thấy. Bỏ qua SAM 2 Propagation.")
        
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
