import sys
import os
import json
import torch
import numpy as np
from tqdm import tqdm

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
    frame_names = [p for p in os.listdir(video_dir) if p.endswith('.jpg') or p.endswith('.png')]
    frame_names.sort()
    return [os.path.join(video_dir, p) for p in frame_names]

def extract_prompts_yolo_world(model, frame_path, text_prompt="human", conf=0.1):
    model.set_classes([text_prompt])
    results = model.predict(frame_path, conf=conf, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    return boxes

def eval_mot20():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    gt_path = "data/VideoCount/MOT20-Count/anno/MOT20-frame-level-counts-gt.json"
    if not os.path.exists(gt_path):
        print(f"Không tìm thấy file GT: {gt_path}")
        return

    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    # Khởi tạo Models
    print("Đang tải YOLO-World...")
    yolo_model = YOLOWorld("yolov8s-world.pt")
    yolo_model.to(device)
    
    if build_sam2_video_predictor is None:
        print("SAM2 chưa được cài đặt. Dừng pipeline.")
        return
        
    print("Đang tải SAM 2.1 Video Predictor...")
    sam2_checkpoint = "sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    videos = ["MOT20-01", "MOT20-02", "MOT20-05"]
    text_prompt = "person"
    
    total_abs_error = 0
    valid_frames = 0
    
    for video in videos:
        video_dir = os.path.join("data/VideoCount/MOT20-Count/frames", video)
        if not os.path.exists(video_dir):
            print(f"Không tìm thấy thư mục {video_dir}, bỏ qua.")
            continue
            
        print(f"\n--- Đang xử lý Video: {video} ---")
        frames = get_frames(video_dir)
        if not frames:
            continue
            
        mot20_gt = gt_data.get(video, {}).get("human", {})
        if not mot20_gt:
            print(f"Không tìm thấy GT cho video {video}, dùng key fallback.")
            mot20_gt = gt_data.get(video, {}) # try without "human"
            
        inference_state = predictor.init_state(video_path=video_dir)
        boxes = extract_prompts_yolo_world(yolo_model, frames[0], text_prompt, conf=0.1)
        print(f"Tìm thấy {len(boxes)} persons ở frame đầu.")
        
        obj_id_counter = 1
        for box in boxes:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id_counter,
                box=box
            )
            obj_id_counter += 1
            
        if len(boxes) > 0:
            print(f"Đang track qua {len(frames)} frames...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                frame_name = f"{out_frame_idx + 1:06d}.jpg"
                if frame_name in mot20_gt:
                    gt_count = mot20_gt[frame_name]
                    # mask.any() if pixel > 0
                    pred_count = sum(1 for i, obj_id in enumerate(out_obj_ids) if (out_mask_logits[i] > 0.0).any())
                    
                    total_abs_error += abs(pred_count - gt_count)
                    valid_frames += 1
        else:
            print("Không có box nào được tìm thấy. Bỏ qua SAM 2 Propagation.")
            
    if valid_frames > 0:
        mae = total_abs_error / valid_frames
        print(f"\n=== KẾT QUẢ BENCHMARK MOT20 ===")
        print(f"  Tổng số frames đánh giá : {valid_frames}")
        print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    else:
        print("Không có frame nào được đánh giá thành công.")

if __name__ == "__main__":
    eval_mot20()
