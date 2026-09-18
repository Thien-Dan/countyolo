import torch
import os
from models.count_yolo import CountYOLO
from utils.router import DualModeRouter
import cv2
from PIL import Image
import torchvision.transforms.functional as TF

device = "cpu"

print("Đang nạp mô hình...")
countyolo = CountYOLO(use_clip=True).to(device)
state_dict = torch.load("checkpoints/checkpoint_best.pth", map_location=device)
countyolo.load_state_dict(state_dict, strict=False)
countyolo.eval()

router = DualModeRouter(size_threshold=15)

video_dir = "data/VideoCount/MOT20-Count/frames/MOT20-01-small"
frame_names = [p for p in os.listdir(video_dir) if p.endswith('.jpg') or p.endswith('.png')]
frame_names.sort()
frame_path = os.path.join(video_dir, frame_names[0])

print(f"Đang xử lý ảnh: {frame_path}")
image = Image.open(frame_path).convert("RGB")
W, H = image.size
scale = 384.0 / H
new_w_orig = int(W * scale)
image = image.resize((new_w_orig, 384), Image.BILINEAR)

W, H = image.size
new_W = ((W - 1) // 32 + 1) * 32
new_H = ((H - 1) // 32 + 1) * 32
image_padded = Image.new("RGB", (new_W, new_H))
image_padded.paste(image, (0, 0))
img_tensor = TF.to_tensor(image_padded).unsqueeze(0).to(device)

with torch.no_grad():
    # Truyền attr_texts để dùng CLIP text encoder làm attr_feats.
    # Trong training, attr_feats được tính từ exemplar_crops (CLIP image).
    # Khi test video không có exemplar annotation, dùng CLIP text làm proxy
    # (cùng joint embedding space) — tránh attr_feats = zeros gây mismatch.
    outputs = countyolo(img_tensor, texts=["objects"], attr_texts=["object"])
    density_map = outputs['density_map']
    cls_scores = outputs['cls_scores']
    box_preds = outputs['box_preds']
    
    from utils.box_utils import decode_fcos_boxes
    decoded_boxes = decode_fcos_boxes(box_preds, stride=4.0)
    # Clamp về 0 để loại bỏ box vượt biên trái/trên ảnh
    decoded_boxes = decoded_boxes.clamp(min=0)
    
    cls_probs = torch.sigmoid(cls_scores[0, 0])
    flat_probs = cls_probs.view(-1)
    
    CONF_THRESH = 0.3
    topk_vals, topk_indices = torch.topk(flat_probs, k=min(300, flat_probs.numel()))
    print(f"Max confidence: {topk_vals[0].item():.4f}")
    mask = topk_vals > CONF_THRESH
    topk_indices = topk_indices[mask]
    
    selected_boxes = decoded_boxes[0, topk_indices]
    
    points, point_labels, large_boxes, idx_p, idx_b = router.route(selected_boxes, density_map)

print(f"Số lượng points: {len(points)}")
print(f"Số lượng boxes: {len(large_boxes)}")

img_cv = cv2.imread(frame_path)
inv_scale = H / 384.0
for box in large_boxes:
    x1, y1, x2, y2 = map(float, box.cpu().numpy())
    x1, y1, x2, y2 = int(x1 * inv_scale), int(y1 * inv_scale), int(x2 * inv_scale), int(y2 * inv_scale)
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
for pt in points:
    x, y = map(float, pt.cpu().numpy())
    x, y = int(x * inv_scale), int(y * inv_scale)
    cv2.circle(img_cv, (x, y), 3, (0, 0, 255), -1)

out_path = "detection_preview.jpg"
cv2.imwrite(out_path, img_cv)
print(f"Đã lưu kết quả vào {out_path}")
