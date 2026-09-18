import torch

def box_iou(boxes1, boxes2):
    """Tính IoU giữa 2 tập hợp bounding boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

def density_guided_soft_nms(boxes, scores, density_map, stride=16.0, score_threshold=0.01):
    """
    Density-Guided Soft-NMS.
    - boxes: (N, 4) định dạng [x1, y1, x2, y2] tuyệt đối trên ảnh gốc.
    - scores: (N, 1) tự tin của box.
    - density_map: (1, 1, H_feat, W_feat) lấy từ Auxiliary Density Head.
    - stride: tỷ lệ thu nhỏ của feature map (P4 có stride=16).
    
    Thuật toán sẽ tự động điều chỉnh sigma theo giá trị mật độ (density).
    Nếu vùng đông đúc -> sigma lớn -> penalty thấp -> giữ lại nhiều box.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        
    N = boxes.shape[0]
    
    # 1. Trích xuất giá trị density tại tâm của từng box
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    
    idx_x = (centers_x / stride).long().clamp(min=0, max=density_map.shape[3] - 1)
    idx_y = (centers_y / stride).long().clamp(min=0, max=density_map.shape[2] - 1)
    
    # Lấy density cho từng box (N,)
    box_densities = density_map[0, 0, idx_y, idx_x]
    
    # Chuẩn hóa density về khoảng (0, 1) để làm weight điều chỉnh
    # Ta có thể dùng hàm sigmoid hoặc kẹp giá trị
    norm_densities = torch.sigmoid(box_densities - 1.0) # Vùng đông -> tiến gần 1, thưa -> tiến gần 0.2
    
    # Các tham số Soft-NMS cơ bản
    base_sigma = 0.5 
    
    # Cấu trúc lưu vết
    kept_indices = []
    
    # Sao chép scores để không ảnh hưởng dữ liệu gốc
    scores_copy = scores.clone()
    
    for i in range(N):
        # Lấy box có score cao nhất hiện tại
        max_idx = torch.argmax(scores_copy)
        max_score = scores_copy[max_idx]
        
        if max_score < score_threshold:
            break
            
        kept_indices.append(max_idx.item())
        
        # Đánh dấu box này đã xử lý bằng cách set score = 0
        scores_copy[max_idx] = 0.0
        
        # Tính IoU của box lớn nhất với tất cả các box còn lại
        ious = box_iou(boxes[max_idx:max_idx+1], boxes)[0]
        
        # ĐIỀU CHỈNH ADAPTIVE THEO DENSITY
        # Nếu box đang xét nằm trong vùng đông đúc (norm_density cao),
        # ta TĂNG sigma lên => Hình phạt giảm => Giữ lại nhiều box xung quanh
        current_density = norm_densities[max_idx]
        
        # Công thức điều chỉnh sigma: sigma_thực_tế = base_sigma * (1 + current_density * 2)
        # Tức là vùng cực đông đúc sigma có thể gấp 3 lần vùng bình thường
        adaptive_sigma = base_sigma * (1.0 + current_density * 2.0)
        
        # Áp dụng Gaussian Penalty của Soft-NMS
        penalty = torch.exp(-(ious ** 2) / adaptive_sigma)
        
        # Suy giảm điểm số các box khác
        scores_copy = scores_copy * penalty
        
    return torch.tensor(kept_indices, dtype=torch.int64, device=boxes.device)
