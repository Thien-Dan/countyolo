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
    iou = inter / union.clamp(min=1e-6)
    return iou

def box_density_mass(density_map, boxes, stride=8.0):
    """
    Tổng density (≈ số vật thể) nằm trong mỗi box, tính bằng integral image.
    - density_map: (1, 1, H_feat, W_feat), đơn vị số lượng.
    - boxes: (N, 4) [x1, y1, x2, y2] trong CÙNG hệ tọa độ ảnh đã đưa vào model tạo density.
    """
    d = density_map[0, 0].float()
    H, W = d.shape
    integral = torch.zeros((H + 1, W + 1), dtype=d.dtype, device=d.device)
    integral[1:, 1:] = d.cumsum(0).cumsum(1)

    # Làm tròn về ô lưới gần nhất; box nhỏ hơn 1 ô vẫn phủ ít nhất 1 ô
    x1 = (boxes[:, 0] / stride).round().long().clamp(0, W - 1)
    y1 = (boxes[:, 1] / stride).round().long().clamp(0, H - 1)
    x2 = torch.max((boxes[:, 2] / stride).round().long(), x1 + 1).clamp(max=W)
    y2 = torch.max((boxes[:, 3] / stride).round().long(), y1 + 1).clamp(max=H)
    return integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]

def density_guided_soft_nms(boxes, scores, density_map, stride=8.0, score_threshold=0.01):
    """
    Density-Guided Soft-NMS.
    - boxes: (N, 4) định dạng [x1, y1, x2, y2], cùng hệ tọa độ với ảnh đưa vào Density Head.
    - scores: (N,) độ tự tin của box.
    - density_map: (1, 1, H_feat, W_feat) lấy từ Density Head (đơn vị số lượng).
      None -> Soft-NMS thường (sigma cố định), dùng làm baseline ablation.
    - stride: tỷ lệ thu nhỏ của feature map (P3 có stride=8).

    Độ đông cục bộ = số vật thể (theo density) trong vùng lân cận box (box nở rộng gấp đôi) trừ đi chính nó.
    Vùng đông -> sigma lớn -> penalty thấp -> giữ lại nhiều box.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    N = boxes.shape[0]
    scores = scores.reshape(-1)

    # 1. Độ đông cục bộ quanh mỗi box
    if density_map is None:
        norm_densities = torch.zeros(N, device=boxes.device)
    else:
        wh = boxes[:, 2:] - boxes[:, :2]
        expanded = torch.cat([boxes[:, :2] - wh / 2, boxes[:, 2:] + wh / 2], dim=1)
        neighbours = (box_density_mass(density_map, expanded, stride) - 1.0).clamp(min=0)
        # 0 hàng xóm -> 0 (thưa); nhiều hàng xóm -> tiến về 1 (đông)
        norm_densities = 1.0 - torch.exp(-neighbours)

    # Các tham số Soft-NMS cơ bản
    base_sigma = 0.5

    kept_indices = []
    scores_copy = scores.clone()

    for _ in range(N):
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

        # sigma_thực_tế = base_sigma * (1 + density * 2): vùng cực đông sigma gấp 3 lần vùng thưa
        adaptive_sigma = base_sigma * (1.0 + norm_densities[max_idx] * 2.0)

        # Áp dụng Gaussian Penalty của Soft-NMS
        penalty = torch.exp(-(ious ** 2) / adaptive_sigma)
        scores_copy = scores_copy * penalty

    return torch.tensor(kept_indices, dtype=torch.int64, device=boxes.device)
