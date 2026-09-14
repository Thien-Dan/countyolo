import torch

def compute_sat(density_map: torch.Tensor) -> torch.Tensor:
    """
    Tính Summed-Area Table (SAT) từ một Density Map trên GPU/CPU.
    
    Args:
        density_map (torch.Tensor): Tensor chứa density map, shape (B, 1, H, W).
        
    Returns:
        torch.Tensor: Bảng SAT có shape (B, 1, H+1, W+1). 
        Được pad thêm 1 hàng 0 ở trên và 1 cột 0 ở bên trái để tiện tính toán box.
    """
    # Tính tích phân qua chiều W (dim=3) rồi chiều H (dim=2)
    sat = torch.cumsum(density_map, dim=3)
    sat = torch.cumsum(sat, dim=2)
    
    # Pad thêm 0 để xử lý gọn các box nằm sát lề (x1=0 hoặc y1=0)
    B, C, H, W = density_map.shape
    padded_sat = torch.zeros((B, C, H + 1, W + 1), dtype=sat.dtype, device=sat.device)
    padded_sat[:, :, 1:, 1:] = sat
    
    return padded_sat

def extract_box_density(
    sat: torch.Tensor,
    boxes: torch.Tensor,
    channel_idx: int = 0
) -> torch.Tensor:
    """
    Truy xuất tổng mật độ (sum density) của các bounding box từ bảng SAT trong O(1).

    Args:
        sat (torch.Tensor): Bảng SAT đã pad, shape (B, C, H+1, W+1).
        boxes (torch.Tensor): Tensor chứa tọa độ các box, shape (N, 5).
            Mỗi row có định dạng: (batch_idx, x1, y1, x2, y2).
            Tọa độ là số nguyên (pixels), dạng inclusive [x1, x2] và [y1, y2].
            Phải đảm bảo x2 >= x1 và y2 >= y1.
        channel_idx (int): Chỉ số channel cần truy xuất từ SAT. Mặc định = 0.
            Hỗ trợ multi-scale density maps khi C > 1.

    Returns:
        torch.Tensor: Tổng mật độ trong mỗi box, shape (N, 1).
    """
    # Chuyển đổi tọa độ sang kiểu int để index
    batch_idx = boxes[:, 0].long()
    x1 = boxes[:, 1].long()
    y1 = boxes[:, 2].long()
    x2 = boxes[:, 3].long()
    y2 = boxes[:, 4].long()

    # Sanity check: đảm bảo box hợp lệ (x2 >= x1, y2 >= y1)
    # Clamp để tránh negative density thay vì raise exception
    # (giúp tránh crash trong pipeline khi có boxes bị degenerate)
    x2 = torch.maximum(x1, x2)
    y2 = torch.maximum(y1, y2)

    # Do sat đã pad thêm 1 dòng/cột ở đầu nên tọa độ x2, y2 (đáy phải) tương ứng với index x2+1, y2+1
    # Công thức: D = SAT[y2+1, x2+1] - SAT[y1, x2+1] - SAT[y2+1, x1] + SAT[y1, x1]
    ch = channel_idx
    val_br = sat[batch_idx, ch, y2 + 1, x2 + 1]
    val_tr = sat[batch_idx, ch, y1,     x2 + 1]
    val_bl = sat[batch_idx, ch, y2 + 1, x1]
    val_tl = sat[batch_idx, ch, y1,     x1]

    sum_density = val_br - val_tr - val_bl + val_tl

    return sum_density.unsqueeze(-1)
