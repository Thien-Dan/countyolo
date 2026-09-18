import torch
import torch.nn.functional as F

def decode_fcos_boxes(box_preds, stride=4.0):
    """
    Giải mã đầu ra box_preds (l, t, r, b) của mạng FCOS thành tọa độ tuyệt đối (x1, y1, x2, y2).
    
    Args:
        box_preds: Tensor (B, 4, H_feat, W_feat)
        stride: Bước nhảy của feature map so với ảnh gốc (mặc định 4)
        
    Returns:
        decoded_boxes: Tensor (B, N_preds, 4) định dạng [x1, y1, x2, y2]
    """
    B, _, H_feat, W_feat = box_preds.shape
    device = box_preds.device
    
    # Tạo grid tọa độ (y, x) trên Feature Map
    shift_y, shift_x = torch.meshgrid(
        torch.arange(H_feat, device=device), 
        torch.arange(W_feat, device=device), 
        indexing='ij'
    )
    
    # Tính tọa độ tâm của từng cell trên ảnh gốc
    cx = (shift_x.float() + 0.5) * stride
    cy = (shift_y.float() + 0.5) * stride
    cx = cx.view(-1).unsqueeze(0).expand(B, -1) # (B, N)
    cy = cy.view(-1).unsqueeze(0).expand(B, -1) # (B, N)
    
    # Flatten predicted offsets
    flat_boxes = box_preds.view(B, 4, -1) # (B, 4, N)
    
    # Dùng exp() theo FCOS chuẩn: network học offset trong log-space,
    # exp() đảm bảo l,t,r,b luôn dương (box không bị "lật ngược").
    # Nhân stride 1 lần để chuyển từ feature-pixel về pixel ảnh gốc.
    # Không nhân stride lần 2 vào cx,cy vì cx/cy đã được tính bằng cách
    # nhân shift_x/shift_y * stride ở trên.
    l = flat_boxes[:, 0, :].exp() * stride
    t = flat_boxes[:, 1, :].exp() * stride
    r = flat_boxes[:, 2, :].exp() * stride
    b = flat_boxes[:, 3, :].exp() * stride
    
    # Tọa độ tuyệt đối (x1, y1, x2, y2)
    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b
    
    # Stack lại thành (B, N, 4)
    decoded_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    
    return decoded_boxes
