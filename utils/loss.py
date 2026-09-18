import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class UncertaintyLoss(nn.Module):
    """
    Tự động cân bằng các nhánh loss dựa trên Uncertainty (Kendall et al.)
    L = \sum_i (L_i / (2 * sigma_i^2) + log(sigma_i))
    """
    def __init__(self, num_losses: int = 3):
        super().__init__()
        # Khởi tạo log_var thay vì sigma để tránh giá trị âm
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: list) -> torch.Tensor:
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss

class CountYOLOLoss(nn.Module):
    """
    Module tính tổng loss cho CountYOLO:
    1. Density Loss: MSE giữa prediction (đã qua ReLU) và D_gt.
    2. Class Loss: Binary Cross Entropy cho dự đoán có/không có đối tượng.
    3. Box Loss: L1 Loss cho tọa độ hộp.
    """
    def __init__(self, w_density=1.0, w_cls=1.0, w_box=1.0, w_count=0.1):
        super().__init__()
        self.w_density = w_density
        self.w_cls = w_cls
        self.w_box = w_box
        self.w_count = w_count  # Trọng số count regularization

    def forward(self, pred_density, pred_cls, pred_boxes, gt_density, gt_points, gt_boxes, matcher_cost_matrix):
        """
        Args:
            pred_density: (B, 1, H, W)
            pred_cls: (B, num_classes, H/4, W/4)
            pred_boxes: (B, 4, H/4, W/4)
            gt_density: (B, 1, H, W)
            gt_points: list of tensors, len B, each (N_gt, 2)
            gt_boxes: list of tensors, len B, each (M_gt, 4)
            matcher_cost_matrix: (B, N_preds, N_gt) - matrix cost đã gộp Cost_dens
        """
        B, C_cls, H_p, W_p = pred_cls.shape
        N_preds = H_p * W_p
        
        # 1. DENSITY LOSS (Tính trên toàn bộ spatial dims)
        # Đảm bảo pred_density không âm
        pred_density = F.relu(pred_density)
        loss_mse = F.mse_loss(pred_density, gt_density)
        
        # Count regularization: buộc tổng density map ≈ số objects thực tế.
        # Ngăn model hội tụ vào trivial solution (predict toàn 0) để minimize MSE
        # trên density map thưa thớt (>95% pixels = 0).
        pred_count = pred_density.sum(dim=[2, 3])   # (B, 1)
        gt_count   = gt_density.sum(dim=[2, 3])     # (B, 1)
        loss_count = F.l1_loss(pred_count, gt_count)
        
        loss_density = (loss_mse + self.w_count * loss_count) * self.w_density
        
        loss_cls = torch.tensor(0.0, device=pred_density.device)
        loss_box = torch.tensor(0.0, device=pred_density.device)
        
        # Chuẩn bị flatten predictions
        flat_cls = pred_cls.view(B, C_cls, -1).permute(0, 2, 1) # (B, N_preds, C_cls)
        # pred_boxes đã được flatten và xử lý scale từ train.py (shape: B, N_preds, 4)
        flat_boxes = pred_boxes
        
        # 2. MATCHING & LOSS BOX/CLS TỪNG ẢNH
        total_matched = 0
        for b in range(B):
            N_gt = len(gt_points[b])
            if N_gt == 0:
                continue
                
            # Rút trích cost matrix cho ảnh hiện tại (tối đa lấy N_preds x N_gt)
            # Do cost_matrix từ matcher có thể rất lớn, scipy linear_sum_assignment có thể chậm
            cost_matrix = matcher_cost_matrix[b].detach().cpu().numpy() # (N_preds, N_gt)
            
            # Khử rủi ro: Nếu cost_matrix quá lớn, ta chỉ giữ lại top-k query (VD k=500) có cost thấp nhất với bất kỳ GT nào
            # để tránh scipy.optimize bị treo (complexity O(n^3)).
            if cost_matrix.shape[0] > 1000:
                min_costs = cost_matrix.min(axis=1)
                top_indices = min_costs.argsort()[:max(1000, N_gt*5)]
                sub_cost = cost_matrix[top_indices]
                row_ind, col_ind = linear_sum_assignment(sub_cost)
                # Map lại index gốc
                row_ind = top_indices[row_ind]
            else:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
            # Lấy pred được gán
            matched_cls = flat_cls[b, row_ind] # (N_gt, C_cls)
            matched_boxes = flat_boxes[b, row_ind] # (N_gt, 4)
            
            # Tính Classification Loss (Binary BCE) cho các box được gán (target = 1)
            # Khởi tạo target là 0 cho tất cả, sau đó gán 1 cho matched
            target_cls = torch.zeros_like(flat_cls[b])
            target_cls[row_ind, 0] = 1.0 
            loss_cls += F.binary_cross_entropy_with_logits(flat_cls[b], target_cls)
            
            # Tính Box Loss (L1 trên center point):
            # - Không dùng pseudo GT box từ exemplar mean size vì 3 exemplar boxes
            #   không đại diện cho kích thước tất cả objects trong ảnh.
            # - Thay vào đó, chỉ tính L1 loss trên tâm (cx, cy) của pred box so với
            #   GT point annotation. Gradient này đúng hướng và không phụ thuộc vào
            #   kích thước box giả.
            gt_pts = gt_points[b][col_ind].to(pred_density.device)  # (N_matched, 2): [y_scaled, x_scaled]
            gt_cx = gt_pts[:, 1]  # x_center
            gt_cy = gt_pts[:, 0]  # y_center
            gt_ctrs = torch.stack([gt_cx, gt_cy], dim=1)  # (N_matched, 2)
            
            # Tính tâm pred box từ [x1, y1, x2, y2]
            pred_cx = (matched_boxes[:, 0] + matched_boxes[:, 2]) / 2.0
            pred_cy = (matched_boxes[:, 1] + matched_boxes[:, 3]) / 2.0
            pred_ctrs = torch.stack([pred_cx, pred_cy], dim=1)  # (N_matched, 2)
            
            loss_box += F.l1_loss(pred_ctrs, gt_ctrs)
            total_matched += 1
            
        if total_matched > 0:
            loss_cls /= total_matched
            loss_box /= total_matched
            
        return {
            'loss_density': loss_density,
            'loss_count':   loss_count,
            'loss_cls':     loss_cls * self.w_cls,
            'loss_box':     loss_box * self.w_box,
            'total_loss':   loss_density + loss_cls * self.w_cls + loss_box * self.w_box
        }
