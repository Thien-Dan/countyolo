import torch
import torch.nn as nn
from .integral_image import compute_sat, extract_box_density

class DensityGuidedMatcher(nn.Module):
    """
    Density-Guided Matcher.
    Tích hợp chi phí Hungarian Matching thông thường với chi phí mật độ (Cost_dens)
    được tra cứu từ bảng SAT.
    """
    def __init__(self, lambda_dens_max: float = 2.0, warmup_epochs: int = 5):
        super().__init__()
        self.lambda_dens_max = lambda_dens_max
        self.warmup_epochs = warmup_epochs

    def get_lambda_dens(self, current_epoch: int) -> float:
        """Warm-up schedule cho tham số lambda_dens(t)"""
        if self.warmup_epochs <= 0:
            return self.lambda_dens_max
            
        progress = min(1.0, current_epoch / self.warmup_epochs)
        return self.lambda_dens_max * progress

    @torch.no_grad()
    def compute_cost_dens(self, density_map: torch.Tensor, pred_boxes: torch.Tensor) -> torch.Tensor:
        """
        Tính Cost_dens qua SAT.
        
        Args:
            density_map: (B, 1, H, W) 
            pred_boxes: (B, N_preds, 4) định dạng [x1, y1, x2, y2] ở tỉ lệ ảnh H, W.
            
        Returns:
            Cost_dens: (B, N_preds, 1) đã squash về đoạn [0, 1].
        """
        B, N_preds, _ = pred_boxes.shape
        
        # 1. Tính SAT (O(1) lookup map)
        sat = compute_sat(density_map)
        
        # 2. Tạo tensor boxes cho extract_box_density (format: batch_idx, x1, y1, x2, y2)
        # Flatten boxes về shape (B*N_preds, 5)
        batch_idx = torch.arange(B, device=density_map.device).view(B, 1).expand(B, N_preds)
        flat_batch_idx = batch_idx.reshape(-1, 1)
        flat_pred_boxes = pred_boxes.reshape(-1, 4)
        
        sat_boxes = torch.cat([flat_batch_idx, flat_pred_boxes], dim=-1)
        
        # 3. Truy xuất tổng mật độ (sum density)
        # Giả sử D_sum là tổng density trong box
        # shape: (B*N_preds, 1)
        sum_density = extract_box_density(sat, sat_boxes, channel_idx=0)
        sum_density = sum_density.view(B, N_preds, 1)
        
        # 4. Tính chi phí: Cost_dens = |D_sum - 1| (vì mỗi box chỉ chứa 1 vật thể/nhãn lý tưởng)
        cost_dens_raw = torch.abs(sum_density - 1.0)
        
        # 5. Squash về đoạn [0, 1] sử dụng hàm Tanh (hoặc 1 - exp(-x))
        cost_dens_squashed = torch.tanh(cost_dens_raw)
        
        return cost_dens_squashed

    def forward(
        self, 
        base_cost_matrix: torch.Tensor, 
        density_map: torch.Tensor, 
        pred_boxes: torch.Tensor,
        current_epoch: int = 0
    ) -> torch.Tensor:
        """
        Gộp Cost_dens vào Cost Matrix tổng thể.
        (compute_cost_dens đã bọc no_grad để tránh memory leak).
        
        Args:
            base_cost_matrix: (B, N_preds, N_targets) - Cost từ classification & box iou
            density_map: (B, 1, H, W)
            pred_boxes: (B, N_preds, 4)
            current_epoch: epoch hiện tại để tính warmup.
            
        Returns:
            final_cost_matrix: (B, N_preds, N_targets)
        """
        # Tính Density Cost
        cost_dens = self.compute_cost_dens(density_map, pred_boxes) # (B, N_preds, 1)
        
        # Broadcasting cost_dens (B, N_preds, 1) lên (B, N_preds, N_targets)
        # Lấy trọng số λ_dens(t)
        lambda_dens = self.get_lambda_dens(current_epoch)
        
        final_cost_matrix = base_cost_matrix + lambda_dens * cost_dens
        return final_cost_matrix
