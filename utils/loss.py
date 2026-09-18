import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class UncertaintyLoss(nn.Module):
    """
    Tu dong can bang cac nhanh loss dua tren Uncertainty (Kendall et al.)
    L = sum_i (L_i / (2 * sigma_i^2) + log(sigma_i))
    """
    def __init__(self, num_losses: int = 3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: list) -> torch.Tensor:
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Focal Loss cho class imbalance (Lin et al., RetinaNet 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Tai sao can Focal Loss thay vi BCE:
    - FSC-147: ~2% foreground / 98% background
    - BCE bi "starved" boi background → model hoc predict 0 het
    - Focal Loss: down-weight easy negatives boi (1-pt)^gamma
      Khi pt ≈ 1 (easy correct), contribution ≈ 0
      Khi pt ≈ 0 (hard), contribution = 1 (full gradient)

    Args:
        inputs:  (N, C) raw logits
        targets: (N, C) binary targets {0, 1}
        alpha:   balance positive/negative (0.25 = 4x weight cho positive)
        gamma:   focusing parameter (2.0 = RetinaNet default)
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_loss = alpha_t * ((1 - p_t) ** gamma) * ce_loss

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    return focal_loss


def compute_centerness_targets(boxes: torch.Tensor) -> torch.Tensor:
    """
    Tinh centerness target tu decoded boxes [x1,y1,x2,y2] va center.

    centerness = sqrt( min(l,r)/max(l,r+eps) * min(t,b)/max(t,b+eps) )

    Boxes chua decode → dung raw (l,t,r,b) tu reg_pred:
        centerness = sqrt( min(l,r)/max(l,r) * min(t,b)/max(t,b) )

    Args:
        boxes: (N, 4) — raw (l, t, r, b) positive predictions
    Returns:
        centerness: (N,) in [0, 1]
    """
    l, t, r, b = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    eps = 1e-6
    centerness = torch.sqrt(
        (torch.min(l, r) / (torch.max(l, r) + eps)) *
        (torch.min(t, b) / (torch.max(t, b) + eps))
    )
    return centerness.clamp(0, 1)


class CountYOLOLoss(nn.Module):
    """
    Loss cho CountYOLO v2 voi multi-scale FCOS head:

    1. Density Loss:  MSE(density_pred, density_gt) + w_count * L1(count)
    2. Focal Loss:    sigmoid_focal_loss(cls_pred, cls_gt) — multi-scale P3+P4
    3. Box Loss:      L1(pred_center, gt_point) — center-point regression
    4. Centerness:    BCE(pred_centerness, gt_centerness) — suppress off-center boxes
    """

    def __init__(self, w_density: float = 1.0, w_cls: float = 1.0,
                 w_box: float = 1.0, w_count: float = 0.1,
                 w_centerness: float = 0.5,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.w_density    = w_density
        self.w_cls        = w_cls
        self.w_box        = w_box
        self.w_count      = w_count
        self.w_centerness = w_centerness
        self.focal_alpha  = focal_alpha
        self.focal_gamma  = focal_gamma

    def forward(self, pred_density, cls_dict, box_dict, ctr_dict,
                gt_density, gt_points, gt_boxes, matcher_cost_matrix):
        """
        Args:
            pred_density: (B, 1, H, W)
            cls_dict:     {'P3': (B,1,Hf3,Wf3), 'P4': (B,1,Hf4,Wf4)}
            box_dict:     {'P3': (B,4,Hf3,Wf3), 'P4': (B,4,Hf4,Wf4)}  raw log-space
            ctr_dict:     {'P3': (B,1,Hf3,Wf3), 'P4': (B,1,Hf4,Wf4)}
            gt_density:   (B, 1, H, W)
            gt_points:    list[Tensor(N_gt, 2)]  — [y, x] trong khong gian anh goc
            gt_boxes:     list[Tensor(M_gt, 4)]
            matcher_cost_matrix: list[Tensor(N_preds, N_gt)] — da merge P3+P4
        Returns:
            dict cac thanh phan loss
        """
        device = pred_density.device
        B = pred_density.shape[0]

        # ── 1. Density Loss ────────────────────────────────────────────────────
        pred_density = F.relu(pred_density)
        loss_mse = F.mse_loss(pred_density, gt_density)

        pred_count = pred_density.sum(dim=[2, 3])
        gt_count   = gt_density.sum(dim=[2, 3])
        loss_count = F.l1_loss(pred_count, gt_count)

        loss_density = (loss_mse + self.w_count * loss_count) * self.w_density

        # ── 2. Flatten multi-scale predictions ────────────────────────────────
        # Merge P3 + P4: concat theo dim N_preds
        flat_cls_P3 = cls_dict['P3'].view(B, 1, -1).permute(0, 2, 1)  # (B, N3, 1)
        flat_cls_P4 = cls_dict['P4'].view(B, 1, -1).permute(0, 2, 1)  # (B, N4, 1)
        flat_cls    = torch.cat([flat_cls_P3, flat_cls_P4], dim=1)     # (B, N3+N4, 1)

        flat_box_P3 = box_dict['P3'].view(B, 4, -1).permute(0, 2, 1)  # (B, N3, 4)
        flat_box_P4 = box_dict['P4'].view(B, 4, -1).permute(0, 2, 1)  # (B, N4, 4)
        flat_box    = torch.cat([flat_box_P3, flat_box_P4], dim=1)     # (B, N3+N4, 4)

        flat_ctr_P3 = ctr_dict['P3'].view(B, 1, -1).permute(0, 2, 1)  # (B, N3, 1)
        flat_ctr_P4 = ctr_dict['P4'].view(B, 1, -1).permute(0, 2, 1)  # (B, N4, 1)
        flat_ctr    = torch.cat([flat_ctr_P3, flat_ctr_P4], dim=1)     # (B, N3+N4, 1)

        # ── 3. Matching & Per-image Loss ───────────────────────────────────────
        loss_cls        = torch.tensor(0.0, device=device)
        loss_box        = torch.tensor(0.0, device=device)
        loss_centerness = torch.tensor(0.0, device=device)
        total_matched   = 0

        for b in range(B):
            N_gt = len(gt_points[b])
            if N_gt == 0:
                continue

            cost_matrix = matcher_cost_matrix[b].detach().cpu().numpy()

            # Sub-sample neu cost_matrix qua lon (O(n^3))
            if cost_matrix.shape[0] > 1000:
                min_costs   = cost_matrix.min(axis=1)
                top_indices = min_costs.argsort()[:max(1000, N_gt * 5)]
                sub_cost    = cost_matrix[top_indices]
                row_ind, col_ind = linear_sum_assignment(sub_cost)
                row_ind = top_indices[row_ind]
            else:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Focal Loss: tren toan bo N_preds (background + foreground)
            target_cls = torch.zeros_like(flat_cls[b])   # (N_preds, 1)
            target_cls[row_ind, 0] = 1.0
            loss_cls += sigmoid_focal_loss(
                flat_cls[b], target_cls,
                alpha=self.focal_alpha, gamma=self.focal_gamma
            )

            # Box Loss: L1 tren center point (ky vong thap hon nho FCOS bias init)
            gt_pts = gt_points[b][col_ind].to(device)   # (N_matched, 2): [y, x]
            gt_ctrs = torch.stack([gt_pts[:, 1], gt_pts[:, 0]], dim=1)  # [x, y]

            matched_boxes = flat_box[b, row_ind]  # (N_matched, 4)
            # Su dung raw (l,t,r,b) chua decode de tinh centerness target
            ctr_target = compute_centerness_targets(matched_boxes)  # (N_matched,)

            # Centerness loss (BCE tren matched positions)
            matched_ctr = flat_ctr[b, row_ind, 0]  # (N_matched,)
            loss_centerness += F.binary_cross_entropy_with_logits(
                matched_ctr, ctr_target
            )

            # Box center prediction tu decoded boxes (dung clamp exp nhu box_utils)
            import math
            MAX_LOG = 4.5
            l_dec = matched_boxes[:, 0].clamp(max=MAX_LOG).exp()
            t_dec = matched_boxes[:, 1].clamp(max=MAX_LOG).exp()
            r_dec = matched_boxes[:, 2].clamp(max=MAX_LOG).exp()
            b_dec = matched_boxes[:, 3].clamp(max=MAX_LOG).exp()

            # Pred center la cx, cy (chua co stride offset — chi lap loss tren offset)
            # → dung L1 giua (r - l) vs 0 va (b - t) vs 0 nhu centerness proxy
            # → hoac dung gt_ctrs neu biet stride (train.py truyen vao decoded boxes)
            # Giu center-point L1 nhu cu cho on dinh
            pred_cx = (r_dec - l_dec) / 2.0
            pred_cy = (b_dec - t_dec) / 2.0

            # NOTE: day la "offset" center, khong phai absolute.
            # Absolute center duoc tinh trong train.py (decode_fcos_boxes).
            # Loss nay chi penalize su bat doi xung (l≠r, t≠b).
            # Center-point absolute loss duoc tinh tu matched decoded boxes trong train.py.
            loss_box += F.l1_loss(pred_cx, torch.zeros_like(pred_cx)) + \
                        F.l1_loss(pred_cy, torch.zeros_like(pred_cy))

            total_matched += 1

        if total_matched > 0:
            loss_cls        /= total_matched
            loss_box        /= total_matched
            loss_centerness /= total_matched

        total_loss = (
            loss_density
            + loss_cls        * self.w_cls
            + loss_box        * self.w_box
            + loss_centerness * self.w_centerness
        )

        return {
            'loss_density':    loss_density,
            'loss_count':      loss_count,
            'loss_cls':        loss_cls        * self.w_cls,
            'loss_box':        loss_box        * self.w_box,
            'loss_centerness': loss_centerness * self.w_centerness,
            'total_loss':      total_loss,
        }
