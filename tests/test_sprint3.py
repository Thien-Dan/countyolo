import os
import torch
import pytest
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loss import CountYOLOLoss, UncertaintyLoss

def test_uncertainty_loss():
    """Verify that UncertaintyLoss correctly balances gradients and updates sigmas."""
    loss_module = UncertaintyLoss(num_losses=3)
    optimizer = torch.optim.SGD(loss_module.parameters(), lr=0.1)
    
    # Mock losses (require_grad = True to simulate network output)
    loss1 = torch.tensor(10.0, requires_grad=True)
    loss2 = torch.tensor(2.0, requires_grad=True)
    loss3 = torch.tensor(0.5, requires_grad=True)
    
    total_loss = loss_module([loss1, loss2, loss3])
    
    # Before step, log_vars are 0
    assert torch.allclose(loss_module.log_vars, torch.zeros(3))
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # After step, log_vars should be updated (not 0 anymore)
    assert not torch.allclose(loss_module.log_vars, torch.zeros(3))
    
    # The gradient of log_var_i should be: -L_i * exp(-log_var_i) + 1
    # Since L_1 = 10, grad_1 = -10 + 1 = -9
    # Optimizer step (SGD): var = var - lr * grad = 0 - 0.1 * (-9) = 0.9
    # So log_vars[0] should be > 0, log_vars[2] might be different.
    assert loss_module.log_vars[0] > 0, "High loss should increase log_var (decrease precision)"

def test_countyolo_loss_forward():
    """Verify CountYOLOLoss logic and shape without crashing."""
    criterion = CountYOLOLoss()
    
    B, H, W = 2, 384, 512
    C_cls = 1
    H_p, W_p = H // 4, W // 4
    
    pred_density = torch.rand((B, 1, H, W), requires_grad=True)
    pred_cls = torch.rand((B, C_cls, H_p, W_p), requires_grad=True)
    pred_boxes = torch.rand((B, 4, H_p, W_p), requires_grad=True)
    
    gt_density = torch.rand((B, 1, H, W))
    # Dummy points: batch 0 has 2 points, batch 1 has 0
    gt_points = [
        torch.tensor([[100.0, 150.0], [200.0, 250.0]]),
        torch.zeros((0, 2))
    ]
    gt_boxes = [
        torch.tensor([[100, 150, 120, 170], [200, 250, 220, 270]]),
        torch.zeros((0, 4))
    ]
    
    # Mock matcher_cost_matrix: (B, N_preds, N_gt)
    N_preds = H_p * W_p
    matcher_cost_matrix = [
        torch.rand((N_preds, 2)),
        torch.zeros((N_preds, 0))
    ]
    
    loss_dict = criterion(pred_density, pred_cls, pred_boxes, gt_density, gt_points, gt_boxes, matcher_cost_matrix)
    
    assert 'loss_density' in loss_dict
    assert 'loss_cls' in loss_dict
    assert 'loss_box' in loss_dict
    assert 'total_loss' in loss_dict
    
    # total_loss should require grad
    assert loss_dict['total_loss'].requires_grad
