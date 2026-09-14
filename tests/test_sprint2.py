import os
import torch
import pytest
import sys
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.density_generator import generate_density_map
from utils.matcher import DensityGuidedMatcher
from utils.probe import discriminability_probe
from datasets.fsc147_dataset import FSC147Dataset

def test_density_generator_sum():
    """Verify that the sum of the density map equals the number of objects."""
    H, W = 100, 100
    # Create 3 points well within the image boundaries
    points = torch.tensor([
        [20, 20],
        [50, 50],
        [80, 80]
    ], dtype=torch.float32)
    
    density_map = generate_density_map(H, W, points, sigma=2.0)
    
    total_density = density_map.sum().item()
    N = len(points)
    
    # Allow small floating point errors
    assert math.isclose(total_density, N, abs_tol=1e-4), f"Expected {N}, got {total_density}"

def test_matcher_no_grad():
    """Verify that DensityGuidedMatcher runs in no_grad mode and works correctly."""
    matcher = DensityGuidedMatcher(lambda_dens_max=2.0, warmup_epochs=5)
    
    B, N_preds, N_targets = 2, 10, 5
    H, W = 64, 64
    
    base_cost = torch.rand((B, N_preds, N_targets), requires_grad=True)
    density_map = torch.rand((B, 1, H, W), requires_grad=True)
    pred_boxes = torch.rand((B, N_preds, 4)) * 30 # [x1, y1, x2, y2]
    # Ensure x2 >= x1, y2 >= y1
    pred_boxes[:, :, 2] += pred_boxes[:, :, 0] + 5
    pred_boxes[:, :, 3] += pred_boxes[:, :, 1] + 5
    
    # Run matcher
    final_cost = matcher(base_cost, density_map, pred_boxes, current_epoch=2)
    
    # final_cost should require grad ONLY because base_cost requires grad, 
    # but the density cost part should NOT introduce density_map into the computation graph
    assert final_cost.requires_grad == True
    
    # We can check if density_map is in the grad graph by calculating gradients
    loss = final_cost.sum()
    loss.backward()
    
    assert base_cost.grad is not None, "base_cost should have gradients"
    assert density_map.grad is None, "density_map MUST NOT have gradients to prevent memory leak"

def test_discriminability_probe():
    """Verify that probe triggers fallback on low variance similarity maps."""
    B, C, H, W = 2, 1, 10, 10
    
    # Batch 0: High variance (distinct peaks)
    sim_map_good = torch.zeros((1, C, H, W))
    sim_map_good[0, 0, 5, 5] = 100.0
    
    # Batch 1: Low variance (flat / blurry)
    sim_map_bad = torch.ones((1, C, H, W)) * 0.5
    
    sim_map = torch.cat([sim_map_good, sim_map_bad], dim=0)
    
    # The variance of sim_map_good will be very high, sim_map_bad will be 0.0
    fallback = discriminability_probe(sim_map, threshold=0.1)
    
    assert fallback[0].item() == False, "High variance should NOT fallback"
    assert fallback[1].item() == True, "Low variance SHOULD fallback"

@pytest.mark.skipif(not os.path.exists('data/FSC147/annotation_FSC147_384.json'), reason="FSC147 dataset not found")
def test_fsc147_dataset():
    """Verify dataset loader: coordinate scaling and density map integrity."""
    dataset = FSC147Dataset(data_dir='data/FSC147', split='train')
    assert len(dataset) > 0, "Dataset should not be empty"

    item = dataset[0]
    assert 'image' in item
    assert 'density_map' in item
    assert 'points' in item
    assert 'boxes' in item

    image       = item['image']        # (3, H, W)
    density_map = item['density_map']  # (1, H, W)
    pts         = item['points']       # (N, 2): [y_scaled, x_scaled]
    boxes       = item['boxes']        # (M, 4): [x1, y1, x2, y2]

    C, H, W = image.shape
    assert C == 3, "Image should have 3 channels"
    assert H == 384, "Image height should be 384"
    assert density_map.shape == (1, H, W), "Density map must match image spatial dims"

    N     = len(pts)
    d_sum = density_map.sum().item()

    if N > 0:
        # Density sum should be very close to N (within 5% tolerance for boundary effects)
        assert d_sum > 0, "Density map must not be zero when points exist"
        assert abs(d_sum - N) / N < 0.05, (
            f"Density sum {d_sum:.2f} deviates more than 5% from N={N}. "
            f"Likely a coordinate scaling or [y,x] ordering bug."
        )

    if len(boxes) > 0:
        # All scaled box coordinates must be within image bounds
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        assert (x1 >= 0).all() and (x2 <= W).all(), "Box x coords out of image width"
        assert (y1 >= 0).all() and (y2 <= H).all(), "Box y coords out of image height"
        assert (x2 > x1).all(), "x2 must be > x1"
        assert (y2 > y1).all(), "y2 must be > y1"

    if len(pts) > 0:
        # All scaled points must be within image bounds
        assert (pts[:, 0] >= 0).all() and (pts[:, 0] < H).all(), "Point y coords out of bounds"
        assert (pts[:, 1] >= 0).all() and (pts[:, 1] < W).all(), "Point x coords out of bounds"
