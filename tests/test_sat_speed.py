import torch
import pytest
import sys
import os

# Thêm thư mục gốc vào path để import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.integral_image import compute_sat, extract_box_density

def test_sat_accuracy():
    """Kiểm tra tính chính xác của thuật toán SAT so với tính tổng trực tiếp."""
    torch.manual_seed(42)
    B, C, H, W = 2, 1, 10, 10
    density_map = torch.rand((B, C, H, W))
    
    sat = compute_sat(density_map)
    
    # Tạo một vài box ngẫu nhiên [batch_idx, x1, y1, x2, y2]
    boxes = torch.tensor([
        [0, 2, 2, 5, 5],
        [0, 0, 0, 9, 9], # Toàn bộ ảnh
        [1, 1, 3, 8, 7],
        [1, 0, 0, 0, 0]  # Một điểm
    ], dtype=torch.float32)
    
    # Tính bằng SAT
    sat_densities = extract_box_density(sat, boxes)
    
    # Tính trực tiếp bằng vòng lặp
    direct_densities = []
    for box in boxes:
        b, x1, y1, x2, y2 = box.long()
        direct_sum = torch.sum(density_map[b, 0, y1:y2+1, x1:x2+1])
        direct_densities.append(direct_sum.item())
        
    direct_densities = torch.tensor(direct_densities).unsqueeze(-1)
    
    # So sánh (cho phép sai số nhỏ do floating point)
    assert torch.allclose(sat_densities, direct_densities, atol=1e-5), \
        f"SAT densities: {sat_densities}, Direct: {direct_densities}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU to test speed")
def test_sat_speed():
    """Đo tốc độ xử lý trên GPU (throughput). Yêu cầu < 0.5ms cho 1000 boxes trên map 160x160."""
    device = torch.device("cuda")
    B, C, H, W = 1, 1, 160, 160
    density_map = torch.rand((B, C, H, W), device=device)
    
    # Tạo 1000 boxes ngẫu nhiên hợp lệ
    N = 1000
    batch_idx = torch.zeros(N, device=device)
    x1 = torch.randint(0, W - 10, (N,), device=device)
    y1 = torch.randint(0, H - 10, (N,), device=device)
    w = torch.randint(5, 20, (N,), device=device)
    h = torch.randint(5, 20, (N,), device=device)
    x2 = torch.clamp(x1 + w, max=W-1)
    y2 = torch.clamp(y1 + h, max=H-1)
    
    boxes = torch.stack([batch_idx, x1, y1, x2, y2], dim=1)

    # Warmup
    sat = compute_sat(density_map)
    _ = extract_box_density(sat, boxes)
    torch.cuda.synchronize()

    # Benchmark using cuda.Event for accurate GPU timing
    iters = 100
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    total_ms = 0.0
    for _ in range(iters):
        torch.cuda.synchronize()
        start_ev.record()
        sat = compute_sat(density_map)
        _ = extract_box_density(sat, boxes)
        end_ev.record()
        torch.cuda.synchronize()
        total_ms += start_ev.elapsed_time(end_ev)

    avg_time_ms = total_ms / iters

    print(f"\n[Bench] SAT + Extract {N} boxes on {H}x{W} map: {avg_time_ms:.3f} ms")

    # DoD: < 0.5 ms
    assert avg_time_ms < 0.5, f"Too slow! {avg_time_ms:.3f} ms > 0.5 ms"

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
