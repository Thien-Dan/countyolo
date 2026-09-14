import torch
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.count_yolo import CountYOLO
from utils.integral_image import compute_sat, extract_box_density

def profile():
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot profile correctly on GPU.")
        return

    device = torch.device("cuda")
    model = CountYOLO().to(device)
    model.eval()

    # Create dummy input
    B, C, H, W = 1, 3, 640, 640
    num_text = 5
    num_attr = 10
    channels = 256
    
    images = torch.rand((B, C, H, W), device=device)
    text_feats = torch.rand((B, num_text, channels), device=device)
    attr_feats = torch.rand((B, num_attr, channels), device=device)
    
    # 1000 random valid boxes for SAT testing (guaranteed x2>=x1, y2>=y1)
    N = 1000
    x1 = torch.randint(0, 140, (N,), device=device)
    y1 = torch.randint(0, 140, (N,), device=device)
    bw = torch.randint(5, 20, (N,), device=device)   # box width
    bh = torch.randint(5, 20, (N,), device=device)   # box height
    x2 = torch.clamp(x1 + bw, max=159)
    y2 = torch.clamp(y1 + bh, max=159)
    boxes = torch.stack([
        torch.zeros(N, device=device),  # batch_idx
        x1, y1, x2, y2
    ], dim=1)

    print("--- Warmup ---")
    with torch.no_grad():
        for _ in range(10):
            feat = model.backbone(images)
            feat = model.neck(feat, text_feats)
            feat = model.fusion(feat, attr_feats)
            density_map = model.density_head(feat)
            cls_scores, box_preds = model.box_head(feat)
            sat = compute_sat(density_map)
            extract_box_density(sat, boxes)
    torch.cuda.synchronize()

    # Start profiling
    print("--- Start Profiling (100 iterations) ---")
    iters = 100
    
    start_event = torch.cuda.Event(enable_timing=True)
    backbone_event = torch.cuda.Event(enable_timing=True)
    fusion_event = torch.cuda.Event(enable_timing=True)
    density_sat_event = torch.cuda.Event(enable_timing=True)
    box_event = torch.cuda.Event(enable_timing=True)

    t_backbone = 0
    t_fusion = 0
    t_density_sat = 0
    t_box = 0

    with torch.no_grad():
        for _ in range(iters):
            torch.cuda.synchronize()
            start_event.record()
            
            # 1. Backbone + Neck
            feat = model.backbone(images)
            feat = model.neck(feat, text_feats)
            backbone_event.record()
            
            # 2. Fusion
            feat_fused = model.fusion(feat, attr_feats)
            fusion_event.record()
            
            # 3. Density & SAT
            density_map = model.density_head(feat_fused)
            sat = compute_sat(density_map)
            _ = extract_box_density(sat, boxes)
            density_sat_event.record()
            
            # 4. Box Head
            cls_scores, box_preds = model.box_head(feat_fused)
            box_event.record()
            
            torch.cuda.synchronize()
            
            t_backbone += start_event.elapsed_time(backbone_event)
            t_fusion += backbone_event.elapsed_time(fusion_event)
            t_density_sat += fusion_event.elapsed_time(density_sat_event)
            t_box += density_sat_event.elapsed_time(box_event)

    t_backbone /= iters
    t_fusion /= iters
    t_density_sat /= iters
    t_box /= iters
    t_total = t_backbone + t_fusion + t_density_sat + t_box

    print(f"\nProfiling Results (ms/frame):")
    print(f"  - Backbone + Neck : {t_backbone:.2f} ms")
    print(f"  - Attribute Fusion: {t_fusion:.2f} ms")
    print(f"  - Density + SAT   : {t_density_sat:.2f} ms")
    print(f"  - Box Head        : {t_box:.2f} ms")
    print("-" * 35)
    print(f"  - Total           : {t_total:.2f} ms")
    
    fps = 1000.0 / t_total
    print(f"  -> Equivalent     : {fps:.1f} FPS")
    
    vram_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"\nMax VRAM Allocated: {vram_allocated:.2f} MB")
    
    if t_total > 20:
        print("\n[WARNING] Total time > 20ms (Not met < 50FPS)")
    else:
        print("\n[PASS] Latency is within safe limits.")

if __name__ == "__main__":
    profile()
