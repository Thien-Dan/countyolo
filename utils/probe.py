import torch

def discriminability_probe(similarity_map: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """
    Discriminability Probe (Đo lường độ phân biệt).
    Tính variance của similarity map giữa T_fused và ảnh gốc. 
    Nếu variance quá thấp (độ tương phản kém, mờ nhạt), kích hoạt cờ fallback sang exemplar.
    
    Args:
        similarity_map (torch.Tensor): Tensor chứa similarity scores, shape (B, 1, H, W).
                                       Đây có thể là attention map hoặc feature correlation.
        threshold (float): Ngưỡng variance để quyết định fallback.
        
    Returns:
        fallback_flag (torch.Tensor): Cờ boolean shape (B,), True nếu cần fallback.
    """
    B = similarity_map.shape[0]
    
    # Flatten spatial dimensions
    sim_flat = similarity_map.view(B, -1)
    
    # Tính variance trên từng mẫu trong batch
    variance = torch.var(sim_flat, dim=1)
    
    # Nếu variance nhỏ hơn ngưỡng -> độ phân biệt kém -> kích hoạt fallback
    fallback_flag = variance < threshold
    
    return fallback_flag
