import torch
import math

def generate_gaussian_kernel(sigma: float, kernel_size: int = None) -> torch.Tensor:
    """
    Sinh một kernel Gaussian 2D. Tích phân (tổng) của kernel luôn bằng 1.0.
    
    Args:
        sigma (float): Độ lệch chuẩn của phân bố.
        kernel_size (int, optional): Kích thước kernel (vuông). Nếu không cung cấp,
                                     sẽ tự động tính toán dựa trên sigma (thường là 3*sigma).
                                     
    Returns:
        torch.Tensor: Kernel 2D shape (K, K).
    """
    if kernel_size is None:
        # Quy tắc 3-sigma để bao phủ 99.7% diện tích phân bố
        kernel_size = max(3, int(2 * math.ceil(3 * sigma) + 1))
        
    # Đảm bảo kernel size là số lẻ
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    center = kernel_size // 2
    x = torch.arange(kernel_size, dtype=torch.float32) - center
    y = torch.arange(kernel_size, dtype=torch.float32) - center
    
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    
    # Tính giá trị Gaussian
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Normalize để tổng luôn bằng 1.0 (Bảo toàn số lượng đối tượng)
    kernel = kernel / kernel.sum()
    
    return kernel

def generate_density_map(
    image_h: int, 
    image_w: int, 
    points: torch.Tensor, 
    sigma: float = 4.0
) -> torch.Tensor:
    """
    Sinh Ground-Truth (GT) Density Map từ các tọa độ điểm tâm.
    
    Args:
        image_h (int): Chiều cao ảnh.
        image_w (int): Chiều rộng ảnh.
        points (torch.Tensor): Tọa độ các điểm (x, y) của các đối tượng, shape (N, 2).
        sigma (float): Độ trải rộng của Gaussian kernel.
        
    Returns:
        torch.Tensor: Density map shape (1, image_h, image_w). Tổng giá trị trên map
                      sẽ xấp xỉ bằng N (số lượng điểm).
    """
    density_map = torch.zeros((1, image_h, image_w), dtype=torch.float32)
    
    if len(points) == 0:
        return density_map
        
    kernel = generate_gaussian_kernel(sigma)
    kernel_size = kernel.shape[0]
    radius = kernel_size // 2
    
    for point in points:
        x, y = int(point[0].item()), int(point[1].item())
        
        # Bỏ qua các điểm nằm hoàn toàn ngoài ảnh
        if x < 0 or y < 0 or x >= image_w or y >= image_h:
            continue
            
        # Xác định bounding box của kernel trên ảnh
        x_min, x_max = max(0, x - radius), min(image_w, x + radius + 1)
        y_min, y_max = max(0, y - radius), min(image_h, y + radius + 1)
        
        # Xác định phần tương ứng của kernel
        k_x_min = radius - (x - x_min)
        k_x_max = radius + (x_max - x)
        k_y_min = radius - (y - y_min)
        k_y_max = radius + (y_max - y)
        
        # Trích xuất phần kernel (đã bị crop nếu sát lề)
        k_cropped = kernel[k_y_min:k_y_max, k_x_min:k_x_max]
        
        # Nếu điểm nằm sát lề, việc crop kernel sẽ làm mất mát tổng diện tích (sum < 1.0).
        # Tùy thuộc vào thiết kế, ta có thể chọn normalize lại phần crop, hoặc giữ nguyên.
        # Ở đây ta giữ nguyên để density phản ánh việc object bị cắt lề.
        
        # Thêm kernel vào density map
        density_map[0, y_min:y_max, x_min:x_max] += k_cropped
        
    return density_map
