import torch
import torch.nn as nn
from ultralytics import YOLOWorld
import torch.nn.functional as F

class SobelEdgeExtractor(nn.Module):
    """
    Trích xuất biên ảnh gốc (Edge Map) bằng bộ lọc Sobel cố định.
    Giúp Density Head bám sát ranh giới vật thể khi đếm đám đông.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def forward(self, x):
        # Chuyển ảnh RGB sang Grayscale
        x_gray = x.mean(dim=1, keepdim=True)
        # Pad để giữ nguyên kích thước
        x_pad = F.pad(x_gray, (1, 1, 1, 1), mode='reflect')
        gx = F.conv2d(x_pad, self.sobel_x)
        gy = F.conv2d(x_pad, self.sobel_y)
        # Tính độ lớn Gradient (Edge Magnitude)
        edges = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        
        # Chuẩn hóa về [0, 1] trên mỗi ảnh trong batch
        batch_size = edges.size(0)
        edges_flat = edges.view(batch_size, -1)
        max_val = edges_flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        max_val = torch.clamp(max_val, min=1e-6)
        edges = edges / max_val
        return edges

class DensityHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Mạng CNN nhẹ để dự đoán Density Map từ Feature Map
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1) # Trả về 1 kênh heatmap
        )
        # Khởi tạo trọng số nhẹ nhàng để tránh vỡ đạo hàm
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x có shape: (B, C, H, W)
        density_map = self.conv_layers(x)
        # Dùng ReLU hoặc Softplus để đảm bảo mật độ luôn >= 0
        return torch.nn.functional.relu(density_map)


class DensityYOLOWorld(nn.Module):
    def __init__(self, yolo_weights="yolov8s-world.pt"):
        super().__init__()
        print(f"Khởi tạo YOLO-World từ {yolo_weights}")
        # Tải mô hình YOLO-World gốc
        self.yolo = YOLOWorld(yolo_weights)
        
        # Đóng băng (Freeze) toàn bộ YOLO-World
        for param in self.yolo.parameters():
            param.requires_grad = False
            
        # Hook để bắt Feature Map
        self.features = None
        self._register_hook()
        
        # Edge Extractor (không có tham số huấn luyện)
        self.edge_extractor = SobelEdgeExtractor()
        
        # P3 (stride 8, 80x80) cho Density Head -- chi tiết không gian tốt hơn P4
        # 128 channels (P3) + 1 channel (Edge Map) = 129 channels
        self.density_head = DensityHead(in_channels=129)

    def _register_hook(self):
        def hook_fn(module, input):
            # input là một tuple, input[0] thường là list các feature maps [P3, P4, P5]
            if isinstance(input[0], (list, tuple)) and len(input[0]) >= 3:
                # P3 (stride 8, 80x80) cho Density Head -- chi tiết không gian tốt hơn P4
                self.features = input[0][0]  # P3
            else:
                self.features = input[0]

        # Gắn hook vào lớp Detect của YOLO (thường là layer cuối cùng)
        detect_layer = self.yolo.model.model[-1]
        detect_layer.register_forward_pre_hook(hook_fn)

    def forward(self, x):
        # 1. Chạy ảnh qua YOLO-World (chỉ để lấy feature, bounding boxes sẽ được xử lý riêng nếu cần)
        # Lưu ý: yolo.model(x) trả về raw predictions, hook sẽ tự động chạy
        raw_preds = self.yolo.model(x)
        
        # 2. Lấy Feature Map đã bắt từ hook
        if self.features is None:
            raise RuntimeError("Không bắt được Feature Map từ Hook!")
            
        # 3. Trích xuất Edge Map từ ảnh đầu vào x
        edges = self.edge_extractor(x)
        
        # Thu nhỏ Edge Map để khớp kích thước với Feature Map P3 (H/8, W/8)
        edges_down = F.interpolate(edges, size=self.features.shape[2:], mode='bilinear', align_corners=False)
        
        # Gộp chung Feature Map P3 (128) và Edge Map (1)
        fused_features = torch.cat([self.features, edges_down], dim=1)  # Shape: (B, 129, H, W)
            
        # 4. Cho qua Density Head
        density_map = self.density_head(fused_features)
        
        return raw_preds, density_map

    def train_mode(self):
        self.yolo.eval() # Luôn đóng băng YOLO
        self.density_head.train()
        
    def eval_mode(self):
        self.yolo.eval()
        self.density_head.eval()
