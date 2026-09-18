import torch
import torch.nn as nn
from ultralytics import YOLOWorld

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
        
        # YOLOv8s neck features thường có channel = 256 ở scale P4
        # Do ta bắt ở input của Detect head nên ta cần biết kênh.
        # Ở đây ta giả định chọn feature ở giữa (P4) có in_channels=256
        self.density_head = DensityHead(in_channels=256)

    def _register_hook(self):
        def hook_fn(module, input):
            # input là một tuple, input[0] thường là list các feature maps [P3, P4, P5]
            if isinstance(input[0], (list, tuple)) and len(input[0]) >= 3:
                # P3 (stride 8), P4 (stride 16), P5 (stride 32)
                # Dùng P4 (stride 16) làm feature cho Density Map
                self.features = input[0][1] 
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
            
        # 3. Cho qua Density Head
        density_map = self.density_head(self.features)
        
        return raw_preds, density_map

    def train_mode(self):
        self.yolo.eval() # Luôn đóng băng YOLO
        self.density_head.train()
        
    def eval_mode(self):
        self.yolo.eval()
        self.density_head.eval()
