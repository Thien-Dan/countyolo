import torch
import torch.nn as nn

class DummyBackbone(nn.Module):
    """Mô phỏng Backbone (VD: CSPNeXt từ YOLO-World)."""
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        # Dùng một vài Conv layer để tạo đủ delay mô phỏng backbone nhẹ
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # Thêm một vài block để tăng computation giả
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Trả về feature map giảm đi 4 lần (stride=4)
        return self.conv(x)
