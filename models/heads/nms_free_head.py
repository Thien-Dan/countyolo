import torch
import torch.nn as nn

class DummyNMSFreeHead(nn.Module):
    """Mô phỏng Dual-Label Head (One-to-Many & One-to-One)."""
    def __init__(self, in_channels=256, num_classes=1):
        super().__init__()
        # Mô phỏng tính toán hồi quy box và phân loại
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 4, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        cls_scores = self.cls_conv(x)
        box_preds = self.reg_conv(x)
        return cls_scores, box_preds
