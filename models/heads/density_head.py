import torch
import torch.nn as nn

class DummyDensityHead(nn.Module):
    """Mô phỏng Density Head dự đoán Density Map."""
    def __init__(self, in_channels=256):
        super().__init__()
        # Conv3x3 + 2 lần Upsample 2x = Upsample 4x
        # Backbone stride-4 -> feature map H/4 x W/4 -> cần 4x để về H x W
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(128)
        self.relu  = nn.ReLU()
        # Upsample 2x lần 1
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_mid  = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2       = nn.BatchNorm2d(64)
        # Upsample 2x lần 2 -> tổng 4x
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2     = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H/4, W/4)
        feat = self.relu(self.bn1(self.conv1(x)))
        feat = self.upsample1(feat)                     # -> (B, 128, H/2, W/2)
        feat = self.relu(self.bn2(self.conv_mid(feat))) # -> (B, 64,  H/2, W/2)
        feat = self.upsample2(feat)                     # -> (B, 64,  H,   W)
        density_map = self.conv2(feat)                  # -> (B, 1,   H,   W)
        return torch.relu(density_map) # Density không âm
