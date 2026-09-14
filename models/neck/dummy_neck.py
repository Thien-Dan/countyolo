import torch
import torch.nn as nn

class DummyRepVLPAN(nn.Module):
    """Mô phỏng RepVL-PAN Neck (Vision-Language Fusion)."""
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        # Mô phỏng tính toán cross-attention nhẹ bằng Conv
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x, text_feats):
        # text_feats: shape (B, N, C) - mô phỏng text embedding
        # Trong dummy này, ta chỉ làm toán x cơ bản và một chút phép nhân để tốn thời gian
        # Lấy trung bình text feats để ra context vector
        context = text_feats.mean(dim=1)[..., None, None] # (B, C, 1, 1)
        
        # Scale feature map bằng text context
        out = self.conv(x) * context.sigmoid()
        return out
