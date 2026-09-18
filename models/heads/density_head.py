import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityHead(nn.Module):
    """
    Multi-scale Density Head.

    Nhan P3 (stride=8) + P4 (stride=16) + P5 (stride=32) tu FPN neck,
    upsample ve cung resolution H/4, fuse, roi decode ra density map (B,1,H,W).

    Pipeline:
        P5 (H/32) → upsample 8x  → (H/4)  ┐
        P4 (H/16) → upsample 4x  → (H/4)  ├ concat → fuse_conv → upsample 4x → density (H)
        P3 (H/8)  → upsample 2x  → (H/4)  ┘

    Tai sao H/4 la diem hoi tu:
        - Du resolution de detect object nho (FSC-147 co object 2px)
        - Khong qua lon de bi OOM khi H=384, W=640
    """

    def __init__(self, in_channels: int = 256, mid_channels: int = 128):
        super().__init__()

        # Align conv: dam bao moi scale co cung so channel truoc khi fuse
        self.align_P3 = self._align_conv(in_channels, mid_channels)
        self.align_P4 = self._align_conv(in_channels, mid_channels)
        self.align_P5 = self._align_conv(in_channels, mid_channels)

        # Fuse conv sau khi concat 3 scale (3 * mid_channels input)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Decode ra density map: upsample 4x tu H/4 → H
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels // 2, mid_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels // 4, 1, kernel_size=1),
        )

    @staticmethod
    def _align_conv(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: dict) -> torch.Tensor:
        """
        Args:
            features: {'P3': (B,C,H/8,W/8), 'P4': (B,C,H/16,W/16), 'P5': (B,C,H/32,W/32)}
        Returns:
            density_map: (B, 1, H, W) — gia tri khong am (ReLU)
        """
        P3 = features['P3']   # (B, C, H/8,  W/8)
        P4 = features['P4']   # (B, C, H/16, W/16)
        P5 = features['P5']   # (B, C, H/32, W/32)

        # Target resolution: H/4 x W/4
        target_h = P3.shape[2] * 2
        target_w = P3.shape[3] * 2

        # Upsample ve H/4
        p3_up = F.interpolate(self.align_P3(P3), size=(target_h, target_w),
                               mode='bilinear', align_corners=False)
        p4_up = F.interpolate(self.align_P4(P4), size=(target_h, target_w),
                               mode='bilinear', align_corners=False)
        p5_up = F.interpolate(self.align_P5(P5), size=(target_h, target_w),
                               mode='bilinear', align_corners=False)

        # Concat + fuse
        fused = self.fuse(torch.cat([p3_up, p4_up, p5_up], dim=1))

        # Decode: upsample 4x → full resolution
        density_map = self.decode(fused)

        return F.relu(density_map)   # Density khong am


# Alias de giu tuong thich voi import cu
DummyDensityHead = DensityHead
