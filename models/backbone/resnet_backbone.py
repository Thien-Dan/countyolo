import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 Backbone with multi-scale output (P3, P4, P5).

    Output feature pyramid:
        P3: stride=8  (from layer2, 512ch → 256ch)  — objects nho
        P4: stride=16 (from layer3, 1024ch → 256ch) — objects vua
        P5: stride=32 (from layer4, 2048ch → 256ch) — objects lon / global context

    Khong upsample o day — FPN Neck se xu ly viec merge cac scale.
    """

    def __init__(self, out_channels: int = 256, pretrained: bool = True):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        # Stem + early layers (shared)
        self.stem = nn.Sequential(
            resnet.conv1,   # stride 2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # stride 4 total
        )
        self.layer1 = resnet.layer1   # stride 4,  256 ch
        self.layer2 = resnet.layer2   # stride 8,  512 ch  → P3
        self.layer3 = resnet.layer3   # stride 16, 1024 ch → P4
        self.layer4 = resnet.layer4   # stride 32, 2048 ch → P5

        # Lateral conv: reduce to out_channels at each scale
        self.lat_P3 = self._lateral(512,  out_channels)
        self.lat_P4 = self._lateral(1024, out_channels)
        self.lat_P5 = self._lateral(2048, out_channels)

    @staticmethod
    def _lateral(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            dict with keys 'P3', 'P4', 'P5'
            P3: (B, out_channels, H/8,  W/8)
            P4: (B, out_channels, H/16, W/16)
            P5: (B, out_channels, H/32, W/32)
        """
        x = self.stem(x)
        x = self.layer1(x)

        c3 = self.layer2(x)   # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32

        P3 = self.lat_P3(c3)
        P4 = self.lat_P4(c4)
        P5 = self.lat_P5(c5)

        return {'P3': P3, 'P4': P4, 'P5': P5}

    def freeze(self):
        """Dong bang toan bo backbone (stem + layer1-4 + lateral convs)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_deep(self):
        """Mo bang layer3 + layer4 + lateral P4/P5 voi lr nho (curriculum)."""
        for module in [self.layer3, self.layer4, self.lat_P4, self.lat_P5]:
            for param in module.parameters():
                param.requires_grad = True
