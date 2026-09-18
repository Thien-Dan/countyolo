import torch
import torch.nn as nn


class FCOSHead(nn.Module):
    """
    FCOS-style detection head voi Centerness branch.

    FCOS (Fully Convolutional One-Stage):
      - Cls branch:        (B, num_classes, H, W) — logits
      - Reg branch:        (B, 4, H, W)           — log-space (l, t, r, b) offsets
      - Centerness branch: (B, 1, H, W)            — logits

    Centerness = sqrt( min(l,r)/max(l,r) * min(t,b)/max(t,b) )
    Giup loc bo cac box predict xa tam vat the (chap nhan voi confidence thap hon).

    Multi-scale: head CHIA SE WEIGHT giua P3 va P4 (tuong tu FCOS goc),
    tiet kiem tham so va giup generalize tot hon.
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 1,
                 num_convs: int = 4, mid_channels: int = 256):
        super().__init__()

        # Shared tower (dung chung cho cls va reg)
        cls_tower = []
        reg_tower = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else mid_channels
            cls_tower += [
                nn.Conv2d(in_ch, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, mid_channels),  # GN tot hon BN khi feature map nho
                nn.ReLU(inplace=True),
            ]
            reg_tower += [
                nn.Conv2d(in_ch, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, mid_channels),
                nn.ReLU(inplace=True),
            ]

        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)

        # Prediction heads
        self.cls_pred        = nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        self.reg_pred        = nn.Conv2d(mid_channels, 4, kernel_size=1)
        self.centerness_pred = nn.Conv2d(mid_channels, 1, kernel_size=1)

        # Scale parameter: moi FPN level co 1 scalar rieng de scale reg output
        # (tuong tu FCOS paper, giup head hoc duoc o scale khac nhau)
        self.scale_P3 = nn.Parameter(torch.ones(1))
        self.scale_P4 = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self):
        """Khoi tao bias cls_pred theo prior probability de on dinh training ban dau."""
        import math
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)
        nn.init.constant_(self.centerness_pred.bias, 0.0)
        # Khoi tao reg_pred voi weight nho → exp(~0) = 1 pixel offset ban dau
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.zeros_(self.reg_pred.bias)

    def forward_single(self, x: torch.Tensor, scale: torch.Tensor):
        """Forward qua 1 scale FPN."""
        cls_feat = self.cls_tower(x)
        reg_feat = self.reg_tower(x)

        cls_scores  = self.cls_pred(cls_feat)           # (B, num_cls, H, W)
        box_preds   = self.reg_pred(reg_feat) * scale   # (B, 4, H, W) scaled
        centerness  = self.centerness_pred(cls_feat)    # (B, 1, H, W)

        return cls_scores, box_preds, centerness

    def forward(self, features: dict) -> dict:
        """
        Args:
            features: {'P3': ..., 'P4': ..., 'P5': ...} — output tu FPN neck
        Returns:
            dict voi P3 va P4 predictions:
            {
              'cls_scores':  {'P3': (B,1,H/8,W/8),  'P4': (B,1,H/16,W/16)},
              'box_preds':   {'P3': (B,4,H/8,W/8),  'P4': (B,4,H/16,W/16)},
              'centerness':  {'P3': (B,1,H/8,W/8),  'P4': (B,1,H/16,W/16)},
            }
        """
        P3 = features['P3']
        P4 = features['P4']
        # P5 khong dung cho detection (resolution qua nho, chi dung cho density)

        cls_P3, box_P3, ctr_P3 = self.forward_single(P3, self.scale_P3)
        cls_P4, box_P4, ctr_P4 = self.forward_single(P4, self.scale_P4)

        return {
            'cls_scores': {'P3': cls_P3, 'P4': cls_P4},
            'box_preds':  {'P3': box_P3, 'P4': box_P4},
            'centerness': {'P3': ctr_P3, 'P4': ctr_P4},
        }


# Alias de giu tuong thich
DummyNMSFreeHead = FCOSHead
