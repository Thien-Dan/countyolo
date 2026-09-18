import torch
import torch.nn as nn
import torch.nn.functional as F


class VLCrossAttention(nn.Module):
    """
    Vision-Language Cross-Attention: visual tokens attend to text tokens.

    Query = visual feature (spatial locations flattened)
    Key/Value = text features

    Giup tung scale trong FPN biet "dang tim doi tuong gi" dua vao text embedding.
    """

    def __init__(self, channels: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_v = nn.LayerNorm(channels)
        self.norm_t = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Linear(channels, channels)
        self.norm_out = nn.LayerNorm(channels)

    def forward(self, visual: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual: (B, C, H, W)  — feature map tu FPN
            text:   (B, N, C)     — text embedding tu CLIPTextEncoder
        Returns:
            out: (B, C, H, W)     — visual sau khi fuse voi text context
        """
        B, C, H, W = visual.shape

        # Flatten spatial dims: (B, H*W, C)
        vis_flat = visual.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Pre-norm
        q = self.norm_v(vis_flat)    # (B, H*W, C)
        k = self.norm_t(text)        # (B, N, C)
        v = k

        # Cross-attention: visual query, text key/value
        attn_out, _ = self.attn(q, k, v)   # (B, H*W, C)

        # Residual + projection
        attn_out = self.norm_out(vis_flat + self.proj(attn_out))

        # Reshape back to spatial (B, C, H, W)
        return attn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class FPN_VL_Neck(nn.Module):
    """
    Feature Pyramid Network (FPN) voi Vision-Language cross-attention
    tai moi scale.

    Luong xu ly:
        P5 → VL_Attn → lateral_P5
                              ↓ upsample + add
        P4 → VL_Attn → lateral_P4
                              ↓ upsample + add
        P3 → VL_Attn → lateral_P3

        Moi scale sau do qua output conv de lam muot feature.

    Output: dict {'P3': (B,C,H/8,W/8), 'P4': (B,C,H/16,W/16), 'P5': (B,C,H/32,W/32)}
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256,
                 num_heads: int = 8):
        super().__init__()

        # VL cross-attention tai moi scale (rieng biet → moi scale hoc context khac nhau)
        self.vl_P3 = VLCrossAttention(in_channels, num_heads)
        self.vl_P4 = VLCrossAttention(in_channels, num_heads)
        self.vl_P5 = VLCrossAttention(in_channels, num_heads)

        # Output conv de smooth sau khi add top-down
        self.out_P3 = self._out_conv(in_channels, out_channels)
        self.out_P4 = self._out_conv(in_channels, out_channels)
        self.out_P5 = self._out_conv(in_channels, out_channels)

    @staticmethod
    def _out_conv(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: dict, text_feats: torch.Tensor) -> dict:
        """
        Args:
            features:   {'P3', 'P4', 'P5'} — output cua ResNet50Backbone
            text_feats: (B, N, C)           — output cua CLIPTextEncoder
        Returns:
            {'P3', 'P4', 'P5'} — da fuse voi text, top-down FPN
        """
        P3, P4, P5 = features['P3'], features['P4'], features['P5']

        # Buoc 1: VL cross-attention tai moi scale doc lap
        P3_vl = self.vl_P3(P3, text_feats)
        P4_vl = self.vl_P4(P4, text_feats)
        P5_vl = self.vl_P5(P5, text_feats)

        # Buoc 2: Top-down FPN merge
        # P4 = P4 + upsample(P5)
        P5_up = F.interpolate(P5_vl, size=P4_vl.shape[-2:], mode='nearest')
        P4_td = P4_vl + P5_up

        # P3 = P3 + upsample(P4_merged)
        P4_up = F.interpolate(P4_td, size=P3_vl.shape[-2:], mode='nearest')
        P3_td = P3_vl + P4_up

        # Buoc 3: Output conv de lam muot
        out_P3 = self.out_P3(P3_td)
        out_P4 = self.out_P4(P4_td)
        out_P5 = self.out_P5(P5_vl)

        return {'P3': out_P3, 'P4': out_P4, 'P5': out_P5}


# Alias de giu tuong thich voi code cu import DummyRepVLPAN
DummyRepVLPAN = FPN_VL_Neck
