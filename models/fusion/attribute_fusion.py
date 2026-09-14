import torch
import torch.nn as nn

class DummyAttributeFusion(nn.Module):
    """Mô phỏng Visual-Conditioned Attention Fusion."""
    def __init__(self, channels=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, visual_feats, attr_feats):
        # visual_feats: (B, C, H, W)
        # attr_feats: (B, N, C) - features của các attributes
        B, C, H, W = visual_feats.shape
        # Flatten HW cho attention
        v_flat = visual_feats.view(B, C, -1).permute(0, 2, 1) # (B, H*W, C)
        
        # Cross-attention (visual query attr)
        attn_out, _ = self.attn(query=v_flat, key=attr_feats, value=attr_feats)
        v_fused = self.norm1(v_flat + attn_out)
        
        # FFN
        ffn_out = self.ffn(v_fused)
        out = self.norm2(v_fused + ffn_out)
        
        # Reshape lại
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out
