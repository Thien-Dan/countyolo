import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

# Kích thước embedding của CLIP joint space (ViT-B/32)
# get_image_features() → 512-dim — CÙNG không gian với get_text_features()
CLIP_PROJECTION_DIM = 512

class ExemplarEncoder(nn.Module):
    """
    Mã hóa ảnh exemplar (visual crops) bằng CLIP Image Encoder.

    **Dùng get_image_features() (512-dim)** — đúng CLIP joint embedding space,
    cùng với CLIPTextEncoder.get_text_features() (cũng 512-dim).

    Tại sao điều này đảm bảo Train/Infer khớp nhau:
        Train  : ExemplarEncoder(crop_ảnh)     → (B, N_ex, 256)  ← Visual CLIP space
        Infer  : CLIPTextEncoder(LLM_attrs)    → (B, N_attrs, 256) ← Text CLIP space
        CLIP pre-training đảm bảo cả 2 đều gần nhau trong 512-dim trước projection.
        Sau projection 512→256 (cùng architecture), không gian vẫn tương đồng.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", out_channels: int = 256):
        super().__init__()

        # Full CLIPModel để dùng get_image_features() → 512-dim joint space
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Đóng băng toàn bộ CLIP Vision
        for param in self.clip.parameters():
            param.requires_grad = False

        # Projection: 512 (CLIP joint) → 256 (CountYOLO internal dim)
        # CÙNG architecture với CLIPTextEncoder.projection → tương đồng space sau projection
        self.projection = nn.Sequential(
            nn.Linear(CLIP_PROJECTION_DIM, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )

    def forward(self, crop_tensors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            crop_tensors: (B, N_ex, 3, H_crop, W_crop)
                          Ảnh crop exemplar đã resize về kích thước CLIP (224×224)
        Returns:
            exemplar_feats: (B, N_ex, out_channels)
        """
        B, N_ex, C, H, W = crop_tensors.shape

        # Flatten để xử lý song song
        flat_crops = crop_tensors.view(B * N_ex, C, H, W)  # (B*N_ex, 3, H, W)

        self.clip.eval()
        with torch.no_grad():
            # get_image_features() → 512-dim CLIP joint space (cùng với get_text_features)
            image_feats = self.clip.get_image_features(pixel_values=flat_crops)  # (B*N_ex, 512)

        projected = self.projection(image_feats)  # (B*N_ex, 256)
        return projected.view(B, N_ex, -1)          # (B, N_ex, 256)
