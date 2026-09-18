import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

# Kích thước embedding của CLIP joint space (ViT-B/32)
# get_image_features() → 512-dim — CÙNG không gian với get_text_features()
CLIP_PROJECTION_DIM = 512

# CLIP ViT-B/32 image normalization constants (xac nhan tu CLIPProcessor)
# do_normalize=True: pixel = (pixel - mean) / std
# Input cua ExemplarEncoder la [0,1] tensor (tu TF.to_tensor / F.interpolate),
# can normalize truoc khi truyen vao get_image_features().
CLIP_IMG_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_IMG_STD  = [0.26862954, 0.26130258, 0.27577711]


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
                          Crop exemplar [0,1] float tensor (output cua TF.to_tensor / F.interpolate)
        Returns:
            exemplar_feats: (B, N_ex, out_channels)
        """
        B, N_ex, C, H, W = crop_tensors.shape

        # Flatten de xu ly song song
        flat_crops = crop_tensors.view(B * N_ex, C, H, W)  # (B*N_ex, 3, H, W)

        # Normalize theo CLIP mean/std truoc khi truyen vao ViT.
        # TF.to_tensor() / F.interpolate cho ra [0,1]. CLIP yeu cau normalize:
        #   pixel = (pixel - mean) / std
        mean = torch.tensor(CLIP_IMG_MEAN, device=flat_crops.device).view(1, 3, 1, 1)
        std  = torch.tensor(CLIP_IMG_STD,  device=flat_crops.device).view(1, 3, 1, 1)
        flat_crops = (flat_crops - mean) / std

        self.clip.eval()
        with torch.no_grad():
            image_feats = self.clip.get_image_features(pixel_values=flat_crops)
            
            # Xu ly cac phien ban transformers tra ve object thay vi tensor
            if not isinstance(image_feats, torch.Tensor):
                if hasattr(image_feats, 'image_embeds'):
                    image_feats = image_feats.image_embeds
                elif hasattr(image_feats, 'pooler_output'):
                    image_feats = image_feats.pooler_output
                elif isinstance(image_feats, (tuple, list)):
                    image_feats = image_feats[0]

        projected = self.projection(image_feats)  # (B*N_ex, 256)
        return projected.view(B, N_ex, -1)          # (B, N_ex, 256)
