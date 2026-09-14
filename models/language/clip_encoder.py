import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

# Kích thước embedding của CLIP joint space (ViT-B/32)
# get_text_features() → 512-dim (CLIP projection space, cùng với get_image_features())
CLIP_PROJECTION_DIM = 512

class CLIPTextEncoder(nn.Module):
    """
    Trích xuất đặc trưng văn bản dùng CLIP Text Encoder.

    **Dùng get_text_features() (512-dim)** thay vì pooler_output (768-dim) để đảm bảo
    cùng không gian embedding với ExemplarEncoder.get_image_features() (cũng 512-dim).

    Luồng:
        Train  : texts = ["person", "car"]      → (N, 256)  ← text_feats cho Neck
        Infer  : texts = LLM parsed category    → (N, 256)  ← text_feats cho Neck
        Infer  : attr_texts = ["red", "parked"] → (N, 256)  ← attr_feats cho AttributeFusion
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", out_channels: int = 256):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        # Tải full CLIPModel để dùng get_text_features() → đúng 512-dim joint space
        self.clip = CLIPModel.from_pretrained(model_name)

        # Đóng băng toàn bộ CLIP
        for param in self.clip.parameters():
            param.requires_grad = False

        # Projection: 512 (CLIP joint) → 256 (CountYOLO internal dim)
        self.projection = nn.Sequential(
            nn.Linear(CLIP_PROJECTION_DIM, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )

    def forward(self, texts: list, device: torch.device) -> torch.Tensor:
        """
        Args:
            texts:  List[str] — ["bird", "car"] hoặc ["red", "parked"] (LLM attributes)
            device: Device hiện tại
        Returns:
            feats: Tensor (len(texts), out_channels)
        """
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids      = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        self.clip.eval()
        with torch.no_grad():
            # get_text_features() trả về 512-dim — đúng CLIP joint embedding space
            text_feats = self.clip.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )  # (N, 512)

        return self.projection(text_feats)  # (N, 256)
