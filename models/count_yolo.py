import torch
import torch.nn as nn
from .backbone.resnet_backbone import ResNet50Backbone
from .neck.dummy_neck import DummyRepVLPAN
from .fusion.attribute_fusion import DummyAttributeFusion
from .heads.density_head import DummyDensityHead
from .heads.nms_free_head import DummyNMSFreeHead
from .language.clip_encoder import CLIPTextEncoder
from .language.exemplar_encoder import ExemplarEncoder

class CountYOLO(nn.Module):
    """Mô hình tổng thể CountYOLO v4 Skeleton tích hợp CLIP."""
    def __init__(self, use_clip=False):
        super().__init__()
        self.use_clip = use_clip
        # In_channels mặc định là 3 (RGB), out của backbone quy chuẩn là 256
        self.backbone = ResNet50Backbone(in_channels=3, out_channels=256, pretrained=True)
        
        # Neck kết hợp thông tin
        self.neck = DummyRepVLPAN(in_channels=256, out_channels=256)
        
        # Attribute Fusion
        self.fusion = DummyAttributeFusion(channels=256)
        
        # Heads
        self.density_head = DummyDensityHead(in_channels=256)
        self.box_head = DummyNMSFreeHead(in_channels=256, num_classes=1)
        
        if self.use_clip:
            self.text_encoder = CLIPTextEncoder(out_channels=256)
            # ExemplarEncoder dùng chung CLIP model_name và cùng out_channels
            # → train với ảnh exemplar, infer với text LLM đều cùng 256-dim space
            self.exemplar_encoder = ExemplarEncoder(out_channels=256)

    def freeze_modules(self, freeze_clip=True, freeze_backbone=True):
        """
        Đóng băng trọng số để Fine-Tune (Tối ưu VRAM).
        """
        if freeze_clip and hasattr(self, 'text_encoder'):
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                
        if freeze_backbone:
            # Chỉ freeze các layer trích xuất đặc trưng của ResNet
            # Giữ lại lớp giảm channel `reduction_conv` để học feature phù hợp với CountYOLO
            for name, param in self.backbone.named_parameters():
                if "reduction" not in name:
                    param.requires_grad = False

    def forward(self, images, text_feats=None, attr_feats=None, texts=None,
                exemplar_crops=None, attr_texts=None):
        """
        Args:
            images:          Tensor (B, 3, H, W)
            text_feats:      Tensor (B, N, C)          - Pre-computed text embeddings
            attr_feats:      Tensor (B, N, C)          - Pre-computed attribute embeddings
            texts:           List[str]                 - Category text (Train & Infer)
            exemplar_crops:  Tensor (B, N_ex, 3, H, W) - Visual exemplar crops (TRAIN only)
            attr_texts:      List[str]                 - LLM parsed attributes (INFER only)
                             Ví dụ: ["red", "parked", "small"] do LLM trích xuất từ câu hỏi

        Thứ tự ưu tiên cho attr_feats:
            1. exemplar_crops → ExemplarEncoder (CLIP Image, 512-dim) → (B, N_ex, 256)  [TRAIN]
            2. attr_texts     → CLIPTextEncoder (CLIP Text,  512-dim) → (B, N_at, 256)  [INFER-LLM]
            3. attr_feats     → Dùng trực tiếp nếu caller đã tự encode                     [ADVANCED]
            4. zeros fallback → (B, 1, 256)                                              [FALLBACK]

        Cả (1) và (2) đều qua CLIP joint space (512-dim) trước projection → Đảm bảo khoa học align.
        """
        device = images.device
        B = images.size(0)

        # ── 1. Text Features (cho Neck Vision-Language) ──────────────────────────────────
        if self.use_clip and texts is not None:
            text_feats_clip = self.text_encoder(texts, device)      # (N, 256)
            text_feats = text_feats_clip.unsqueeze(0).expand(B, -1, -1)  # (B, N, 256)

        if text_feats is None:
            text_feats = torch.zeros((B, 1, 256), device=device)

        # ── 2. Attribute Features (cho AttributeFusion) ───────────────────────────────────
        if self.use_clip and exemplar_crops is not None and attr_feats is None:
            # TRAIN: ExemplarEncoder dùng CLIP Image get_image_features() → 512-dim
            attr_feats = self.exemplar_encoder(exemplar_crops)          # (B, N_ex, 256)

        elif self.use_clip and attr_texts is not None and attr_feats is None:
            # INFER (LLM): CLIPTextEncoder dùng CLIP Text get_text_features() → 512-dim
            # Cùng joint space với ExemplarEncoder → AttributeFusion nhận được kiểu data tương tự
            attr_feats_raw = self.text_encoder(attr_texts, device)       # (N_at, 256)
            attr_feats = attr_feats_raw.unsqueeze(0).expand(B, -1, -1)  # (B, N_at, 256)

        if attr_feats is None:
            attr_feats = torch.zeros((B, 1, 256), device=device)    # Fallback an toàn
            
        # 1. Trích xuất đặc trưng ảnh
        feat = self.backbone(images)

        # 2. Tương tác Vision-Language
        feat = self.neck(feat, text_feats)

        # 3. Fuse với các visual attributes
        # Đặt tên feat_fused để tách biệt rõ feature trước và sau fusion
        feat_fused = self.fusion(feat, attr_feats)

        # 4. Đầu ra: density_head và box_head đều được feed feat_fused
        density_map = self.density_head(feat_fused)
        cls_scores, box_preds = self.box_head(feat_fused)

        return {
            "density_map": density_map,
            "cls_scores": cls_scores,
            "box_preds": box_preds
        }
