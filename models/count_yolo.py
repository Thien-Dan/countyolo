import torch
import torch.nn as nn
from .backbone.resnet_backbone import ResNet50Backbone
from .neck.dummy_neck import FPN_VL_Neck
from .fusion.attribute_fusion import DummyAttributeFusion
from .heads.density_head import DensityHead
from .heads.nms_free_head import FCOSHead
from .language.clip_encoder import CLIPTextEncoder
from .language.exemplar_encoder import ExemplarEncoder


class CountYOLO(nn.Module):
    """
    CountYOLO v2 — Multi-scale architecture.

    Thay doi so voi v1 (Dummy modules):
      - Backbone: ResNet50 xuat multi-scale P3/P4/P5 (khong upsample na)
      - Neck: FPN_VL_Neck (real cross-attention + top-down FPN)
      - DensityHead: fuse P3+P4+P5 → density map
      - FCOSHead: predict tren P3 + P4 voi centerness branch

    Giu nguyen (novel contributions):
      - CLIPTextEncoder + ExemplarEncoder (CLIP joint space alignment)
      - DummyAttributeFusion (real cross-attention, chi doi ten)
      - LLM routing interface (texts / exemplar_crops / attr_texts)
    """

    def __init__(self, use_clip: bool = False, channels: int = 256):
        super().__init__()
        self.use_clip = use_clip
        self.channels = channels

        # Backbone: multi-scale P3/P4/P5
        self.backbone = ResNet50Backbone(out_channels=channels, pretrained=True)

        # Neck: FPN + VL cross-attention
        self.neck = FPN_VL_Neck(in_channels=channels, out_channels=channels, num_heads=8)

        # Attribute Fusion: cross-attention visual × exemplar/attr
        # Nhan P3 (resolution cao nhat) lam visual query
        self.fusion = DummyAttributeFusion(channels=channels)

        # Density Head: nhan full P3/P4/P5 dict
        self.density_head = DensityHead(in_channels=channels, mid_channels=128)

        # FCOS Head: nhan P3/P4 dict (P5 khong dung cho detection)
        self.box_head = FCOSHead(in_channels=channels, num_classes=1,
                                 num_convs=4, mid_channels=channels)

        if self.use_clip:
            self.text_encoder    = CLIPTextEncoder(out_channels=channels)
            self.exemplar_encoder = ExemplarEncoder(out_channels=channels)

    def freeze_modules(self, freeze_clip: bool = True, freeze_backbone: bool = True):
        """
        Dong bang trong so cho fine-tune.
        Dung backbone.unfreeze_deep() khi muon mo bang layer3/layer4 sau epoch N.
        """
        if freeze_clip and hasattr(self, 'text_encoder'):
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.exemplar_encoder.clip.parameters():
                param.requires_grad = False

        if freeze_backbone:
            # Dong bang toan bo backbone (stem, layer1-4, lateral convs)
            self.backbone.freeze()

    def unfreeze_backbone_deep(self):
        """
        Mo bang layer3 + layer4 (deep features) voi lr nho hon (curriculum).
        Goi tu train.py sau epoch N (vd epoch 6).
        """
        self.backbone.unfreeze_deep()

    def forward(self, images: torch.Tensor, text_feats=None, attr_feats=None,
                texts=None, exemplar_crops=None, attr_texts=None) -> dict:
        """
        Args:
            images:         (B, 3, H, W)
            texts:          List[str]  — category text cho VL neck
            exemplar_crops: (B, N_ex, 3, 224, 224) — [TRAIN] CLIP image exemplar
            attr_texts:     List[str]  — [INFER] LLM parsed attributes

        Returns:
            {
              'density_map': (B, 1, H, W),
              'cls_scores':  {'P3': (B,1,H/8,W/8),  'P4': (B,1,H/16,W/16)},
              'box_preds':   {'P3': (B,4,H/8,W/8),  'P4': (B,4,H/16,W/16)},
              'centerness':  {'P3': (B,1,H/8,W/8),  'P4': (B,1,H/16,W/16)},
            }
        """
        device = images.device
        B      = images.size(0)

        # ── 1. Text Features cho Neck ─────────────────────────────────────────
        if self.use_clip and texts is not None:
            t_feats  = self.text_encoder(texts, device)          # (N_texts, C)
            # Expand sang (B, N_texts, C) — moi anh trong batch chia se cung text
            text_feats = t_feats.unsqueeze(0).expand(B, -1, -1)

        if text_feats is None:
            text_feats = torch.zeros((B, 1, self.channels), device=device)

        # ── 2. Attribute Features cho Fusion ─────────────────────────────────
        if self.use_clip and exemplar_crops is not None and attr_feats is None:
            attr_feats = self.exemplar_encoder(exemplar_crops)   # (B, N_ex, C)

        elif self.use_clip and attr_texts is not None and attr_feats is None:
            a_feats    = self.text_encoder(attr_texts, device)   # (N_attrs, C)
            attr_feats = a_feats.unsqueeze(0).expand(B, -1, -1) # (B, N_attrs, C)

        if attr_feats is None:
            attr_feats = torch.zeros((B, 1, self.channels), device=device)

        # ── 3. Backbone: multi-scale features ────────────────────────────────
        backbone_feats = self.backbone(images)   # {'P3', 'P4', 'P5'}

        # ── 4. FPN Neck: VL cross-attention + top-down merge ─────────────────
        neck_feats = self.neck(backbone_feats, text_feats)   # {'P3', 'P4', 'P5'}

        # ── 5. Attribute Fusion tren P3 (resolution cao nhat) ─────────────────
        # Sau fusion, P3 co ca visual va exemplar/attr context
        fused_P3 = self.fusion(neck_feats['P3'], attr_feats)  # (B, C, H/8, W/8)

        # Xay dung lai dict voi P3 da fused
        fused_feats = {'P3': fused_P3, 'P4': neck_feats['P4'], 'P5': neck_feats['P5']}

        # ── 6. Heads ──────────────────────────────────────────────────────────
        density_map  = self.density_head(fused_feats)     # (B, 1, H, W)
        head_outputs = self.box_head(fused_feats)          # dict P3/P4 predictions

        return {
            'density_map': density_map,
            'cls_scores':  head_outputs['cls_scores'],
            'box_preds':   head_outputs['box_preds'],
            'centerness':  head_outputs['centerness'],
        }
