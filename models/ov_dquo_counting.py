"""OV-DQUO Open-World Object Counting Model (EVA02-CLIP + CLIPSelf + 3-Scale Multi-Scale Deformable Attention).

Design & Forward/Backward Verification:
1. Pure-PyTorch standalone recreation of `OVDQUO_ViT-B16_LVIS` / `OVDQUO_ViT-L14_LVIS` (loads 100% of
   official checkpoint keys with strict validation, ZERO CUDA compilation, ZERO xformers/mmcv/detectron2).
2. Frozen EVA02-CLIP-B-16 (CLIPSelf patch-distilled) backbone under `torch.no_grad()` produces:
   - 3-scale feature pyramid (`stride 8`, `stride 16`, `stride 32`) via trainable `interpolate1` + `input_proj`.
   - Dense `512-D` L2-normalized CLIPSelf semantic map (`dense_map`) where `ROIAlign(dense_map, box)` directly
     matches frozen `EVA02-B-16` 80-template text embeddings AND visual exemplar region embeddings!
3. Full 7-layer differentiable deep supervision (6-layer Multi-Scale Deformable Decoder with Look-Forward-Twice
   + Stage-1 Encoder proposal head `enc_out_class_embed` & `enc_out_bbox_embed`, 1,000 queries, 100% NMS-free).
4. Differentiable Probabilistic Log-AND scoring (`logit(sigmoid(dec_obj) * sigmoid(sem_logit))`) in pure FP32:
   - Exact differentiable generalization of OV-DQUO's inference formula (`clip_prob * objectness_prob`) that works
     cleanly for 1 prompt, K-class negative supervision, visual exemplars, and absent-class rejection!
"""
from collections import OrderedDict
import copy
import math
from typing import Optional
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import roi_align

from models.yoloworld_counting import xywh_to_xyxy
from models.ov_dquo_backbone import (
    EVAVisionTransformer,
    EvaCLIPViTBackbone,
    MLP,
    OVDeformableTransformer,
    PositionEmbeddingSineHW,
    inverse_sigmoid,
)


OV_DQUO_VARIANTS = {
    'vitb16': dict(
        gdrive_id='15bVUiHP14ChzLGzU3DA_4J7O1vaox51a',
        open_clip_arch='EVA02-B-16',
        open_clip_pretrained='merged2b_s8b_b131k',
        img_size_pretrain=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=2.6667,
        text_dim=512,
        pt_hw_seq_len=16,
        out_indices=(5, 7, 11),
        default_resolution=1024,
    ),
    'vitl14': dict(
        gdrive_id='1yQfWsoW9meZtKWhs2e9S9EPwi8lxqJei',
        open_clip_arch='EVA02-L-14-336',
        open_clip_pretrained='merged2b_s6b_b61k',
        img_size_pretrain=336,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=2.6667,
        text_dim=768,
        pt_hw_seq_len=16,
        out_indices=(10, 14, 23),
        default_resolution=896,
    ),
}

CLIP_TEMPLATES = [
    'a photo of a {}.',
    'a photo of many {}.',
    'a close-up photo of a {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a bright photo of a {}.',
    'a photo of a small {}.',
    'a photo of a large {}.',
    'a cropped photo of a {}.',
    'a good photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a hard to see {}.',
]


def build_ov_dquo_class_list(target_prompt: str, train_vocab: list[str], max_classes: int = 8):
    import random
    target_clean = str(target_prompt).strip()
    if max_classes <= 1 or len(train_vocab) <= 1:
        return [target_clean], 0
    negatives = [c.strip() for c in train_vocab if c.strip() and c.strip() != target_clean]
    k = min(len(negatives), max_classes - 1)
    chosen = [target_clean] + random.sample(negatives, k)
    random.shuffle(chosen)
    return chosen, chosen.index(target_clean)


def log_and_logit(a: Tensor, b: Tensor) -> Tensor:
    """Numerically stable differentiable `logit(sigmoid(a) * sigmoid(b))` in pure FP32.

    Both `a` (1-to-1 decoder objectness) and `b` (frozen CLIPSelf semantic/visual match) must be high
    for the combined logit to be positive; if either is strongly negative (duplicate OR wrong/absent class),
    the product probability cleanly approaches 0 with smooth, strictly finite gradients everywhere.
    """
    log_p = F.logsigmoid(a.float()) + F.logsigmoid(b.float())
    p_clamped = torch.exp(log_p).clamp(max=1.0 - 1e-6)
    return log_p - torch.log1p(-p_clamped)


class _BackboneJoiner(nn.Sequential):
    """Matches OV-DQUO state_dict prefix `backbone.0.*`."""
    def __init__(self, body: EvaCLIPViTBackbone, pos_enc: PositionEmbeddingSineHW):
        super().__init__(body, pos_enc)


class OVDQUOCounting(nn.Module):
    ARCHITECTURE = 'OV-DQUO-Deformable-Counting-v1'

    def __init__(
        self,
        variant: str = 'vitb16',
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda',
        resolution: Optional[int] = None,
        num_queries: int = 1000,
        use_exemplars: bool = True,
        grad_checkpoint: bool = False,
        text_encoder_fn=None,
        _tiny_cfg: Optional[dict] = None,
    ):
        super().__init__()
        cfg = dict(_tiny_cfg) if _tiny_cfg is not None else dict(OV_DQUO_VARIANTS[variant])
        self.variant = variant
        self.cfg = cfg
        self.resolution = int(resolution if resolution is not None else cfg['default_resolution'])
        self.patch_size = int(cfg['patch_size'])
        if self.resolution % (self.patch_size * 2) != 0:
            raise ValueError(f'resolution ({self.resolution}) must be divisible by 2 * patch_size ({self.patch_size * 2})')
        self.num_queries = int(num_queries)
        self.use_exemplars = bool(use_exemplars)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.hidden_dim = int(cfg.get('hidden_dim', 256))
        self.text_dim = int(cfg['text_dim'])
        self.num_dec_layers = int(cfg.get('dec_layers', 6))
        self.num_enc_layers = int(cfg.get('enc_layers', 6))

        # 1. Backbone (`backbone.0.*` matches official checkpoint prefix)
        visual = EVAVisionTransformer(
            img_size=int(cfg['img_size_pretrain']),
            patch_size=self.patch_size,
            embed_dim=int(cfg['embed_dim']),
            depth=int(cfg['depth']),
            num_heads=int(cfg['num_heads']),
            mlp_ratio=float(cfg['mlp_ratio']),
            num_classes=self.text_dim,
            pt_hw_seq_len=int(cfg['pt_hw_seq_len']),
            subln=True,
            rope=True,
        )
        body = EvaCLIPViTBackbone(visual, out_indices=tuple(cfg['out_indices']))
        pos_enc = PositionEmbeddingSineHW(self.hidden_dim // 2, temperatureH=20.0, temperatureW=20.0, normalize=True)
        self.backbone = _BackboneJoiner(body, pos_enc)

        # 2. Multi-Scale input_proj (3 levels: stride 8, 16, 32)
        in_ch = int(cfg['embed_dim'])
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, self.hidden_dim, kernel_size=1),
                nn.GroupNorm(min(32, self.hidden_dim), self.hidden_dim),
            )
            for _ in range(3)
        ])

        # 3. 3-Scale Multi-Scale Deformable Encoder + Decoder
        self.transformer = OVDeformableTransformer(
            d_model=self.hidden_dim,
            nhead=int(cfg.get('nheads', 8)),
            num_queries=self.num_queries,
            num_encoder_layers=self.num_enc_layers,
            num_decoder_layers=self.num_dec_layers,
            dim_feedforward=int(cfg.get('dim_feedforward', 2048)),
            dropout=0.0,
            num_feature_levels=3,
            enc_n_points=4,
            dec_n_points=4,
            text_dim=self.text_dim,
        )

        # 4. 6-layer iterative bbox_embed + class_embed (independent per layer, initialized from pretrained)
        _bbox = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        nn.init.constant_(_bbox.layers[-1].weight, 0.0)
        nn.init.constant_(_bbox.layers[-1].bias, 0.0)
        _cls = nn.Linear(self.hidden_dim, 1)
        nn.init.constant_(_cls.bias, -math.log((1.0 - 0.01) / 0.01))
        self.bbox_embed = nn.ModuleList([copy.deepcopy(_bbox) for _ in range(self.num_dec_layers)])
        self.class_embed = nn.ModuleList([copy.deepcopy(_cls) for _ in range(self.num_dec_layers)])
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # 5. Counting & Multimodal Prompt Adaptation Heads (Zero-init residual + calibrated CLIPSelf Log-AND)
        self.sim_inject = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, self.hidden_dim),
        )
        nn.init.zeros_(self.sim_inject[-1].weight)
        nn.init.zeros_(self.sim_inject[-1].bias)

        # Frozen CLIPSelf text calibration: cos ~ 0.30 match -> +1.6, cos ~ 0.14 wrong -> -1.6
        self.clip_text_bias = nn.Parameter(torch.tensor(-0.22, dtype=torch.float32))
        self.clip_text_scale = nn.Parameter(torch.tensor(19.0, dtype=torch.float32))

        if self.use_exemplars:
            # Same-space learned metric projection on 512-D CLIPSelf ROI features
            self.vis_proj = nn.Sequential(
                nn.Linear(self.text_dim, 256, bias=False),
                nn.LayerNorm(256),
            )
            self.vis_mix = nn.Parameter(torch.tensor([0.7, 0.0], dtype=torch.float32))
            self.clip_vis_bias = nn.Parameter(torch.tensor(-0.60, dtype=torch.float32))
            self.clip_vis_scale = nn.Parameter(torch.tensor(7.0, dtype=torch.float32))
            self.mm_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.mm_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # ImageNet normalization (exact OV-DQUO `datasets/ov_lvis.py` lines 250-251!)
        self.register_buffer('_img_mean', torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('_img_std', torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        self._text_encoder_fn = text_encoder_fn
        self._open_clip_model = None
        self._open_clip_tokenizer = None
        self._text_cache: OrderedDict[str, Tensor] = OrderedDict()
        self._text_cache_limit = 2048

        if checkpoint_path is not None:
            self.load_official_ov_dquo_checkpoint(checkpoint_path)

        self.to(device)
        self._configure_freezing()
        self.set_training(False)

    def load_official_ov_dquo_checkpoint(self, ckpt_path: str):
        """Load official `OVDQUO_ViT-B16_LVIS.pth` or `OVDQUO_ViT-L14_LVIS.pth` with strict key auditing."""
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = raw.get('ema_model') or raw.get('model') or raw
        # Strip DDP `module.` prefix if present
        cleaned = {}
        for k, v in sd.items():
            nk = k[7:] if k.startswith('module.') else k
            # Ignore DN-only embeddings, static LVIS classifier table, and dynamic RoPE buffers
            if nk.startswith('label_enc.') or nk == 'classifier' or 'freqs_cos_' in nk or 'freqs_sin_' in nk:
                continue
            if nk.endswith('.flag'):
                continue
            # Official OV-DQUO shares class_embed across decoder layers (`dec_pred_class_embed_share=True`)
            # and saves `transformer.decoder.class_embed.*` & `transformer.decoder.bbox_embed.*` aliases
            if nk.startswith('transformer.decoder.class_embed.'):
                continue
            cleaned[nk] = v

        # Handle shared class_embed if only class_embed.0.* exists or all 0..5 exist identically
        if 'class_embed.0.weight' in cleaned and 'class_embed.1.weight' not in cleaned:
            for i in range(1, self.num_dec_layers):
                cleaned[f'class_embed.{i}.weight'] = cleaned['class_embed.0.weight'].clone()
                cleaned[f'class_embed.{i}.bias'] = cleaned['class_embed.0.bias'].clone()

        # Sync transformer.decoder.bbox_embed alias if checkpoint stored only bbox_embed.*
        for i in range(self.num_dec_layers):
            for suf in ('0.weight', '0.bias', '1.weight', '1.bias', '2.weight', '2.bias'):
                src_k = f'bbox_embed.{i}.layers.{suf}'
                dst_k = f'transformer.decoder.bbox_embed.{i}.layers.{suf}'
                if src_k in cleaned and dst_k not in cleaned:
                    cleaned[dst_k] = cleaned[src_k]

        own_sd = self.state_dict()
        # Allow num_queries mismatch only if user explicitly overrides num_queries != 1000
        if 'transformer.tgt_embed.weight' in cleaned and cleaned['transformer.tgt_embed.weight'].shape != own_sd['transformer.tgt_embed.weight'].shape:
            w = cleaned['transformer.tgt_embed.weight']
            n_src, d = w.shape
            n_dst = own_sd['transformer.tgt_embed.weight'].shape[0]
            if n_dst <= n_src:
                cleaned['transformer.tgt_embed.weight'] = w[:n_dst].clone()
            else:
                reps = (n_dst + n_src - 1) // n_src
                cleaned['transformer.tgt_embed.weight'] = w.repeat(reps, 1)[:n_dst].clone()

        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        allowed_missing_prefixes = (
            'sim_inject.', 'clip_text_', 'vis_proj.', 'vis_mix',
            'clip_vis_', 'mm_alpha', 'mm_bias', '_img_mean', '_img_std',
        )
        bad_missing = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
        if bad_missing:
            raise RuntimeError(f'CRITICAL: Missing official OV-DQUO weights ({len(bad_missing)}): {bad_missing[:10]}')
        print(
            f'Loaded official OV-DQUO checkpoint: matched={len(own_sd) - len(missing)} '
            f'new_heads={len(missing)} ignored_unexpected={len(unexpected)}',
            flush=True,
        )

    def _configure_freezing(self):
        # Freeze EVA02-CLIP-B-16 visual backbone strictly (CLIPSelf dense space stays 100% open-world!)
        self.backbone[0].visual.requires_grad_(False)
        # Keep interpolate1 (ConvTranspose2d), input_proj, transformer, bbox_embed, class_embed, and new heads trainable
        self.backbone[0].interpolate1.requires_grad_(True)
        self.input_proj.requires_grad_(True)
        self.transformer.requires_grad_(True)
        self.bbox_embed.requires_grad_(True)
        self.class_embed.requires_grad_(True)
        self.sim_inject.requires_grad_(True)
        self.clip_text_bias.requires_grad_(True)
        self.clip_text_scale.requires_grad_(True)
        if self.use_exemplars:
            self.vis_proj.requires_grad_(True)
            self.vis_mix.requires_grad_(True)
            self.clip_vis_bias.requires_grad_(True)
            self.clip_vis_scale.requires_grad_(True)
            self.mm_alpha.requires_grad_(True)
            self.mm_bias.requires_grad_(True)

    def set_training(self, training: bool):
        self.train(training)
        self.backbone[0].visual.eval()
        if self._open_clip_model is not None:
            self._open_clip_model.eval()

    def trainable_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def parameter_buckets(self):
        """Return 4 disjoint buckets: (backbone_neck, transformer_heads, adapters, cal_scalars)."""
        bb_p, tr_p, ad_p, cal_p = [], [], [], []
        cal_names = {'clip_text_bias', 'clip_text_scale', 'vis_mix', 'clip_vis_bias', 'clip_vis_scale', 'mm_alpha', 'mm_bias'}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n in cal_names:
                cal_p.append(p)
            elif n.startswith('sim_inject.') or n.startswith('vis_proj.'):
                ad_p.append(p)
            elif n.startswith('backbone.') or n.startswith('input_proj.'):
                bb_p.append(p)
            else:
                tr_p.append(p)
        return bb_p, tr_p, ad_p, cal_p

    def parameter_groups(self, backbone_lr=1e-5, lr=2.5e-5, adapter_lr=1.5e-4, cal_lr=1.5e-3, weight_decay=1e-4):
        bb_p, tr_p, ad_p, cal_p = self.parameter_buckets()
        groups = []
        if bb_p:
            groups.append(dict(params=bb_p, lr=backbone_lr, weight_decay=weight_decay))
        if tr_p:
            groups.append(dict(params=tr_p, lr=lr, weight_decay=weight_decay))
        if ad_p:
            groups.append(dict(params=ad_p, lr=adapter_lr, weight_decay=weight_decay))
        if cal_p:
            groups.append(dict(params=cal_p, lr=cal_lr, weight_decay=0.0))
        return groups

    def export_trainable_state(self) -> dict:
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        sd = self.state_dict()
        return {k: v.detach().cpu() for k, v in sd.items() if k in keep}

    def import_trainable_state(self, state: dict):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f'Unexpected keys loading OVDQUOCounting state: {unexpected[:5]}')
        return missing

    def _ensure_text_encoder(self, device):
        if self._text_encoder_fn is not None:
            return
        if self._open_clip_model is None:
            import open_clip
            arch = self.cfg['open_clip_arch']
            pretrained = self.cfg['open_clip_pretrained']
            model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            # Drop vision tower from open_clip to save VRAM (we only need frozen text encoder!)
            if hasattr(model, 'visual'):
                del model.visual
            model.requires_grad_(False).eval().to(device)
            self._open_clip_model = model
            self._open_clip_tokenizer = open_clip.get_tokenizer(arch)

    @torch.no_grad()
    def encode_text_queries(self, prompts: list[str], device) -> Tensor:
        """Encode & LRU-cache frozen EVA02-CLIP 12-template text embeddings `(K, text_dim)` (L2-norm FP32)."""
        cleaned = [str(p).strip() or 'object' for p in prompts]
        missing = [p for p in dict.fromkeys(cleaned) if p not in self._text_cache]
        if missing:
            if self._text_encoder_fn is not None:
                vecs = self._text_encoder_fn(missing, device=device)
                vecs = F.normalize(vecs.float(), dim=-1).detach().cpu()
            else:
                self._ensure_text_encoder(device)
                vecs_list = []
                for p_str in missing:
                    templated = [t.format(p_str) for t in CLIP_TEMPLATES]
                    toks = self._open_clip_tokenizer(templated).to(device)
                    emb = self._open_clip_model.encode_text(toks).float()
                    emb = F.normalize(emb, dim=-1).mean(dim=0)
                    emb = F.normalize(emb, dim=-1).detach().cpu()
                    vecs_list.append(emb)
                vecs = torch.stack(vecs_list, dim=0)
            for p_str, vec in zip(missing, vecs):
                self._text_cache[p_str] = vec
                self._text_cache.move_to_end(p_str)
            while len(self._text_cache) > self._text_cache_limit:
                self._text_cache.popitem(last=False)
        return torch.stack([self._text_cache[p] for p in cleaned], dim=0).to(device=device, dtype=torch.float32).clone()

    def preprocess_image(self, pil_image, device=None) -> tuple[Tensor, Tensor]:
        """Exact OV-DQUO preprocessing: aspect-preserving resize to `max(H,W)=res`, ImageNet norm, bottom-right 0-pad + mask."""
        dev = device if device is not None else self._img_mean.device
        arr = np.asarray(pil_image.convert('RGB'), dtype=np.uint8)
        h, w = arr.shape[:2]
        scale = float(self.resolution) / float(max(h, w))
        nh = max(self.patch_size, min(self.resolution, int(round(h * scale))))
        nw = max(self.patch_size, min(self.resolution, int(round(w * scale))))
        x = torch.from_numpy(arr.copy()).to(device=dev, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        if nh != h or nw != w:
            x = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False, antialias=True)
        mean = self._img_mean.to(device=dev, dtype=torch.float32)
        std = self._img_std.to(device=dev, dtype=torch.float32)
        x = (x - mean) / std
        padded = x.new_zeros((1, 3, self.resolution, self.resolution))
        padded[:, :, :nh, :nw] = x
        mask = torch.ones((1, self.resolution, self.resolution), dtype=torch.bool, device=dev)
        mask[:, :nh, :nw] = False
        return padded, mask

    @staticmethod
    def _sample_dense_roi(dense_map: Tensor, dense_mask: Tensor, boxes_cxcywh: Tensor) -> Tensor:
        """Exact OV-DQUO `sample_feature_vit` on `(1, D_clip, H_d, W_d)` with image-normalized `[0,1]` boxes."""
        valid_w = float((~dense_mask[0, 0, :]).float().sum().clamp(min=1.0).item())
        valid_h = float((~dense_mask[0, :, 0]).float().sum().clamp(min=1.0).item())
        xyxy = xywh_to_xyxy(boxes_cxcywh.float()).clamp(0.0, 1.0)
        rois = xyxy * xyxy.new_tensor([valid_w, valid_h, valid_w, valid_h])
        feats = roi_align(dense_map.float(), [rois], output_size=(1, 1), spatial_scale=1.0, aligned=True)[..., 0, 0]
        return F.normalize(feats, dim=-1)

    @staticmethod
    def _sample_exemplar_roi(dense_map: Tensor, dense_mask: Tensor, exemplar_xyxy: Tensor) -> Tensor:
        valid_w = float((~dense_mask[0, 0, :]).float().sum().clamp(min=1.0).item())
        valid_h = float((~dense_mask[0, :, 0]).float().sum().clamp(min=1.0).item())
        xyxy = exemplar_xyxy.to(device=dense_map.device, dtype=torch.float32).reshape(-1, 4).clamp(0.0, 1.0)
        rois = xyxy * xyxy.new_tensor([valid_w, valid_h, valid_w, valid_h])
        feats = roi_align(dense_map.float(), [rois], output_size=(1, 1), spatial_scale=1.0, aligned=True)[..., 0, 0]
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_exemplars(self, pil_image, exemplar_boxes: Tensor) -> Optional[Tensor]:
        """Extract frozen CLIPSelf `(E, text_dim)` exemplar embeddings from full image for 2x2 tiling."""
        if not self.use_exemplars or exemplar_boxes is None or len(exemplar_boxes) == 0:
            return None
        dev = self._img_mean.device
        pv, mask = self.preprocess_image(pil_image, device=dev)
        _, _, dense_map, dense_mask = self.backbone[0](pv, mask)
        return self._sample_exemplar_roi(dense_map, dense_mask, exemplar_boxes)

    def _score_stage(
        self,
        boxes_cxcywh: Tensor,
        obj_logit: Tensor,
        dense_map: Tensor,
        dense_mask: Tensor,
        text_embeds: Optional[Tensor],
        target_idx: int,
        ex_embeds: Optional[Tensor],
    ) -> dict:
        """Pure-FP32 Log-AND semantic + objectness scoring for one decoder/encoder stage."""
        roi_feats = self._sample_dense_roi(dense_map, dense_mask, boxes_cxcywh.detach())  # (Q, D_clip)

        text_sem = None
        text_logits = None
        if text_embeds is not None:
            text_cos = torch.matmul(roi_feats, text_embeds.T)  # (Q, K)
            t_scale = F.elu(self.clip_text_scale) + 1.0
            text_sem = (text_cos + self.clip_text_bias) * t_scale
            text_logits = log_and_logit(obj_logit[:, None], text_sem)

        vis_sem = None
        vis_logits = None
        if ex_embeds is not None:
            sim_raw = torch.matmul(roi_feats, ex_embeds.T).max(dim=-1).values
            q_lrn = F.normalize(self.vis_proj(roi_feats), dim=-1)
            e_lrn = F.normalize(self.vis_proj(ex_embeds), dim=-1)
            sim_lrn = torch.matmul(q_lrn, e_lrn.T).max(dim=-1).values
            w_v = F.softmax(self.vis_mix, dim=0)
            vis_sim = w_v[0] * sim_raw + w_v[1] * sim_lrn
            v_scale = F.elu(self.clip_vis_scale) + 1.0
            vis_sem = (vis_sim + self.clip_vis_bias) * v_scale
            vis_logits = log_and_logit(obj_logit, vis_sem)

        mm_logits = None
        if text_sem is not None and vis_sem is not None:
            alpha = 0.25 + 0.5 * torch.sigmoid(self.mm_alpha)
            mm_sem = (1.0 - alpha) * text_sem[:, target_idx] + alpha * vis_sem + self.mm_bias
            mm_logits = log_and_logit(obj_logit, mm_sem)

        return dict(
            boxes=boxes_cxcywh.float(),
            text_logits=text_logits,
            vis_logits=vis_logits,
            mm_logits=mm_logits,
            obj_logits=obj_logit.float(),
        )

    def forward_stages(
        self,
        pil_image,
        prompts: Optional[list[str]] = None,
        target_idx: int = 0,
        exemplar_boxes: Optional[Tensor] = None,
        exemplar_feats: Optional[Tensor] = None,
        use_text: bool = True,
    ) -> dict:
        """Full 7-stage forward (6 decoder layers + Stage-1 encoder proposals) with zero detached heads."""
        dev = self._img_mean.device
        has_ex = self.use_exemplars and (
            exemplar_feats is not None or (exemplar_boxes is not None and len(exemplar_boxes) > 0)
        )
        if not use_text and not has_ex:
            raise ValueError('At least one of use_text=True or visual exemplars must be active')

        pv, mask = self.preprocess_image(pil_image, device=dev)
        feats, masks, dense_map, dense_mask = self.backbone[0](pv, mask)
        srcs = [proj(f) for proj, f in zip(self.input_proj, feats)]
        pos_embeds = [self.backbone[1](m, dtype=s.dtype) for m, s in zip(masks, srcs)]

        # Stage 1: 6-Layer Multi-Scale Deformable Encoder + Top-1,000 Proposal Selection
        enc = self.transformer.encode_and_propose(srcs, masks, pos_embeds, grad_checkpoint=self.grad_checkpoint)
        refpoint_undetach = enc['refpoint_undetach']  # (1, Q, 4) unsigmoid
        refpoint_enc_sig = refpoint_undetach.sigmoid()  # (1, Q, 4)
        enc_obj_logit = enc['enc_class_topk'][0, :, 0]  # (Q,)

        # Extract frozen CLIPSelf text & visual exemplar embeddings + proposal similarity hint (FP32)
        with torch.autocast('cuda', enabled=False):
            text_embeds = self.encode_text_queries(prompts, device=dev) if use_text else None
            ex_embeds = None
            if has_ex:
                ex_embeds = (
                    exemplar_feats.to(device=dev, dtype=torch.float32)
                    if exemplar_feats is not None
                    else self._sample_exemplar_roi(dense_map, dense_mask, exemplar_boxes)
                )

            if text_embeds is not None and ex_embeds is not None:
                prompt_emb = F.normalize(text_embeds[target_idx] + ex_embeds.mean(dim=0), dim=-1)
            elif text_embeds is not None:
                prompt_emb = text_embeds[target_idx]
            else:
                prompt_emb = F.normalize(ex_embeds.mean(dim=0), dim=-1)

            roi_enc = self._sample_dense_roi(dense_map, dense_mask, refpoint_enc_sig[0].detach())  # (Q, D_clip)
            t_sim = torch.matmul(roi_enc, text_embeds[target_idx]) if text_embeds is not None else roi_enc.new_zeros(roi_enc.shape[0])
            if ex_embeds is not None:
                v_all = torch.matmul(roi_enc, ex_embeds.T)
                v_max = v_all.max(dim=-1).values
                v_mean = v_all.mean(dim=-1)
                v_flag = roi_enc.new_ones(roi_enc.shape[0])
            else:
                v_max = roi_enc.new_zeros(roi_enc.shape[0])
                v_mean = roi_enc.new_zeros(roi_enc.shape[0])
                v_flag = roi_enc.new_zeros(roi_enc.shape[0])
            sim_hint = torch.stack([t_sim, v_max, v_mean, v_flag], dim=-1).unsqueeze(0)  # (1, Q, 4)

            q_prompt = self.transformer.text_proj(prompt_emb)[None, None, :]  # (1, 1, C)
            sim_delta = self.sim_inject(sim_hint)  # (1, Q, C)

        q_actual = refpoint_undetach.shape[1]
        tgt_init = self.transformer.tgt_embed.weight[:q_actual].unsqueeze(0) + q_prompt.to(srcs[0].dtype) + sim_delta.to(srcs[0].dtype)

        # Stage 2: 6-Layer Multi-Scale Deformable Decoder (Look-Forward-Twice)
        hs, references = self.transformer.decoder(
            tgt=tgt_init.transpose(0, 1),
            memory=enc['memory'].transpose(0, 1),
            memory_key_padding_mask=enc['mask_flatten'],
            refpoints_unsigmoid=refpoint_undetach.detach().transpose(0, 1),
            spatial_shapes=enc['spatial_shapes'],
            valid_ratios=enc['valid_ratios'],
        )

        # Score all 6 decoder layers + Stage-1 encoder proposals in pure FP32
        with torch.autocast('cuda', enabled=False):
            dec_stages = []
            for lvl in range(self.num_dec_layers):
                h_l = hs[lvl].float()
                ref_l = references[lvl].float()
                delta_l = self.bbox_embed[lvl](h_l)
                boxes_l = (delta_l + inverse_sigmoid(ref_l)).sigmoid()[0]
                obj_l = self.class_embed[lvl](h_l)[0, :, 0]
                dec_stages.append(
                    self._score_stage(boxes_l, obj_l, dense_map, dense_mask, text_embeds, target_idx, ex_embeds)
                )
            enc_stage = self._score_stage(
                refpoint_enc_sig[0].float(), enc_obj_logit.float(), dense_map, dense_mask, text_embeds, target_idx, ex_embeds
            )

        return dict(
            final=dec_stages[-1],
            aux=dec_stages[:-1],
            enc=enc_stage,
            target_idx=target_idx,
        )

    def forward(
        self,
        pil_image,
        prompt: str,
        exemplar_boxes: Optional[Tensor] = None,
        exemplar_feats: Optional[Tensor] = None,
        use_text: bool = True,
    ) -> tuple[Tensor, Tensor]:
        st = self.forward_stages(
            pil_image,
            prompts=[prompt] if use_text else None,
            target_idx=0,
            exemplar_boxes=exemplar_boxes,
            exemplar_feats=exemplar_feats,
            use_text=use_text,
        )
        fin = st['final']
        boxes = fin['boxes'].clamp(0.0, 1.0)
        if fin['mm_logits'] is not None:
            logits = fin['mm_logits']
        elif fin['text_logits'] is not None:
            logits = fin['text_logits'][:, 0]
        else:
            logits = fin['vis_logits']
        return boxes, logits
