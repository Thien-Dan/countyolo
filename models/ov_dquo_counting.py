"""OV-DQUO v2 Open-World Object Counting Model (EVA02-CLIP-L/14 or B/16 + CLIPSelf + 3-Scale MSDeformAttn).

v2 Upgrades over v1 (which already reached dev_mae ~ 19 on ViT-B/16):
1. Prompt-Conditioned Multi-Scale Memory Fusion (`PromptMemoryFusion`):
   - Extracts stride-8 (L0) + stride-16 (L1) visual exemplar tokens directly from the 6-layer
     Multi-Scale Deformable Encoder `memory` via pure-FP32 `ROIAlign` with exact `valid_ratios`.
   - Injects both same-space stride-8/16 exemplar similarity + upsampled CLIPSelf text/visual maps
     AND residual Cross-Attention (with learnable `null_kv` for background tokens) into `memory`
     BEFORE Stage-1 proposal selection and the 6-layer Deformable Decoder.
2. Prompt & Exemplar-Guided Stage-1 Proposal Selection:
   - Ranks encoder proposals using `enc_out_class_embed(output_memory) + enc_prompt_prior * prompt_sim_spatial`
     so all 1,000 decoder queries focus directly on the requested concept and visual exemplars from Step 0.
3. Multi-Scale Stride-8 Decoder-Space Exemplar Matching (`dec_vis_proj`):
   - Breaks the coarse `48x48` frozen `dense_map` ceiling on tiny dense objects by combining frozen CLIPSelf
     similarity with class-agnostic cosine similarity between decoder query states `hs[lvl]` (which attend to
     stride-8 features via 6-layer MSDeformAttn) and stride-8/16 exemplar memory tokens `ex_mem`!
4. Supports both `OVDQUO_ViT-L14_LVIS` (`39.3 AP_novel`, +9.6 AP over ViT-B/16!) and `OVDQUO_ViT-B16_LVIS`
   with strict checkpoint key verification, 7-layer differentiable Log-AND supervision, and 2x2 dense tiling.
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
        default_resolution=768,
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
        default_resolution=672,
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
    """Numerically stable differentiable `logit(sigmoid(a) * sigmoid(b))` in pure FP32."""
    log_p = F.logsigmoid(a.float()) + F.logsigmoid(b.float())
    p_clamped = torch.exp(log_p).clamp(max=1.0 - 1e-6)
    return log_p - torch.log1p(-p_clamped)


class PromptMemoryFusion(nn.Module):
    """Fuses Text Prompt + Multi-Scale Stride-8/16 Visual Exemplars into 3-scale encoder `memory` (pure FP32).

    Includes:
    - Multi-scale ROIAlign exemplar extraction from L0 (stride 8) & L1 (stride 16) encoder memory.
    - Same-space spatial exemplar similarity + upsampled CLIPSelf text/visual similarity injection.
    - Residual Cross-Attention with learnable `null_kv` token so background pixels stay uncontaminated.
    """
    def __init__(self, d_model: int = 256, nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        self.sim_proj = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
        )
        # Small non-zero init so Step-0 preserves pretrained memory while Step-1 gets immediate gradients
        nn.init.normal_(self.sim_proj[-1].weight, std=1e-3)
        nn.init.zeros_(self.sim_proj[-1].bias)

        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)
        self.null_kv = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_kv, std=0.02)
        # Bounded residual gate 0.5 * tanh(gate), init ~0.05 -> gentle at Step 0, full gradient on Step 1
        self.fuse_gate = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    @staticmethod
    def extract_memory_exemplars(
        memory_fp32: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        exemplar_boxes_img: Tensor,
    ) -> Tensor:
        """Extract stride-8 (L0) & stride-16 (L1) exemplar tokens `(E, d_model)` from encoder `memory`."""
        shapes = spatial_shapes.tolist()
        splits = [int(h * w) for h, w in shapes]
        lvl_feats = memory_fp32.split(splits, dim=1)
        xyxy = exemplar_boxes_img.to(device=memory_fp32.device, dtype=torch.float32).reshape(-1, 4).clamp(0.0, 1.0)
        ex_list = []
        # Pool from finest two scales (L0 stride-8 & L1 stride-16)
        for lvl in range(min(2, len(shapes))):
            h, w = shapes[lvl]
            vr_w = float(valid_ratios[0, lvl, 0].item()) * float(w)
            vr_h = float(valid_ratios[0, lvl, 1].item()) * float(h)
            fmap = lvl_feats[lvl].transpose(1, 2).reshape(1, -1, h, w).contiguous()
            rois = [xyxy * xyxy.new_tensor([vr_w, vr_h, vr_w, vr_h])]
            pooled = roi_align(fmap, rois, output_size=(1, 1), spatial_scale=1.0, aligned=True)[..., 0, 0]
            ex_list.append(pooled)
        return torch.stack(ex_list, dim=0).mean(dim=0)  # (E, d_model)

    def forward(
        self,
        memory: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        dense_map: Tensor,
        text_target_embed: Optional[Tensor],
        q_prompt_tok: Tensor,
        ex_clip: Optional[Tensor],
        ex_clip_proj: Optional[Tensor],
        ex_mem: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Return (fused_memory `(1, S, C)`, centered spatial prompt prior `(1, S)` for Stage-1 selection)."""
        orig_dtype = memory.dtype
        mem_fp32 = memory.float()
        bsz, seq_len, _ = mem_fp32.shape
        shapes = spatial_shapes.tolist()

        # 1. Upsample dense CLIPSelf text & visual similarity maps to all 3 encoder scales
        _, _, hd, wd = dense_map.shape
        dense_flat = dense_map.float().flatten(2).transpose(1, 2)  # (1, Hd*Wd, D_clip)

        if text_target_embed is not None:
            t_map_2d = torch.matmul(dense_flat, text_target_embed).reshape(1, 1, hd, wd)
            has_t = mem_fp32.new_ones((bsz, seq_len))
        else:
            t_map_2d = None
            has_t = mem_fp32.new_zeros((bsz, seq_len))

        if ex_clip is not None:
            v_map_2d = torch.matmul(dense_flat, ex_clip.T).max(dim=-1).values.reshape(1, 1, hd, wd)
            has_v = mem_fp32.new_ones((bsz, seq_len))
        else:
            v_map_2d = None
            has_v = mem_fp32.new_zeros((bsz, seq_len))

        t_up_list, v_up_list = [], []
        for h, w in shapes:
            if t_map_2d is not None:
                t_up_list.append(F.interpolate(t_map_2d, size=(h, w), mode='bilinear', align_corners=False).flatten(1))
            else:
                t_up_list.append(mem_fp32.new_zeros((bsz, h * w)))
            if v_map_2d is not None:
                v_up_list.append(F.interpolate(v_map_2d, size=(h, w), mode='bilinear', align_corners=False).flatten(1))
            else:
                v_up_list.append(mem_fp32.new_zeros((bsz, h * w)))
        t_sim_all = torch.cat(t_up_list, dim=1)  # (1, S)
        v_sim_all = torch.cat(v_up_list, dim=1)  # (1, S)

        # 2. Same-space stride-8/16 memory exemplar cosine similarity
        if ex_mem is not None:
            m_sim = torch.matmul(F.normalize(mem_fp32, dim=-1), F.normalize(ex_mem, dim=-1).T)  # (1, S, E)
            m_sim_max = m_sim.max(dim=-1).values
            m_sim_mean = m_sim.mean(dim=-1)
        else:
            m_sim_max = mem_fp32.new_zeros((bsz, seq_len))
            m_sim_mean = mem_fp32.new_zeros((bsz, seq_len))

        sim_feats = torch.stack([t_sim_all, v_sim_all, m_sim_max, m_sim_mean, has_t, has_v], dim=-1)  # (1, S, 6)
        sim_delta = self.sim_proj(sim_feats)

        # 3. Cross-Attention from memory to compact prompt tokens + null_kv (<= 8 KV tokens -> <0.5ms!)
        kv_list = [q_prompt_tok.float(), self.null_kv.expand(bsz, -1, -1)]
        if ex_mem is not None:
            kv_list.append(ex_mem.float().unsqueeze(0))
        if ex_clip_proj is not None:
            kv_list.append(ex_clip_proj.float().unsqueeze(0))
        kv = self.kv_norm(torch.cat(kv_list, dim=1))
        q = self.q_norm(mem_fp32)
        ca_out = self.out_norm(self.cross_attn(q, kv, kv, need_weights=False)[0])

        gate = 0.5 * torch.tanh(self.fuse_gate)
        fused = (mem_fp32 + sim_delta + gate * ca_out).to(orig_dtype)

        # 4. Centered spatial prompt prior for Stage-1 top-1,000 proposal selection
        if text_target_embed is not None and ex_clip is not None:
            # Blend centered CLIPSelf text sim + centered visual sim (CLIPSelf + stride-8 mem)
            prior = 0.5 * (t_sim_all - 0.22) * 4.0 + 0.5 * (0.5 * (v_sim_all - 0.55) + 0.5 * (m_sim_max - 0.55)) * 2.5
        elif text_target_embed is not None:
            prior = (t_sim_all - 0.22) * 4.0
        else:
            prior = (0.5 * (v_sim_all - 0.55) + 0.5 * (m_sim_max - 0.55)) * 2.5
        return fused, prior.clamp(-3.0, 3.0)


class _BackboneJoiner(nn.Sequential):
    """Matches OV-DQUO state_dict prefix `backbone.0.*`."""
    def __init__(self, body: EvaCLIPViTBackbone, pos_enc: PositionEmbeddingSineHW):
        super().__init__(body, pos_enc)


class OVDQUOCounting(nn.Module):
    ARCHITECTURE = 'OV-DQUO-Deformable-Counting-v2'

    def __init__(
        self,
        variant: str = 'vitl14',
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda',
        resolution: Optional[int] = None,
        num_queries: int = 1000,
        use_exemplars: bool = True,
        grad_checkpoint: bool = True,
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

        # 4. 6-layer iterative bbox_embed + class_embed
        _bbox = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        nn.init.constant_(_bbox.layers[-1].weight, 0.0)
        nn.init.constant_(_bbox.layers[-1].bias, 0.0)
        _cls = nn.Linear(self.hidden_dim, 1)
        nn.init.constant_(_cls.bias, -math.log((1.0 - 0.01) / 0.01))
        self.bbox_embed = nn.ModuleList([copy.deepcopy(_bbox) for _ in range(self.num_dec_layers)])
        self.class_embed = nn.ModuleList([copy.deepcopy(_cls) for _ in range(self.num_dec_layers)])
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # 5. v2 Prompt-Conditioned Memory Fusion + Guided Stage-1 Proposal Selection
        self.mem_fusion = PromptMemoryFusion(d_model=self.hidden_dim, nhead=int(cfg.get('nheads', 8)))
        self.enc_prompt_prior = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))

        self.sim_inject = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, self.hidden_dim),
        )
        nn.init.normal_(self.sim_inject[-1].weight, std=1e-3)
        nn.init.zeros_(self.sim_inject[-1].bias)

        # Frozen CLIPSelf text calibration
        self.clip_text_bias = nn.Parameter(torch.tensor(-0.22, dtype=torch.float32))
        self.clip_text_scale = nn.Parameter(torch.tensor(19.0, dtype=torch.float32))

        if self.use_exemplars:
            # Same-space metric projection on frozen CLIPSelf ROI features
            self.vis_proj = nn.Sequential(
                nn.Linear(self.text_dim, 256, bias=False),
                nn.LayerNorm(256),
            )
            # Class-agnostic Stride-8 Multi-Scale Deformable Decoder exemplar metric projection
            self.dec_vis_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                nn.LayerNorm(self.hidden_dim),
            )
            # Softmax blend over [CLIPSelf raw, CLIPSelf learned, Stride-8 Decoder memory]
            self.vis_mix = nn.Parameter(torch.tensor([0.5, 0.0, 0.3], dtype=torch.float32))
            self.clip_vis_bias = nn.Parameter(torch.tensor(-0.60, dtype=torch.float32))
            self.clip_vis_scale = nn.Parameter(torch.tensor(7.0, dtype=torch.float32))
            self.mm_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.mm_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Exact OV-DQUO ImageNet normalization
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
        cleaned = {}
        for k, v in sd.items():
            nk = k[7:] if k.startswith('module.') else k
            if nk.startswith('label_enc.') or nk == 'classifier' or 'freqs_cos_' in nk or 'freqs_sin_' in nk:
                continue
            if nk.endswith('.flag') or nk.startswith('transformer.decoder.class_embed.'):
                continue
            cleaned[nk] = v

        if 'class_embed.0.weight' in cleaned and 'class_embed.1.weight' not in cleaned:
            for i in range(1, self.num_dec_layers):
                cleaned[f'class_embed.{i}.weight'] = cleaned['class_embed.0.weight'].clone()
                cleaned[f'class_embed.{i}.bias'] = cleaned['class_embed.0.bias'].clone()

        for i in range(self.num_dec_layers):
            for suf in ('0.weight', '0.bias', '1.weight', '1.bias', '2.weight', '2.bias'):
                src_k = f'bbox_embed.{i}.layers.{suf}'
                dst_k = f'transformer.decoder.bbox_embed.{i}.layers.{suf}'
                if src_k in cleaned and dst_k not in cleaned:
                    cleaned[dst_k] = cleaned[src_k]

        own_sd = self.state_dict()
        if 'transformer.tgt_embed.weight' in cleaned and cleaned['transformer.tgt_embed.weight'].shape != own_sd['transformer.tgt_embed.weight'].shape:
            w = cleaned['transformer.tgt_embed.weight']
            n_src, _ = w.shape
            n_dst = own_sd['transformer.tgt_embed.weight'].shape[0]
            if n_dst <= n_src:
                cleaned['transformer.tgt_embed.weight'] = w[:n_dst].clone()
            else:
                reps = (n_dst + n_src - 1) // n_src
                cleaned['transformer.tgt_embed.weight'] = w.repeat(reps, 1)[:n_dst].clone()

        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        allowed_missing_prefixes = (
            'mem_fusion.', 'enc_prompt_prior', 'sim_inject.', 'clip_text_',
            'vis_proj.', 'dec_vis_proj.', 'vis_mix', 'clip_vis_', 'mm_alpha', 'mm_bias',
            '_img_mean', '_img_std',
        )
        bad_missing = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
        if bad_missing:
            raise RuntimeError(f'CRITICAL: Missing official OV-DQUO weights ({len(bad_missing)}): {bad_missing[:10]}')
        print(
            f'Loaded official OV-DQUO ({self.variant}) checkpoint: matched={len(own_sd) - len(missing)} '
            f'new_v2_heads={len(missing)} ignored_unexpected={len(unexpected)}',
            flush=True,
        )

    def _configure_freezing(self):
        self.backbone[0].visual.requires_grad_(False)
        self.backbone[0].interpolate1.requires_grad_(True)
        self.input_proj.requires_grad_(True)
        self.transformer.requires_grad_(True)
        self.bbox_embed.requires_grad_(True)
        self.class_embed.requires_grad_(True)
        self.mem_fusion.requires_grad_(True)
        self.enc_prompt_prior.requires_grad_(True)
        self.sim_inject.requires_grad_(True)
        self.clip_text_bias.requires_grad_(True)
        self.clip_text_scale.requires_grad_(True)
        if self.use_exemplars:
            self.vis_proj.requires_grad_(True)
            self.dec_vis_proj.requires_grad_(True)
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
        cal_names = {
            'enc_prompt_prior', 'mem_fusion.fuse_gate', 'clip_text_bias', 'clip_text_scale',
            'vis_mix', 'clip_vis_bias', 'clip_vis_scale', 'mm_alpha', 'mm_bias',
        }
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n in cal_names:
                cal_p.append(p)
            elif n.startswith('mem_fusion.') or n.startswith('sim_inject.') or n.startswith('vis_proj.') or n.startswith('dec_vis_proj.'):
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
            if hasattr(model, 'visual'):
                del model.visual
            model.requires_grad_(False).eval().to(device)
            self._open_clip_model = model
            self._open_clip_tokenizer = open_clip.get_tokenizer(arch)

    @torch.no_grad()
    def encode_text_queries(self, prompts: list[str], device) -> Tensor:
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
    def encode_exemplars(self, pil_image, exemplar_boxes: Tensor) -> Optional[dict[str, Tensor]]:
        """Extract BOTH frozen CLIPSelf `(E, text_dim)` AND stride-8/16 encoder `(E, 256)` exemplar features for 2x2 tiling."""
        if not self.use_exemplars or exemplar_boxes is None or len(exemplar_boxes) == 0:
            return None
        dev = self._img_mean.device
        pv, mask = self.preprocess_image(pil_image, device=dev)
        feats, masks, dense_map, dense_mask = self.backbone[0](pv, mask)
        srcs = [proj(f) for proj, f in zip(self.input_proj, feats)]
        pos_embeds = [self.backbone[1](m, dtype=s.dtype) for m, s in zip(masks, srcs)]
        enc = self.transformer.encode_memory(srcs, masks, pos_embeds, grad_checkpoint=False)
        with torch.autocast('cuda', enabled=False):
            ex_clip = self._sample_exemplar_roi(dense_map, dense_mask, exemplar_boxes)
            ex_mem = self.mem_fusion.extract_memory_exemplars(
                enc['memory'].float(), enc['spatial_shapes'], enc['valid_ratios'], exemplar_boxes,
            )
        return dict(ex_clip=ex_clip, ex_mem=ex_mem)

    def _score_stage(
        self,
        boxes_cxcywh: Tensor,
        obj_logit: Tensor,
        query_hidden: Tensor,
        dense_map: Tensor,
        dense_mask: Tensor,
        text_embeds: Optional[Tensor],
        target_idx: int,
        ex_clip: Optional[Tensor],
        ex_mem: Optional[Tensor],
    ) -> dict:
        """Pure-FP32 Log-AND semantic + multi-scale stride-8 decoder exemplar scoring for one stage."""
        roi_feats = self._sample_dense_roi(dense_map, dense_mask, boxes_cxcywh.detach())

        text_sem = None
        text_logits = None
        if text_embeds is not None:
            text_cos = torch.matmul(roi_feats, text_embeds.T)
            t_scale = F.elu(self.clip_text_scale) + 1.0
            text_sem = (text_cos + self.clip_text_bias) * t_scale
            text_logits = log_and_logit(obj_logit[:, None], text_sem)

        vis_sem = None
        vis_logits = None
        if ex_clip is not None:
            sim_clip_raw = torch.matmul(roi_feats, ex_clip.T).max(dim=-1).values
            q_lrn = F.normalize(self.vis_proj(roi_feats), dim=-1)
            e_lrn = F.normalize(self.vis_proj(ex_clip), dim=-1)
            sim_clip_lrn = torch.matmul(q_lrn, e_lrn.T).max(dim=-1).values

            if ex_mem is not None:
                q_dec = F.normalize(self.dec_vis_proj(query_hidden.float()), dim=-1)
                e_dec = F.normalize(self.dec_vis_proj(ex_mem.float()), dim=-1)
                sim_dec = torch.matmul(q_dec, e_dec.T).max(dim=-1).values
            else:
                sim_dec = sim_clip_raw

            w_v = F.softmax(self.vis_mix, dim=0)
            vis_sim = w_v[0] * sim_clip_raw + w_v[1] * sim_clip_lrn + w_v[2] * sim_dec
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
        exemplar_feats: Optional[dict[str, Tensor]] = None,
        use_text: bool = True,
    ) -> dict:
        """Full 7-stage forward with Prompt-Conditioned Memory Fusion & Guided Stage-1 Proposal Selection."""
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

        # 1. 6-Layer Multi-Scale Deformable Encoder -> 3-scale `memory`
        enc = self.transformer.encode_memory(srcs, masks, pos_embeds, grad_checkpoint=self.grad_checkpoint)

        # 2. Extract frozen CLIPSelf text + multi-scale stride-8/16 exemplars & fuse into `memory` (FP32)
        with torch.autocast('cuda', enabled=False):
            text_embeds = self.encode_text_queries(prompts, device=dev) if use_text else None
            ex_clip, ex_mem = None, None
            if has_ex:
                if isinstance(exemplar_feats, dict):
                    ex_clip = exemplar_feats['ex_clip'].to(device=dev, dtype=torch.float32)
                    ex_mem = exemplar_feats['ex_mem'].to(device=dev, dtype=torch.float32)
                elif exemplar_feats is not None:
                    ex_clip = exemplar_feats.to(device=dev, dtype=torch.float32)
                    ex_mem = None
                else:
                    ex_clip = self._sample_exemplar_roi(dense_map, dense_mask, exemplar_boxes)
                    ex_mem = self.mem_fusion.extract_memory_exemplars(
                        enc['memory'].float(), enc['spatial_shapes'], enc['valid_ratios'], exemplar_boxes,
                    )

            if text_embeds is not None and ex_clip is not None:
                prompt_emb = F.normalize(text_embeds[target_idx] + ex_clip.mean(dim=0), dim=-1)
            elif text_embeds is not None:
                prompt_emb = text_embeds[target_idx]
            else:
                prompt_emb = F.normalize(ex_clip.mean(dim=0), dim=-1)

            q_prompt = self.transformer.text_proj(prompt_emb)[None, None, :]  # (1, 1, C)
            ex_clip_proj = self.transformer.text_proj(ex_clip) if ex_clip is not None else None  # (E, C)

            fused_memory, prompt_prior_spatial = self.mem_fusion(
                enc['memory'],
                enc['spatial_shapes'],
                enc['valid_ratios'],
                dense_map,
                text_target_embed=text_embeds[target_idx] if text_embeds is not None else None,
                q_prompt_tok=q_prompt,
                ex_clip=ex_clip,
                ex_clip_proj=ex_clip_proj,
                ex_mem=ex_mem,
            )
            enc_prompt_bias = self.enc_prompt_prior * prompt_prior_spatial

        # 3. Prompt & Exemplar-Guided Stage-1 Top-1,000 Proposal Selection
        prop = self.transformer.propose_from_memory(
            fused_memory, enc['mask_flatten'], enc['spatial_shapes'], enc_prompt_bias=enc_prompt_bias,
        )
        refpoint_undetach = prop['refpoint_undetach']
        refpoint_enc_sig = refpoint_undetach.sigmoid()
        enc_obj_logit = prop['enc_class_topk'][0, :, 0]
        enc_hidden_topk = prop['tgt_undetach'][0]  # (Q, C)

        with torch.autocast('cuda', enabled=False):
            roi_enc = self._sample_dense_roi(dense_map, dense_mask, refpoint_enc_sig[0].detach())
            t_sim = torch.matmul(roi_enc, text_embeds[target_idx]) if text_embeds is not None else roi_enc.new_zeros(roi_enc.shape[0])
            if ex_clip is not None:
                v_all = torch.matmul(roi_enc, ex_clip.T)
                v_max = v_all.max(dim=-1).values
                v_mean = v_all.mean(dim=-1)
                v_flag = roi_enc.new_ones(roi_enc.shape[0])
            else:
                v_max = roi_enc.new_zeros(roi_enc.shape[0])
                v_mean = roi_enc.new_zeros(roi_enc.shape[0])
                v_flag = roi_enc.new_zeros(roi_enc.shape[0])
            sim_hint = torch.stack([t_sim, v_max, v_mean, v_flag], dim=-1).unsqueeze(0)
            sim_delta = self.sim_inject(sim_hint)

        q_actual = refpoint_undetach.shape[1]
        tgt_init = self.transformer.tgt_embed.weight[:q_actual].unsqueeze(0) + q_prompt.to(srcs[0].dtype) + sim_delta.to(srcs[0].dtype)

        # 4. 6-Layer Multi-Scale Deformable Decoder attending to Prompt-Fused Multi-Scale Memory
        hs, references = self.transformer.decoder(
            tgt=tgt_init.transpose(0, 1),
            memory=fused_memory.transpose(0, 1),
            memory_key_padding_mask=enc['mask_flatten'],
            refpoints_unsigmoid=refpoint_undetach.detach().transpose(0, 1),
            spatial_shapes=enc['spatial_shapes'],
            valid_ratios=enc['valid_ratios'],
        )

        # 5. Score all 6 decoder layers + Stage-1 encoder proposals in pure FP32
        with torch.autocast('cuda', enabled=False):
            dec_stages = []
            for lvl in range(self.num_dec_layers):
                h_l = hs[lvl].float()
                ref_l = references[lvl].float()
                delta_l = self.bbox_embed[lvl](h_l)
                boxes_l = (delta_l + inverse_sigmoid(ref_l)).sigmoid()[0]
                obj_l = self.class_embed[lvl](h_l)[0, :, 0]
                dec_stages.append(
                    self._score_stage(boxes_l, obj_l, h_l[0], dense_map, dense_mask, text_embeds, target_idx, ex_clip, ex_mem)
                )
            enc_stage = self._score_stage(
                refpoint_enc_sig[0].float(), enc_obj_logit.float(), enc_hidden_topk.float(),
                dense_map, dense_mask, text_embeds, target_idx, ex_clip, ex_mem,
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
        exemplar_feats: Optional[dict[str, Tensor]] = None,
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
