"""Unified strict-key inference wrapper for OV-DQUO v1, v2, and v3.1 checkpoints in the SAM2 VideoCount pipeline.

Supports exact bit-for-bit loading of:
- `OV-DQUO-Deformable-Counting-v1` (commit `592a8d1`, e.g. ViT-B/16 Full-Val 13.32)
- `OV-DQUO-Deformable-Counting-v2` (commit `53da63e`, e.g. ViT-L/14 Full-Val 10.85 / F1 0.830)
- `OV-DQUO-Deformable-Counting-v3.1` (current `OVDQUOCounting`, e.g. ViT-L/14 Full-Val 11.08 / absent@0.26 3.50)

All three expose identical `detect_with_features(pil_image, prompt, threshold, tile_trigger)` and
`extract_box_features(dense_map, dense_mask, boxes_xyxy_px, image_size)` methods.
"""
from collections import OrderedDict
import copy
from typing import Optional
import numpy as np
from PIL import Image
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from models.yoloworld_counting import xywh_to_xyxy
from models.ov_dquo_backbone import (
    EVAVisionTransformer,
    EvaCLIPViTBackbone,
    MLP,
    OVDeformableTransformer,
    PositionEmbeddingSineHW,
    inverse_sigmoid,
)
from models.ov_dquo_counting import (
    CLIP_TEMPLATES,
    OV_DQUO_VARIANTS,
    OVDQUOCounting,
    _BackboneJoiner,
    log_and_logit,
)
from models.ov_dquo_v1_infer import OVDQUOV1Detector


def _expand_bbox_aliases(state: dict) -> dict:
    """Mirror aliased `bbox_embed.*` <-> `transformer.decoder.bbox_embed.*` keys from `export_trainable_state`."""
    out = dict(state)
    for k in list(out.keys()):
        if k.startswith('transformer.decoder.bbox_embed.'):
            out.setdefault(k[len('transformer.decoder.'):], out[k])
        elif k.startswith('bbox_embed.'):
            out.setdefault('transformer.decoder.' + k, out[k])
    return out


def _postprocess_detections(boxes_cxcywh: Tensor, scores: Tensor, pil_image: Image.Image,
                            threshold: float, dense_map: Tensor, dense_mask: Tensor, tiled: bool) -> dict:
    w, h = pil_image.size
    keep = scores > float(threshold)
    if not keep.any():
        return dict(
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            dense_map=dense_map,
            dense_mask=dense_mask,
            tiled=tiled,
        )
    b_kept = boxes_cxcywh[keep]
    s_kept = scores[keep]
    order = torch.argsort(s_kept, descending=True)
    b_kept = b_kept[order]
    s_kept = s_kept[order]
    xyxy = xywh_to_xyxy(b_kept).clamp(0.0, 1.0)
    xyxy_px = (xyxy * xyxy.new_tensor([float(w), float(h), float(w), float(h)])).numpy().astype(np.float32)
    valid_sz = (xyxy_px[:, 2] - xyxy_px[:, 0] >= 1.0) & (xyxy_px[:, 3] - xyxy_px[:, 1] >= 1.0)
    return dict(
        boxes_xyxy=xyxy_px[valid_sz],
        scores=s_kept.numpy().astype(np.float32)[valid_sz],
        dense_map=dense_map,
        dense_mask=dense_mask,
        tiled=tiled,
    )


class _PromptMemoryFusionV2(nn.Module):
    """Exact state-dict match for `PromptMemoryFusion` in OV-DQUO v2 (commit `53da63e`)."""
    def __init__(self, d_model: int = 256, nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        self.sim_proj = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
        )
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)
        self.null_kv = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fuse_gate = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward_text_only(
        self,
        memory: Tensor,
        spatial_shapes: Tensor,
        dense_map: Tensor,
        text_target_embed: Tensor,
        q_prompt_tok: Tensor,
    ) -> tuple[Tensor, Tensor]:
        orig_dtype = memory.dtype
        mem_fp32 = memory.float()
        bsz, seq_len, _ = mem_fp32.shape
        shapes = spatial_shapes.tolist()

        _, _, hd, wd = dense_map.shape
        dense_flat = dense_map.float().flatten(2).transpose(1, 2)
        t_map_2d = torch.matmul(dense_flat, text_target_embed).reshape(1, 1, hd, wd)
        t_up_list = [
            F.interpolate(t_map_2d, size=(h, w), mode='bilinear', align_corners=False).flatten(1)
            for h, w in shapes
        ]
        t_sim_all = torch.cat(t_up_list, dim=1)
        zeros = mem_fp32.new_zeros((bsz, seq_len))
        ones = mem_fp32.new_ones((bsz, seq_len))
        sim_feats = torch.stack([t_sim_all, zeros, zeros, zeros, ones, zeros], dim=-1)
        sim_delta = self.sim_proj(sim_feats)

        kv = self.kv_norm(torch.cat([q_prompt_tok.float(), self.null_kv.expand(bsz, -1, -1)], dim=1))
        q = self.q_norm(mem_fp32)
        ca_out = self.out_norm(self.cross_attn(q, kv, kv, need_weights=False)[0])

        gate = 0.5 * torch.tanh(self.fuse_gate)
        fused = (mem_fp32 + sim_delta + gate * ca_out).to(orig_dtype)
        prior = ((t_sim_all - 0.22) * 4.0).clamp(-3.0, 3.0)
        return fused, prior


class OVDQUOV2Detector(nn.Module):
    """Exact state-dict match for `OV-DQUO-Deformable-Counting-v2` (commit `53da63e`)."""
    ARCHITECTURE = 'OV-DQUO-Deformable-Counting-v2'
    extract_box_features = staticmethod(OVDQUOV1Detector.extract_box_features)

    def __init__(
        self,
        variant: str = 'vitl14',
        base_checkpoint_path: Optional[str] = None,
        tuned_checkpoint_path: Optional[str] = None,
        device: str = 'cuda',
        resolution: Optional[int] = 672,
        num_queries: int = 1000,
        use_exemplars: bool = True,
        text_encoder_fn=None,
        _tiny_cfg: Optional[dict] = None,
    ):
        super().__init__()
        cfg = dict(_tiny_cfg) if _tiny_cfg is not None else dict(OV_DQUO_VARIANTS[variant])
        self.variant = variant
        self.cfg = cfg
        self.resolution = int(resolution if resolution is not None else cfg['default_resolution'])
        self.patch_size = int(cfg['patch_size'])
        self.num_queries = int(num_queries)
        self.use_exemplars = bool(use_exemplars)
        self.hidden_dim = int(cfg.get('hidden_dim', 256))
        self.text_dim = int(cfg['text_dim'])
        self.num_dec_layers = int(cfg.get('dec_layers', 6))
        self.num_enc_layers = int(cfg.get('enc_layers', 6))
        nheads = int(cfg.get('nheads', 8))

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

        in_ch = int(cfg['embed_dim'])
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, self.hidden_dim, kernel_size=1),
                nn.GroupNorm(min(32, self.hidden_dim), self.hidden_dim),
            )
            for _ in range(3)
        ])

        self.transformer = OVDeformableTransformer(
            d_model=self.hidden_dim,
            nhead=nheads,
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

        _bbox = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        _cls = nn.Linear(self.hidden_dim, 1)
        self.bbox_embed = nn.ModuleList([copy.deepcopy(_bbox) for _ in range(self.num_dec_layers)])
        self.class_embed = nn.ModuleList([copy.deepcopy(_cls) for _ in range(self.num_dec_layers)])
        self.transformer.decoder.bbox_embed = self.bbox_embed

        self.mem_fusion = _PromptMemoryFusionV2(d_model=self.hidden_dim, nhead=nheads)
        self.enc_prompt_prior = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))
        self.sim_inject = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, self.hidden_dim),
        )
        self.clip_text_bias = nn.Parameter(torch.tensor(-0.22, dtype=torch.float32))
        self.clip_text_scale = nn.Parameter(torch.tensor(19.0, dtype=torch.float32))

        if self.use_exemplars:
            self.vis_proj = nn.Sequential(nn.Linear(self.text_dim, 256, bias=False), nn.LayerNorm(256))
            self.dec_vis_proj = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False), nn.LayerNorm(self.hidden_dim))
            self.vis_mix = nn.Parameter(torch.tensor([0.5, 0.0, 0.3], dtype=torch.float32))
            self.clip_vis_bias = nn.Parameter(torch.tensor(-0.60, dtype=torch.float32))
            self.clip_vis_scale = nn.Parameter(torch.tensor(7.0, dtype=torch.float32))
            self.mm_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.mm_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.register_buffer('_img_mean', torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('_img_std', torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        self._text_encoder_fn = text_encoder_fn
        self._open_clip_model = None
        self._open_clip_tokenizer = None
        self._text_cache: OrderedDict[str, Tensor] = OrderedDict()

        if base_checkpoint_path is not None:
            self.load_official_base(base_checkpoint_path)
        if tuned_checkpoint_path is not None:
            self.load_tuned_v2(tuned_checkpoint_path)

        self.to(device)
        self.requires_grad_(False)
        self.eval()

    def load_official_base(self, ckpt_path: str):
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
            cleaned['transformer.tgt_embed.weight'] = (
                w[:n_dst].clone() if n_dst <= n_src else w.repeat((n_dst + n_src - 1) // n_src, 1)[:n_dst].clone()
            )
        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        allowed = (
            'mem_fusion.', 'enc_prompt_prior', 'sim_inject.', 'clip_text_',
            'vis_proj.', 'dec_vis_proj.', 'vis_mix', 'clip_vis_', 'mm_alpha', 'mm_bias',
            '_img_mean', '_img_std',
        )
        bad_missing = [k for k in missing if not k.startswith(allowed)]
        if bad_missing:
            raise RuntimeError(f'Missing official OV-DQUO v2 base weights ({len(bad_missing)}): {bad_missing[:10]}')
        print(f'[OVDQUOV2Detector] Loaded base checkpoint: matched={len(own_sd) - len(missing)} ignored={len(unexpected)}', flush=True)

    def load_tuned_v2(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        arch = ckpt.get('architecture')
        if arch is not None and arch != self.ARCHITECTURE:
            raise ValueError(f'Expected {self.ARCHITECTURE}, got {arch}')
        state = _expand_bbox_aliases(ckpt['trainable'] if 'trainable' in ckpt else ckpt)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f'Unexpected keys loading OV-DQUO v2 tuned checkpoint: {unexpected[:10]}')
        non_visual_missing = [k for k in missing if not k.startswith('backbone.0.visual.') and k not in ('_img_mean', '_img_std')]
        if non_visual_missing:
            raise RuntimeError(f'Missing trainable OV-DQUO v2 keys ({len(non_visual_missing)}): {non_visual_missing[:10]}')
        print(
            f'[OVDQUOV2Detector] Loaded tuned v2 checkpoint: keys={len(state)} '
            f'epoch={ckpt.get("epoch")} dev_mae={ckpt.get("dev_mae")}',
            flush=True,
        )

    encode_text = OVDQUOV1Detector.encode_text
    _ensure_text_encoder = OVDQUOV1Detector._ensure_text_encoder
    preprocess_image = OVDQUOV1Detector.preprocess_image

    @torch.no_grad()
    def _forward_single_text(self, pil_image: Image.Image, prompt: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dev = self._img_mean.device
        use_amp = (dev.type == 'cuda')
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            pv, mask = self.preprocess_image(pil_image, device=dev)
            feats, masks, dense_map, dense_mask = self.backbone[0](pv, mask)
            srcs = [proj(f) for proj, f in zip(self.input_proj, feats)]
            pos_embeds = [self.backbone[1](m, dtype=s.dtype) for m, s in zip(masks, srcs)]
            enc = self.transformer.encode_memory(srcs, masks, pos_embeds, grad_checkpoint=False)

            with torch.autocast('cuda', enabled=False):
                text_emb = self.encode_text(prompt, device=dev)
                q_prompt = self.transformer.text_proj(text_emb)[None, None, :]
                fused_memory, prompt_prior_spatial = self.mem_fusion.forward_text_only(
                    enc['memory'], enc['spatial_shapes'], dense_map, text_emb, q_prompt,
                )
                enc_prompt_bias = self.enc_prompt_prior * prompt_prior_spatial

            prop = self.transformer.propose_from_memory(
                fused_memory, enc['mask_flatten'], enc['spatial_shapes'], enc_prompt_bias=enc_prompt_bias,
            )
            refpoint_undetach = prop['refpoint_undetach']
            refpoint_enc_sig = refpoint_undetach.sigmoid()

            with torch.autocast('cuda', enabled=False):
                roi_enc = OVDQUOV1Detector._sample_dense_roi(dense_map, dense_mask, refpoint_enc_sig[0])
                t_sim = torch.matmul(roi_enc, text_emb)
                zeros = roi_enc.new_zeros(roi_enc.shape[0])
                sim_hint = torch.stack([t_sim, zeros, zeros, zeros], dim=-1).unsqueeze(0)
                sim_delta = self.sim_inject(sim_hint)

            q_actual = refpoint_undetach.shape[1]
            tgt_init = self.transformer.tgt_embed.weight[:q_actual].unsqueeze(0) + q_prompt.to(srcs[0].dtype) + sim_delta.to(srcs[0].dtype)
            hs, references = self.transformer.decoder(
                tgt=tgt_init.transpose(0, 1),
                memory=fused_memory.transpose(0, 1),
                memory_key_padding_mask=enc['mask_flatten'],
                refpoints_unsigmoid=refpoint_undetach.detach().transpose(0, 1),
                spatial_shapes=enc['spatial_shapes'],
                valid_ratios=enc['valid_ratios'],
            )
            with torch.autocast('cuda', enabled=False):
                lvl = self.num_dec_layers - 1
                h_l = hs[lvl].float()
                ref_l = references[lvl].float()
                delta_l = self.bbox_embed[lvl](h_l)
                boxes_cxcywh = (delta_l + inverse_sigmoid(ref_l)).sigmoid()[0].clamp(0.0, 1.0)
                obj_l = self.class_embed[lvl](h_l)[0, :, 0]
                roi_feats = OVDQUOV1Detector._sample_dense_roi(dense_map, dense_mask, boxes_cxcywh)
                text_cos = torch.matmul(roi_feats, text_emb)
                t_scale = F.elu(self.clip_text_scale) + 1.0
                text_sem = (text_cos + self.clip_text_bias) * t_scale
                scores = log_and_logit(obj_l, text_sem).sigmoid()
        return boxes_cxcywh.float().cpu(), scores.float().cpu(), dense_map.half().cpu(), dense_mask.cpu()

    @torch.no_grad()
    def detect_with_features(self, pil_image: Image.Image, prompt: str, threshold: float = 0.26, tile_trigger: int = 400) -> dict:
        boxes_cxcywh, scores, dense_map, dense_mask = self._forward_single_text(pil_image, prompt)
        w, h = pil_image.size
        tiled = False
        if tile_trigger > 0 and int((scores > threshold).sum().item()) >= tile_trigger:
            tiled = True
            mx, my = w // 2, h // 2
            quads = [
                (0, 0, mx, my, 0.0, 0.0, mx / w, my / h),
                (mx, 0, w, my, mx / w, 0.0, (w - mx) / w, my / h),
                (0, my, mx, h, 0.0, my / h, mx / w, (h - my) / h),
                (mx, my, w, h, mx / w, my / h, (w - mx) / w, (h - my) / h),
            ]
            all_b, all_s = [], []
            for x0, y0, x1, y1, ox, oy, sw, sh in quads:
                qb, qs, _, _ = self._forward_single_text(pil_image.crop((x0, y0, x1, y1)), prompt)
                qb[:, 0] = ox + qb[:, 0] * sw
                qb[:, 1] = oy + qb[:, 1] * sh
                qb[:, 2] = qb[:, 2] * sw
                qb[:, 3] = qb[:, 3] * sh
                all_b.append(qb)
                all_s.append(qs)
            boxes_cxcywh = torch.cat(all_b, dim=0)
            scores = torch.cat(all_s, dim=0)
        return _postprocess_detections(boxes_cxcywh, scores, pil_image, threshold, dense_map, dense_mask, tiled)


class OVDQUOV31Detector(nn.Module):
    """Strict-key inference wrapper around current `OVDQUOCounting` (`OV-DQUO-Deformable-Counting-v3.1`)."""
    ARCHITECTURE = OVDQUOCounting.ARCHITECTURE
    extract_box_features = staticmethod(OVDQUOV1Detector.extract_box_features)

    def __init__(
        self,
        variant: str = 'vitl14',
        base_checkpoint_path: Optional[str] = None,
        tuned_checkpoint_path: Optional[str] = None,
        device: str = 'cuda',
        resolution: Optional[int] = 672,
        num_queries: int = 1000,
        use_exemplars: bool = True,
        text_encoder_fn=None,
        _tiny_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.model = OVDQUOCounting(
            variant=variant,
            checkpoint_path=base_checkpoint_path,
            device=device,
            resolution=resolution,
            num_queries=num_queries,
            use_exemplars=use_exemplars,
            grad_checkpoint=False,
            text_encoder_fn=text_encoder_fn,
            _tiny_cfg=_tiny_cfg,
        )
        if tuned_checkpoint_path is not None:
            self.load_tuned_v31(tuned_checkpoint_path)
        self.model.to(device)
        self.model.requires_grad_(False)
        self.model.set_training(False)

    def load_tuned_v31(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        arch = ckpt.get('architecture')
        if arch is not None and arch != self.ARCHITECTURE:
            raise ValueError(f'Expected {self.ARCHITECTURE}, got {arch}')
        state = _expand_bbox_aliases(ckpt['trainable'] if 'trainable' in ckpt else ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f'Unexpected keys loading OV-DQUO v3.1 tuned checkpoint: {unexpected[:10]}')
        non_visual_missing = [k for k in missing if not k.startswith('backbone.0.visual.') and k not in ('_img_mean', '_img_std')]
        if non_visual_missing:
            raise RuntimeError(f'Missing trainable OV-DQUO v3.1 keys ({len(non_visual_missing)}): {non_visual_missing[:10]}')
        print(
            f'[OVDQUOV31Detector] Loaded tuned v3.1 checkpoint: keys={len(state)} '
            f'epoch={ckpt.get("epoch")} dev_mae={ckpt.get("dev_mae")}',
            flush=True,
        )

    @torch.no_grad()
    def _forward_single_text(self, pil_image: Image.Image, prompt: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        m = self.model
        dev = m._img_mean.device
        use_amp = (dev.type == 'cuda')
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            pv, mask = m.preprocess_image(pil_image, device=dev)
            feats, masks, dense_map, dense_mask = m.backbone[0](pv, mask)
            srcs = [proj(f) for proj, f in zip(m.input_proj, feats)]
            pos_embeds = [m.backbone[1](msk, dtype=s.dtype) for msk, s in zip(masks, srcs)]
            enc = m.transformer.encode_memory(srcs, masks, pos_embeds, grad_checkpoint=False)

            with torch.autocast('cuda', enabled=False):
                text_embeds = m.encode_text_queries([prompt], device=dev)
                text_target_embed = text_embeds[0]
                text_tpl_embs = m.encode_text_templates(prompt, device=dev)
                q_prompt, q_text_kv, _ = m._build_prompt_tokens(text_target_embed, text_tpl_embs, None)
                fused_memory, prompt_prior_spatial = m.mem_fusion(
                    enc['memory'], enc['spatial_shapes'], enc['valid_ratios'], dense_map,
                    text_target_embed=text_target_embed, q_prompt_tok=q_prompt, q_text_kv=q_text_kv,
                )
                prior_logits = m.enc_prompt_prior * prompt_prior_spatial + m.enc_prior_bias

            prop = m.transformer.propose_from_memory(
                fused_memory, enc['mask_flatten'], enc['spatial_shapes'], enc_prompt_bias=prior_logits.detach(),
            )
            refpoint_undetach = prop['refpoint_undetach']
            refpoint_enc_sig = refpoint_undetach.sigmoid()
            enc_hidden_topk = prop['tgt_undetach'][0]

            with torch.autocast('cuda', enabled=False):
                roi_enc = m._sample_dense_roi(dense_map, dense_mask, refpoint_enc_sig[0])
                t_sim = torch.matmul(roi_enc, text_target_embed)
                zeros = roi_enc.new_zeros(roi_enc.shape[0])
                sim_hint = torch.stack([t_sim, zeros, zeros, zeros], dim=-1).unsqueeze(0)
                sim_delta = m.sim_inject(sim_hint)
                enc_tgt_delta = (0.5 * torch.tanh(m.enc_tgt_gate)) * m.enc_tgt_proj(enc_hidden_topk.float().unsqueeze(0))

            q_actual = refpoint_undetach.shape[1]
            tgt_init = (
                m.transformer.tgt_embed.weight[:q_actual].unsqueeze(0)
                + q_prompt.to(srcs[0].dtype)
                + sim_delta.to(srcs[0].dtype)
                + enc_tgt_delta.to(srcs[0].dtype)
            )
            hs, references = m.transformer.decoder(
                tgt=tgt_init.transpose(0, 1),
                memory=fused_memory.transpose(0, 1),
                memory_key_padding_mask=enc['mask_flatten'],
                refpoints_unsigmoid=refpoint_undetach.detach().transpose(0, 1),
                spatial_shapes=enc['spatial_shapes'],
                valid_ratios=enc['valid_ratios'],
            )
            with torch.autocast('cuda', enabled=False):
                lvl = m.num_dec_layers - 1
                h_l = hs[lvl].float()
                ref_l = references[lvl].float()
                delta_l = m.bbox_embed[lvl](h_l)
                boxes_cxcywh = (delta_l + inverse_sigmoid(ref_l)).sigmoid()[0].clamp(0.0, 1.0)
                obj_l = m.class_embed[lvl](h_l)[0, :, 0]
                st = m._score_stage(
                    boxes_cxcywh, obj_l, h_l[0], dense_map, dense_mask,
                    text_embeds, target_idx=0, ex_clip=None, ex_mem=None, is_enc=False,
                )
                scores = st['text_logits'][:, 0].sigmoid()
        return boxes_cxcywh.float().cpu(), scores.float().cpu(), dense_map.half().cpu(), dense_mask.cpu()

    @torch.no_grad()
    def detect_with_features(self, pil_image: Image.Image, prompt: str, threshold: float = 0.26, tile_trigger: int = 400) -> dict:
        boxes_cxcywh, scores, dense_map, dense_mask = self._forward_single_text(pil_image, prompt)
        w, h = pil_image.size
        tiled = False
        if tile_trigger > 0 and int((scores > threshold).sum().item()) >= tile_trigger:
            tiled = True
            mx, my = w // 2, h // 2
            quads = [
                (0, 0, mx, my, 0.0, 0.0, mx / w, my / h),
                (mx, 0, w, my, mx / w, 0.0, (w - mx) / w, my / h),
                (0, my, mx, h, 0.0, my / h, mx / w, (h - my) / h),
                (mx, my, w, h, mx / w, my / h, (w - mx) / w, (h - my) / h),
            ]
            all_b, all_s = [], []
            for x0, y0, x1, y1, ox, oy, sw, sh in quads:
                qb, qs, _, _ = self._forward_single_text(pil_image.crop((x0, y0, x1, y1)), prompt)
                qb[:, 0] = ox + qb[:, 0] * sw
                qb[:, 1] = oy + qb[:, 1] * sh
                qb[:, 2] = qb[:, 2] * sw
                qb[:, 3] = qb[:, 3] * sh
                all_b.append(qb)
                all_s.append(qs)
            boxes_cxcywh = torch.cat(all_b, dim=0)
            scores = torch.cat(all_s, dim=0)
        return _postprocess_detections(boxes_cxcywh, scores, pil_image, threshold, dense_map, dense_mask, tiled)


def load_ov_dquo_video_detector(
    base_checkpoint_path: str,
    tuned_checkpoint_path: str,
    device: str = 'cuda',
    resolution: Optional[int] = None,
    text_encoder_fn=None,
) -> nn.Module:
    """Auto-detect v1 / v2 / v3.1 from tuned checkpoint metadata and load with strict key verification."""
    meta = torch.load(tuned_checkpoint_path, map_location='cpu', weights_only=False)
    arch = meta.get('architecture', OVDQUOV1Detector.ARCHITECTURE)
    variant = meta.get('variant', 'vitb16')
    res = int(resolution if resolution is not None else meta.get('resolution', OV_DQUO_VARIANTS[variant]['default_resolution']))
    use_ex = bool(meta.get('use_exemplars', True))
    if arch == OVDQUOV1Detector.ARCHITECTURE:
        return OVDQUOV1Detector(
            variant=variant, base_checkpoint_path=base_checkpoint_path,
            tuned_checkpoint_path=tuned_checkpoint_path, device=device,
            resolution=res, use_exemplars=use_ex, text_encoder_fn=text_encoder_fn,
        )
    if arch == OVDQUOV2Detector.ARCHITECTURE:
        return OVDQUOV2Detector(
            variant=variant, base_checkpoint_path=base_checkpoint_path,
            tuned_checkpoint_path=tuned_checkpoint_path, device=device,
            resolution=res, use_exemplars=use_ex, text_encoder_fn=text_encoder_fn,
        )
    if arch == OVDQUOV31Detector.ARCHITECTURE:
        return OVDQUOV31Detector(
            variant=variant, base_checkpoint_path=base_checkpoint_path,
            tuned_checkpoint_path=tuned_checkpoint_path, device=device,
            resolution=res, use_exemplars=use_ex, text_encoder_fn=text_encoder_fn,
        )
    raise ValueError(f'Unsupported OV-DQUO checkpoint architecture: {arch}')
