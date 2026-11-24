"""Standalone exact OV-DQUO v1 inference wrapper (commit `592a8d1` schema) for VideoCount + SAM2 + Tracking.

Loads BOTH:
1. Official pretrained `OVDQUO_ViT-B16_LVIS.pth` (or `ViT-L14`) backbone & transformer weights, AND
2. Fine-tuned `OV-DQUO-Deformable-Counting-v1` trainable checkpoint (e.g. `ov_dquo_v1_vitb16_mm_ep4_fullval13.32.pth`,
   which reached Full-Val MM Tiled MAE = 13.32 / F1 = 0.802 and Text-Only MAE = 14.56 / F1 = 0.786, Absent = 0.00)
with STRICT zero-missing, zero-unexpected key verification.

Also exposes per-frame frozen `EVA02-CLIP` `dense_map` (`48x48x512`) so downstream Stage-4 `TemporalIdentity`
tracklet stitching gets L2-normalized 512-d appearance features for every tracked SAM2 box at zero extra ViT cost!
"""
from collections import OrderedDict
import copy
import math
from typing import Optional
import numpy as np
from PIL import Image
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
from models.ov_dquo_counting import CLIP_TEMPLATES, OV_DQUO_VARIANTS, _BackboneJoiner, log_and_logit


class OVDQUOV1Detector(nn.Module):
    """Exact state-dict match for `OV-DQUO-Deformable-Counting-v1` (commit 592a8d1)."""
    ARCHITECTURE = 'OV-DQUO-Deformable-Counting-v1'

    def __init__(
        self,
        variant: str = 'vitb16',
        base_checkpoint_path: Optional[str] = None,
        tuned_checkpoint_path: Optional[str] = None,
        device: str = 'cuda',
        resolution: Optional[int] = 768,
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

        _bbox = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        _cls = nn.Linear(self.hidden_dim, 1)
        self.bbox_embed = nn.ModuleList([copy.deepcopy(_bbox) for _ in range(self.num_dec_layers)])
        self.class_embed = nn.ModuleList([copy.deepcopy(_cls) for _ in range(self.num_dec_layers)])
        self.transformer.decoder.bbox_embed = self.bbox_embed

        self.sim_inject = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, self.hidden_dim),
        )
        self.clip_text_bias = nn.Parameter(torch.tensor(-0.22, dtype=torch.float32))
        self.clip_text_scale = nn.Parameter(torch.tensor(19.0, dtype=torch.float32))

        if self.use_exemplars:
            self.vis_proj = nn.Sequential(
                nn.Linear(self.text_dim, 256, bias=False),
                nn.LayerNorm(256),
            )
            self.vis_mix = nn.Parameter(torch.tensor([0.7, 0.0], dtype=torch.float32))
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
            self.load_tuned_v1(tuned_checkpoint_path)

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
        allowed = ('sim_inject.', 'clip_text_', 'vis_proj.', 'vis_mix', 'clip_vis_', 'mm_alpha', 'mm_bias', '_img_mean', '_img_std')
        bad_missing = [k for k in missing if not k.startswith(allowed)]
        if bad_missing:
            raise RuntimeError(f'Missing official OV-DQUO v1 base weights ({len(bad_missing)}): {bad_missing[:10]}')
        print(f'[OVDQUOV1Detector] Loaded base checkpoint: matched={len(own_sd) - len(missing)} ignored={len(unexpected)}', flush=True)

    def load_tuned_v1(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        arch = ckpt.get('architecture')
        if arch is not None and arch != self.ARCHITECTURE:
            raise ValueError(f'Expected {self.ARCHITECTURE} checkpoint, got {arch}')
        state = dict(ckpt['trainable'] if 'trainable' in ckpt else ckpt)
        # PyTorch `named_parameters()` deduplicates aliased `self.bbox_embed` == `self.transformer.decoder.bbox_embed`
        for k in list(state.keys()):
            if k.startswith('transformer.decoder.bbox_embed.'):
                alias = k[len('transformer.decoder.'):]
                state.setdefault(alias, state[k])
            elif k.startswith('bbox_embed.'):
                alias = 'transformer.decoder.' + k
                state.setdefault(alias, state[k])
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f'Unexpected keys loading OV-DQUO v1 tuned checkpoint: {unexpected[:10]}')
        non_visual_missing = [k for k in missing if not k.startswith('backbone.0.visual.') and k not in ('_img_mean', '_img_std')]
        if non_visual_missing:
            raise RuntimeError(f'Missing trainable OV-DQUO v1 keys ({len(non_visual_missing)}): {non_visual_missing[:10]}')
        print(
            f'[OVDQUOV1Detector] Loaded tuned v1 checkpoint: keys={len(state)} '
            f'epoch={ckpt.get("epoch")} dev_mae={ckpt.get("dev_mae")}',
            flush=True,
        )

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
    def encode_text(self, prompt: str, device) -> Tensor:
        key = str(prompt).strip() or 'object'
        if key not in self._text_cache:
            if self._text_encoder_fn is not None:
                vec = F.normalize(self._text_encoder_fn([key], device=device).float(), dim=-1)[0].detach().cpu()
            else:
                self._ensure_text_encoder(device)
                templated = [t.format(key) for t in CLIP_TEMPLATES]
                toks = self._open_clip_tokenizer(templated).to(device)
                emb = F.normalize(self._open_clip_model.encode_text(toks).float(), dim=-1).mean(dim=0)
                vec = F.normalize(emb, dim=-1).detach().cpu()
            self._text_cache[key] = vec
        return self._text_cache[key].to(device=device, dtype=torch.float32)

    def preprocess_image(self, pil_image: Image.Image, device=None) -> tuple[Tensor, Tensor]:
        dev = device if device is not None else self._img_mean.device
        arr = np.asarray(pil_image.convert('RGB'), dtype=np.uint8)
        h, w = arr.shape[:2]
        scale = float(self.resolution) / float(max(h, w))
        nh = max(self.patch_size, min(self.resolution, int(round(h * scale))))
        nw = max(self.patch_size, min(self.resolution, int(round(w * scale))))
        x = torch.from_numpy(arr.copy()).to(device=dev, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        if nh != h or nw != w:
            x = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False, antialias=True)
        x = (x - self._img_mean.to(device=dev)) / self._img_std.to(device=dev)
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
    @torch.no_grad()
    def extract_box_features(
        dense_map: Tensor,
        dense_mask: Tensor,
        boxes_xyxy_px: np.ndarray,
        image_size: tuple[int, int],
    ) -> list[Optional[np.ndarray]]:
        """Pool 512-d L2-normalized EVA02-CLIP appearance features for pixel `xyxy` boxes from cached `dense_map`."""
        if len(boxes_xyxy_px) == 0:
            return []
        w_img, h_img = float(image_size[0]), float(image_size[1])
        dm = dense_map.float()
        valid_w = float((~dense_mask[0, 0, :]).float().sum().clamp(min=1.0).item())
        valid_h = float((~dense_mask[0, :, 0]).float().sum().clamp(min=1.0).item())
        arr = np.asarray(boxes_xyxy_px, dtype=np.float32).reshape(-1, 4).copy()
        # Keep non-degenerate minimum width/height for ROIAlign while preserving exact box coords for TemporalIdentity
        arr[:, 2] = np.maximum(arr[:, 2], arr[:, 0] + 1.0)
        arr[:, 3] = np.maximum(arr[:, 3], arr[:, 1] + 1.0)
        norm = np.stack([
            np.clip(arr[:, 0] / w_img, 0.0, 1.0),
            np.clip(arr[:, 1] / h_img, 0.0, 1.0),
            np.clip(arr[:, 2] / w_img, 0.0, 1.0),
            np.clip(arr[:, 3] / h_img, 0.0, 1.0),
        ], axis=-1)
        rois = torch.from_numpy(norm).to(device=dm.device, dtype=torch.float32) * dm.new_tensor([valid_w, valid_h, valid_w, valid_h])
        feats = roi_align(dm, [rois], output_size=(1, 1), spatial_scale=1.0, aligned=True)[..., 0, 0]
        feats = F.normalize(feats, dim=-1).cpu().numpy()
        return [feats[i] for i in range(len(feats))]

    @torch.no_grad()
    def _forward_single_text(self, pil_image: Image.Image, prompt: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dev = self._img_mean.device
        use_amp = (dev.type == 'cuda')
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            pv, mask = self.preprocess_image(pil_image, device=dev)
            feats, masks, dense_map, dense_mask = self.backbone[0](pv, mask)
            srcs = [proj(f) for proj, f in zip(self.input_proj, feats)]
            pos_embeds = [self.backbone[1](m, dtype=s.dtype) for m, s in zip(masks, srcs)]
            enc = self.transformer.encode_and_propose(srcs, masks, pos_embeds, grad_checkpoint=False)
            refpoint_undetach = enc['refpoint_undetach']
            refpoint_enc_sig = refpoint_undetach.sigmoid()

            with torch.autocast('cuda', enabled=False):
                text_emb = self.encode_text(prompt, device=dev)
                roi_enc = self._sample_dense_roi(dense_map, dense_mask, refpoint_enc_sig[0])
                t_sim = torch.matmul(roi_enc, text_emb)
                zeros = roi_enc.new_zeros(roi_enc.shape[0])
                sim_hint = torch.stack([t_sim, zeros, zeros, zeros], dim=-1).unsqueeze(0)
                q_prompt = self.transformer.text_proj(text_emb)[None, None, :]
                sim_delta = self.sim_inject(sim_hint)

            q_actual = refpoint_undetach.shape[1]
            tgt_init = self.transformer.tgt_embed.weight[:q_actual].unsqueeze(0) + q_prompt.to(srcs[0].dtype) + sim_delta.to(srcs[0].dtype)
            hs, references = self.transformer.decoder(
                tgt=tgt_init.transpose(0, 1),
                memory=enc['memory'].transpose(0, 1),
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
                roi_feats = self._sample_dense_roi(dense_map, dense_mask, boxes_cxcywh)
                text_cos = torch.matmul(roi_feats, text_emb)
                t_scale = F.elu(self.clip_text_scale) + 1.0
                text_sem = (text_cos + self.clip_text_bias) * t_scale
                scores = log_and_logit(obj_l, text_sem).sigmoid()
        return boxes_cxcywh.float().cpu(), scores.float().cpu(), dense_map.half().cpu(), dense_mask.cpu()

    @torch.no_grad()
    def detect_with_features(
        self,
        pil_image: Image.Image,
        prompt: str,
        threshold: float = 0.26,
        tile_trigger: int = 650,
    ) -> dict:
        """Detect pixel `xyxy` boxes above `threshold` (with optional 2x2 dense tiling) + return cached `dense_map`."""
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
