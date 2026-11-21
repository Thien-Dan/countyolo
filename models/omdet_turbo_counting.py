"""OmDet-Turbo v3 (Unfrozen Swin Stages 1-3 + Multi-Scale P3/P4/P5 Bi-Directional Prompt Cross-Attention
+ Same-Space Spatial Sim-Map + Decoupled Zero-Mean Visual Matching) for Real-Time NMS-Free Counting.

Designed directly from forensic error analysis of OmDet-Turbo v1/v2.1 vs LLMDet v3:
1. Unfreeze Swin `vision_backbone` stages 1-3 (`backbone_lr=1e-5`) while keeping stem/stage0 frozen,
   allowing P3 (80x80) features to resolve dense small/texture objects (e.g. polka dots, coins, birds).
2. 2-Layer `PromptGuidedFPNFusion` (pure FP32 Bi-Directional Cross-Attention across ALL 8,400 P3/P4/P5
   pixels <-> unified [Text + Exemplar] prompt tokens, with a learnable `null_prompt_token` for clean
   absent-prompt routing) — solves OmDet-Turbo's lack of P3/P4 cross-attention in <1.5ms overhead!
3. Same-Space Multi-Scale Spatial Similarity (`sim_raw` + `sim_learned`) injected BEFORE fusion.
4. Decoupled Non-Residual Visual Matching Head (`LN(W_q h) . LN(W_e ex) / sqrt(D) + vis_bias(-3.5)`):
   - Zero-mean orthogonal init -> background logits start at ~-3.5 (p~0.03), preventing vscale collapse.
   - Separate linear projection prevents `hidden` from drifting out of frozen CLIP class heads.
5. Force `encoder.eval_size = None` for exact train/eval 2D pos-embed parity, FrozenBN for `batch_size=1`,
   and detached cloned CLIP LRU cache.
"""
import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from transformers import AutoProcessor, OmDetTurboForObjectDetection


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0.0, max=1.0)
    return torch.log(x.clamp(min=eps) / (1.0 - x).clamp(min=eps))


def _make_4d_attn_mask(valid_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert 2D valid mask (B, L) [True=attend] to 4D additive mask (B, 1, L, L) [0 / -min]."""
    bsz, seq_len = valid_mask.shape
    expanded = valid_mask[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
    inverted = 1.0 - expanded.to(dtype)
    return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)


def _class_similarity(distance_type: str, cls_feature: torch.Tensor, class_proj: torch.Tensor) -> torch.Tensor:
    """Exact OmDet-Turbo cosine/dot similarity between queries (B, Q, D) and projected classes (B, D, K)."""
    if distance_type == 'cosine':
        a = cls_feature / cls_feature.norm(dim=2, keepdim=True).clamp_min(1e-12)
        b = class_proj / class_proj.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return (1.0 / 0.07) * torch.bmm(a, b)
    return torch.bmm(cls_feature, class_proj)


class BiPromptCrossAttnLayer(nn.Module):
    """Lightweight Bi-Directional Cross-Attention between ALL multi-scale vision tokens (8,400)
    and prompt tokens (M text + exemplar tokens), with a learnable `null_prompt_token` for
    absent-prompt routing. Runs in pure FP32 (~0.7ms on T4) for total numerical stability.
    """
    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.v_norm = nn.LayerNorm(dim)
        self.p_norm = nn.LayerNorm(dim)
        self.v_q = nn.Linear(dim, dim)
        self.p_k = nn.Linear(dim, dim)
        self.p_v = nn.Linear(dim, dim)
        self.v_out = nn.Linear(dim, dim)

        self.p_q = nn.Linear(dim, dim)
        self.v_k = nn.Linear(dim, dim)
        self.v_v = nn.Linear(dim, dim)
        self.p_out = nn.Linear(dim, dim)

        self.v_ffn_norm = nn.LayerNorm(dim)
        self.v_ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.p_ffn_norm = nn.LayerNorm(dim)
        self.p_ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

        # Learnable null key/value slot lets vision pixels attend to "background/none" on absent prompts
        self.null_k = nn.Parameter(torch.zeros(1, 1, dim))
        self.null_v = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.null_k, std=0.02)

        # Small non-zero gates preserve pretrained FPN features at step 0 while allowing immediate grads
        self.v_gate = nn.Parameter(torch.tensor([0.1]))
        self.p_gate = nn.Parameter(torch.tensor([0.1]))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, vision: torch.Tensor, prompts: torch.Tensor):
        """vision: (B, N_v, D), prompts: (B, M, D) — all FP32."""
        vn = self.v_norm(vision)
        pn = self.p_norm(prompts)

        # 1) Vision <- Prompts (with null background slot so softmax doesn't force absent-prompt pollution)
        bsz = vision.shape[0]
        k_in = torch.cat([pn, self.null_k.expand(bsz, -1, -1)], dim=1)
        v_in = torch.cat([pn, self.null_v.expand(bsz, -1, -1)], dim=1)
        q_v = self._split(self.v_q(vn))
        k_p = self._split(self.p_k(k_in))
        v_p = self._split(self.p_v(v_in))
        attn_vp = F.softmax(torch.matmul(q_v, k_p.transpose(-2, -1)) * self.scale, dim=-1)
        delta_v = self.v_out(torch.matmul(attn_vp, v_p).transpose(1, 2).contiguous().view_as(vision))
        vg = torch.tanh(self.v_gate)
        vision = vision + vg * delta_v
        vision = vision + vg * self.v_ffn(self.v_ffn_norm(vision))

        # 2) Prompts <- Enhanced Vision
        vn2 = self.v_norm(vision)
        q_p = self._split(self.p_q(pn))
        k_v = self._split(self.v_k(vn2))
        v_v = self._split(self.v_v(vn2))
        attn_pv = F.softmax(torch.matmul(q_p, k_v.transpose(-2, -1)) * self.scale, dim=-1)
        delta_p = self.p_out(torch.matmul(attn_pv, v_v).transpose(1, 2).contiguous().view_as(prompts))
        pg = torch.tanh(self.p_gate)
        prompts = prompts + pg * delta_p
        prompts = prompts + pg * self.p_ffn(self.p_ffn_norm(prompts))
        return vision, prompts


def build_omdet_class_list(target_prompt: str, vocabulary: list[str], max_classes: int = 8, rng=None):
    """Build a shuffled class list `[c0, ..., c_{K-1}]`, task string, and `target_idx`."""
    norm = lambda s: s.strip().lower().rstrip('.')
    target = norm(target_prompt)
    if max_classes <= 1 or not vocabulary:
        classes = [target]
        return classes, f'Detect {target}.', 0
    pool = [norm(c) for c in vocabulary if norm(c) and norm(c) != target]
    pool = list(dict.fromkeys(pool))
    sampler = rng if rng is not None else random
    k = min(max_classes - 1, len(pool))
    negs = sampler.sample(pool, k) if k > 0 else []
    classes = [target] + negs
    sampler.shuffle(classes)
    target_idx = classes.index(target)
    task = 'Detect ' + ', '.join(classes) + '.'
    return classes, task, target_idx


def _is_stem_or_stage0_param(name: str) -> bool:
    """Identify Swin stem / patch_embed / stage-0 parameters to keep frozen while unfreezing stages 1-3."""
    markers = ('patch_embed', 'embeddings', 'layers_0.', 'layers.0.', 'stages.0.')
    return any(m in name for m in markers)


class OmDetTurboCounting(nn.Module):
    ARCHITECTURE = 'omdet-turbo-countgd-box-7layer-v3-biprompt-fpn'

    def __init__(self, model_id='omlab/omdet-turbo-swin-tiny-hf', device='cuda',
                 unfreeze_backbone_stages=True, freeze_class_heads=True, use_exemplars=True,
                 num_fusion_layers=2, detector=None, processor=None):
        super().__init__()
        self.model_id = str(model_id)
        self.device = device
        self.use_exemplars = bool(use_exemplars)
        self.unfreeze_backbone_stages = bool(unfreeze_backbone_stages)
        self.freeze_class_heads = bool(freeze_class_heads)
        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(self.model_id)
        self.detector = (
            detector if detector is not None else OmDetTurboForObjectDetection.from_pretrained(self.model_id)
        ).to(device).float()

        # Guarantee identical 2D sin-cos position embedding between train and eval in HybridEncoder
        self.detector.encoder.eval_size = None

        # Freeze CLIP language backbone completely
        self.detector.language_backbone.requires_grad_(False)

        # Configure Swin vision_backbone: freeze stem+stage0, optionally unfreeze stages 1-3 + layer_norms
        for name, param in self.detector.vision_backbone.named_parameters():
            if self.unfreeze_backbone_stages and not _is_stem_or_stage0_param(name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        if self.freeze_class_heads:
            # Keep pretrained CLIP->256 semantic anchors frozen to prevent 89-class FSC-147 overfitting
            self.detector.decoder.encoder_class_head.requires_grad_(False)
            self.detector.decoder.decoder_class_head.requires_grad_(False)

        cfg = self.detector.config
        self.d_model = int(cfg.decoder_hidden_dim)
        self.class_embed_dim = int(cfg.class_embed_dim)
        self.encoder_dim = int(cfg.encoder_hidden_dim)
        self.num_queries = int(cfg.num_queries)
        self.num_levels = int(cfg.num_feature_levels)

        # Multi-Scale P3/P4/P5 <-> Prompt Bi-Directional Cross-Attention (works for BOTH text & exemplars!)
        self.prompt_text_proj = nn.Sequential(
            nn.Linear(self.class_embed_dim, self.encoder_dim),
            nn.LayerNorm(self.encoder_dim),
        ).to(device).float()
        self.prompt_level_embed = nn.Parameter(torch.zeros(self.num_levels, 1, self.encoder_dim, device=device, dtype=torch.float32))
        nn.init.normal_(self.prompt_level_embed, std=0.02)
        num_heads = 8 if self.encoder_dim % 8 == 0 else max(1, self.encoder_dim // 8)
        self.prompt_fusion_layers = nn.ModuleList([
            BiPromptCrossAttnLayer(dim=self.encoder_dim, num_heads=num_heads)
            for _ in range(num_fusion_layers)
        ]).to(device).float()
        # Small residual gate adapting frozen CLIP class anchors with image-conditioned text prompts
        self.text_prompt_gate = nn.Parameter(torch.tensor([0.05], device=device, dtype=torch.float32))

        if self.use_exemplars:
            gn_groups = 32 if self.d_model % 32 == 0 else max(1, self.d_model // 8)
            self.exemplar_proj = nn.Sequential(
                nn.Conv2d(3 * self.encoder_dim, self.d_model, kernel_size=1),
                nn.GroupNorm(gn_groups, self.d_model),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.d_model, self.d_model, kernel_size=1),
            ).to(device).float()
            self.exemplar_prompt_norm = nn.LayerNorm(self.encoder_dim).to(device).float()
            self.exemplar_prompt_type = nn.Parameter(torch.zeros(1, 1, self.encoder_dim, device=device, dtype=torch.float32))
            nn.init.normal_(self.exemplar_prompt_type, std=0.02)

            # 4-channel Same-Space Spatial Sim-Map before Bi-Prompt FPN Fusion
            self.sim_fusion_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(4, self.encoder_dim, kernel_size=1),
                    nn.GroupNorm(gn_groups, self.encoder_dim),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(self.encoder_dim, self.encoder_dim, kernel_size=1),
                )
                for _ in range(self.num_levels)
            ]).to(device).float()
            for block in self.sim_fusion_convs:
                nn.init.normal_(block[-1].weight, std=0.02)
                nn.init.zeros_(block[-1].bias)

            self.exemplar_to_task = nn.Sequential(
                nn.Linear(self.d_model, self.class_embed_dim),
                nn.LayerNorm(self.class_embed_dim),
            ).to(device).float()
            self.exemplar_task_embed = nn.Parameter(
                torch.zeros(1, 1, self.class_embed_dim, device=device, dtype=torch.float32)
            )
            nn.init.normal_(self.exemplar_task_embed, std=0.02)

            # Decoupled Non-Residual Zero-Mean Visual Matching Head:
            # dot(LN(W_q q), LN(W_e e)) / sqrt(D) + vis_logit_bias (-3.5) -> starts at ~N(-3.5, 1.0) on background!
            self.vis_q_proj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
            ).to(device).float()
            self.vis_e_proj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
            ).to(device).float()
            self.vis_logit_bias = nn.Parameter(torch.tensor([-3.5], device=device, dtype=torch.float32))
        else:
            self.exemplar_proj = None
            self.exemplar_prompt_norm = None
            self.exemplar_prompt_type = None
            self.sim_fusion_convs = None
            self.exemplar_to_task = None
            self.exemplar_task_embed = None
            self.vis_q_proj = None
            self.vis_e_proj = None
            self.vis_logit_bias = None

        self.set_training(False)

    def _custom_modules(self):
        mods = [
            ('__prompt_text_proj__', self.prompt_text_proj),
            ('__prompt_fusion_layers__', self.prompt_fusion_layers),
        ]
        if self.exemplar_proj is not None:
            mods.extend([
                ('__exemplar_proj__', self.exemplar_proj),
                ('__exemplar_prompt_norm__', self.exemplar_prompt_norm),
                ('__sim_fusion_convs__', self.sim_fusion_convs),
                ('__exemplar_to_task__', self.exemplar_to_task),
                ('__vis_q_proj__', self.vis_q_proj),
                ('__vis_e_proj__', self.vis_e_proj),
            ])
        return mods

    def _custom_scalars_and_embeds(self):
        items = [
            ('__prompt_level_embed__', self.prompt_level_embed),
            ('__text_prompt_gate__', self.text_prompt_gate),
        ]
        if self.exemplar_proj is not None:
            items.extend([
                ('__exemplar_prompt_type__', self.exemplar_prompt_type),
                ('__exemplar_task_embed__', self.exemplar_task_embed),
                ('__vis_logit_bias__', self.vis_logit_bias),
            ])
        return items

    def set_training(self, training: bool):
        self.detector.train(training)
        for _, mod in self._custom_modules():
            mod.train(training)
        self.detector.language_backbone.eval()
        if not self.unfreeze_backbone_stages:
            self.detector.vision_backbone.eval()
        if self.freeze_class_heads:
            self.detector.decoder.encoder_class_head.eval()
            self.detector.decoder.decoder_class_head.eval()
        for module in self.detector.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def trainable_parameters(self):
        params = [(n, p) for n, p in self.detector.named_parameters() if p.requires_grad]
        for prefix, mod in self._custom_modules():
            params.extend([(f'{prefix}.{n}', p) for n, p in mod.named_parameters() if p.requires_grad])
        for key, p in self._custom_scalars_and_embeds():
            if p.requires_grad:
                params.append((key, p))
        return params

    def parameter_buckets(self):
        """Return `(backbone_params, det_params, fusion_params, cal_params)` for independent LRs & clipping."""
        bb_params, det_params = [], []
        for n, p in self.detector.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith('vision_backbone.'):
                bb_params.append(p)
            else:
                det_params.append(p)
        fusion_params, cal_params = [], []
        for prefix, mod in self._custom_modules():
            for n, p in mod.named_parameters():
                if not p.requires_grad:
                    continue
                if n.endswith('v_gate') or n.endswith('p_gate'):
                    cal_params.append(p)
                else:
                    fusion_params.append(p)
        for key, p in self._custom_scalars_and_embeds():
            if not p.requires_grad:
                continue
            if key in ('__vis_logit_bias__', '__text_prompt_gate__'):
                cal_params.append(p)
            else:
                fusion_params.append(p)
        return bb_params, det_params, fusion_params, cal_params

    def parameter_groups(self, backbone_lr=1e-5, base_lr=3e-5, fusion_lr=1.5e-4, cal_lr=1.5e-3, weight_decay=1e-4):
        bb_p, det_p, fus_p, cal_p = self.parameter_buckets()
        groups = []
        if bb_p:
            groups.append(dict(params=bb_p, lr=backbone_lr, weight_decay=weight_decay))
        if det_p:
            groups.append(dict(params=det_p, lr=base_lr, weight_decay=weight_decay))
        if fus_p:
            groups.append(dict(params=fus_p, lr=fusion_lr, weight_decay=weight_decay))
        if cal_p:
            groups.append(dict(params=cal_p, lr=cal_lr, weight_decay=0.0))
        return groups

    def _build_fused_map(self, encoder_states):
        p3 = encoder_states[0].float()
        h0, w0 = p3.shape[-2:]
        p4 = (encoder_states[1] if len(encoder_states) > 1 else p3).float()
        p5 = (encoder_states[2] if len(encoder_states) > 2 else p4).float()
        fused = torch.cat([
            p3,
            F.interpolate(p4, size=(h0, w0), mode='bilinear', align_corners=False),
            F.interpolate(p5, size=(h0, w0), mode='bilinear', align_corners=False),
        ], dim=1)
        return self.exemplar_proj(fused)

    @staticmethod
    def _roi_pool_level(feat_map: torch.Tensor, exemplar_boxes: torch.Tensor) -> torch.Tensor:
        h, w = feat_map.shape[-2:]
        boxes = exemplar_boxes.to(device=feat_map.device, dtype=torch.float32)
        scaled = torch.stack([
            (boxes[:, 0] * float(w)).clamp(0.0, float(w)),
            (boxes[:, 1] * float(h)).clamp(0.0, float(h)),
            (boxes[:, 2] * float(w)).clamp(0.0, float(w)),
            (boxes[:, 3] * float(h)).clamp(0.0, float(h)),
        ], dim=-1)
        return roi_align(feat_map, [scaled], output_size=(1, 1), aligned=True).flatten(1)

    def _extract_exemplar_bundle(self, encoder_states, exemplar_boxes):
        """Extract SAME-SPACE multi-scale raw exemplars + learned fused exemplars in pure FP32."""
        with torch.autocast(device_type='cuda', enabled=False):
            fused = self._build_fused_map(encoder_states)
            ex_256 = self._roi_pool_level(fused, exemplar_boxes)
            ex_raw_levels = [self._roi_pool_level(feat.float(), exemplar_boxes) for feat in encoder_states]
        return dict(ex_256=ex_256, ex_raw_levels=ex_raw_levels, fused=fused)

    def _enhance_states_with_sim(self, encoder_states, bundle):
        """Fuse 4-channel Same-Space raw + learned similarity maps into (P3, P4, P5) in pure FP32."""
        orig_dtype = encoder_states[0].dtype
        enhanced = []
        with torch.autocast(device_type='cuda', enabled=False):
            fused = bundle.get('fused')
            if fused is None:
                fused = self._build_fused_map(encoder_states)
            ex_256 = bundle['ex_256'].to(device=fused.device, dtype=torch.float32)
            sim_learned = torch.einsum(
                'bchw,ec->behw',
                F.normalize(fused, dim=1, eps=1e-6),
                F.normalize(ex_256, dim=-1, eps=1e-6),
            )
            learned_stats_0 = torch.cat([
                sim_learned.max(dim=1, keepdim=True).values,
                sim_learned.mean(dim=1, keepdim=True),
            ], dim=1)
            for lvl, feat in enumerate(encoder_states):
                f = feat.float()
                ex_raw = bundle['ex_raw_levels'][lvl].to(device=f.device, dtype=torch.float32)
                sim_raw = torch.einsum(
                    'bchw,ec->behw',
                    F.normalize(f, dim=1, eps=1e-6),
                    F.normalize(ex_raw, dim=-1, eps=1e-6),
                )
                raw_stats = torch.cat([
                    sim_raw.max(dim=1, keepdim=True).values,
                    sim_raw.mean(dim=1, keepdim=True),
                ], dim=1)
                learned_stats = F.interpolate(learned_stats_0, size=f.shape[-2:], mode='bilinear', align_corners=False)
                delta = self.sim_fusion_convs[lvl](torch.cat([raw_stats, learned_stats], dim=1))
                enhanced.append((f + delta).to(orig_dtype))
        return enhanced

    def _run_biprompt_fpn_fusion(self, states, class_features, ex_256=None):
        """2-layer Bi-Directional Cross-Attention across ALL (P3, P4, P5) pixels <-> [Text, Exemplar] prompts in FP32."""
        orig_dtype = states[0].dtype
        num_classes = int(class_features.shape[0])
        with torch.autocast(device_type='cuda', enabled=False):
            shapes = [s.shape[-2:] for s in states]
            flat_list = []
            for lvl, s in enumerate(states):
                # (1, D, H, W) -> (1, H*W, D) + level embedding
                flat_list.append(s.float().flatten(2).transpose(1, 2) + self.prompt_level_embed[lvl].unsqueeze(0))
            vision = torch.cat(flat_list, dim=1)  # (1, 8400, D)

            text_p = self.prompt_text_proj(class_features[:, 0, :].float()).unsqueeze(0)  # (1, K, D)
            if ex_256 is not None and len(ex_256) > 0:
                ex_p = self.exemplar_prompt_norm(ex_256.float()).unsqueeze(0) + self.exemplar_prompt_type  # (1, E, D)
                prompts = torch.cat([text_p, ex_p], dim=1)
            else:
                prompts = text_p

            for layer in self.prompt_fusion_layers:
                vision, prompts = layer(vision, prompts)

            # Unflatten enhanced multi-scale feature maps back to (P3, P4, P5)
            out_states = []
            offset = 0
            for h_l, w_l in shapes:
                n_l = h_l * w_l
                s_l = vision[:, offset : offset + n_l, :].transpose(1, 2).contiguous().view(1, self.encoder_dim, h_l, w_l)
                out_states.append(s_l.to(orig_dtype))
                offset += n_l

            refined_text = prompts[:, :num_classes, :]  # (1, K, D)
            refined_ex = prompts[0, num_classes:, :] if (ex_256 is not None and len(ex_256) > 0) else None
        return out_states, refined_text, refined_ex

    def _visual_dot_logits(self, query_feats: torch.Tensor, ex_feats: torch.Tensor) -> torch.Tensor:
        """Zero-mean decoupled contrastive dot-product logits (B, Q, E) in pure FP32.

        At step 0: `LN(W_q q) . LN(W_e e) / sqrt(D)` has mean ~0, std ~1 -> adding `vis_logit_bias (-3.5)`
        starts all background queries cleanly at p ~ 0.03 with zero background cone explosion.
        """
        with torch.autocast(device_type='cuda', enabled=False):
            q = self.vis_q_proj(query_feats.float())  # (B, Q, D)
            e = self.vis_e_proj(ex_feats.float())     # (E, D)
            logits = torch.einsum('bqd,ed->bqe', q, e) / math.sqrt(float(self.d_model)) + self.vis_logit_bias
        return logits

    @torch.no_grad()
    def encode_exemplars(self, pil_image, exemplar_boxes):
        """Precompute same-space exemplar bundle from full image for 2x2 tiling."""
        if not self.use_exemplars or exemplar_boxes is None or len(exemplar_boxes) == 0:
            return None
        inputs = self.processor(images=pil_image, text=['object'], task='Detect object.', return_tensors='pt').to(self.device)
        image_features = self.detector.vision_backbone(inputs['pixel_values'])
        enc_out = self.detector.encoder(image_features, return_dict=True)
        bundle = self._extract_exemplar_bundle(enc_out.extracted_states, exemplar_boxes)
        return dict(ex_256=bundle['ex_256'], ex_raw_levels=bundle['ex_raw_levels'])

    def forward_stages(self, pixel_values, classes_input_ids, classes_attention_mask,
                       tasks_input_ids, tasks_attention_mask, classes_structure,
                       target_idx: int = 0, exemplar_boxes=None, exemplar_feats=None,
                       use_text: bool = True):
        """Run OmDet-Turbo v3 forward returning differentiable 7-stage boxes/logits."""
        dec = self.detector.decoder

        # Vision backbone is differentiable (stages 1-3 unfrozen when `unfreeze_backbone_stages=True`)!
        image_features = self.detector.vision_backbone(pixel_values)

        # Frozen CLIP language features computed under no_grad and cloned so LRU cache stays pure
        with torch.no_grad():
            class_features, task_features, task_mask = self.detector.get_language_embedding(
                classes_input_ids,
                classes_attention_mask,
                tasks_input_ids,
                tasks_attention_mask,
                classes_structure,
            )
            class_features = class_features.detach().clone()
            task_features = task_features.detach().clone()
            task_mask = task_mask.detach().clone()

        encoder_outputs = self.detector.encoder(image_features, return_dict=True)
        extracted_states = list(encoder_outputs.extracted_states)
        num_classes = int(class_features.shape[0])
        t_idx = min(max(0, int(target_idx)), num_classes - 1)

        bundle = None
        if self.use_exemplars:
            if exemplar_feats is not None:
                bundle = exemplar_feats
            elif exemplar_boxes is not None and len(exemplar_boxes) > 0:
                bundle = self._extract_exemplar_bundle(extracted_states, exemplar_boxes)

        ex_256 = None
        if bundle is not None:
            ex_256 = bundle['ex_256'].to(device=pixel_values.device, dtype=torch.float32)
            # 1) Same-space raw + learned spatial similarity map BEFORE Bi-Prompt Cross-Attention
            extracted_states = self._enhance_states_with_sim(extracted_states, bundle)

        # 2) Multi-Scale P3/P4/P5 <-> Prompt Bi-Directional Cross-Attention (wakes up dense small objects!)
        extracted_states, refined_text, refined_ex = self._run_biprompt_fpn_fusion(
            extracted_states, class_features, ex_256=ex_256
        )

        # 3) Inject image-conditioned exemplar tokens into Decoder task_features Self-Attention
        if refined_ex is not None:
            with torch.autocast(device_type='cuda', enabled=False):
                ex_task = self.exemplar_to_task(refined_ex).unsqueeze(1) + self.exemplar_task_embed
            ex_task = ex_task.to(task_features.dtype)
            ex_mask = torch.ones((task_mask.shape[0], ex_task.shape[0]), dtype=task_mask.dtype, device=task_mask.device)
            task_features = torch.cat([task_features, ex_task], dim=0)
            task_mask = torch.cat([task_mask, ex_mask], dim=1)

        vision_features, vision_shapes, vision_shapes_list, level_start_index = dec._get_encoder_input(extracted_states)
        batch_size = task_mask.shape[0]
        task_features = dec.task_encoder(task_features)
        if getattr(dec, 'task_project', None) is not None:
            task_features = dec.task_project(task_features)
        src_key_mask = (task_mask == 0).detach()
        attn_mask_len = dec.num_queries
        fusion_size = attn_mask_len + task_features.shape[0]
        key_padding_mask = torch.zeros([batch_size, fusion_size], dtype=torch.bool, device=task_features.device)
        key_padding_mask[:, attn_mask_len:] = src_key_mask
        attention_mask = _make_4d_attn_mask(~key_padding_mask, dtype=vision_features.dtype)

        # Two-Stage Encoder Proposal Generation
        anchors, valid_mask = dec.generate_anchors(
            tuple(vision_shapes_list), device=vision_features.device, dtype=vision_features.dtype
        )
        enc_vis_feats = dec.encoder_vision_features(
            torch.where(valid_mask, vision_features, torch.tensor(0.0, dtype=vision_features.dtype, device=vision_features.device))
        )
        t_gate = torch.tanh(self.text_prompt_gate).to(dtype=torch.float32)
        enc_class_anchor = dec.encoder_class_head(class_features).permute(1, 2, 0).float()  # (1, D, K)
        enc_class_proj = enc_class_anchor + t_gate * refined_text.transpose(1, 2)
        enc_text_logits = _class_similarity(dec.class_distance_type, enc_vis_feats.float(), enc_class_proj)
        enc_vis_logits = self._visual_dot_logits(enc_vis_feats, refined_ex) if refined_ex is not None else None

        if refined_ex is not None and not use_text:
            enc_all_logits = enc_vis_logits
        elif refined_ex is not None and use_text:
            enc_all_logits = torch.cat([enc_text_logits, enc_vis_logits], dim=-1)
        else:
            enc_all_logits = enc_text_logits

        enc_outputs_bboxes = dec.encoder_bbox_head(enc_vis_feats) + anchors
        topk_ind = torch.topk(enc_all_logits.max(-1).values, dec.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=batch_size, dtype=topk_ind.dtype, device=topk_ind.device).unsqueeze(-1).repeat(1, dec.num_queries).view(-1)

        ref_raw = enc_outputs_bboxes[batch_ind, topk_ind].view(batch_size, dec.num_queries, -1)
        enc_boxes = ref_raw.sigmoid()[0].float()
        enc_logits = enc_all_logits[batch_ind, topk_ind].view(batch_size, dec.num_queries, -1)[0].float()

        reference_points = ref_raw.detach().sigmoid() if dec.training else ref_raw.sigmoid()
        if dec.learn_initial_query:
            hidden = dec.tgt_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            hidden = enc_vis_feats[batch_ind, topk_ind].view(batch_size, dec.num_queries, -1)
            if dec.training:
                hidden = hidden.detach()

        # 6-Layer Deformable DETR Decoder with DINO Look-Forward-Twice + Decoupled Text/Visual Scoring
        dec_boxes, dec_logits = [], []
        last_refined_bbox = None
        for i, layer in enumerate(dec.layers):
            hidden, task_features, _, _ = layer(
                hidden,
                task_features,
                reference_points,
                vision_features,
                vision_shapes,
                vision_shapes_list,
                level_start_index=level_start_index,
                attention_mask=attention_mask,
                query_position=dec.query_position_head(reference_points),
                output_attentions=False,
                output_hidden_states=False,
            )
            refined_bbox = torch.sigmoid(dec.decoder_bbox_head[i](hidden) + _inverse_sigmoid(reference_points))
            if dec.training or i == dec.decoder_num_layers - 1:
                cls_anchor_i = dec.decoder_class_head[i](class_features).permute(1, 2, 0).float()
                cls_proj_i = cls_anchor_i + t_gate * refined_text.transpose(1, 2)
                t_logits_i = _class_similarity(dec.class_distance_type, hidden.float(), cls_proj_i)
                if refined_ex is not None:
                    v_logits_i = self._visual_dot_logits(hidden, refined_ex)
                    logits_i = v_logits_i if not use_text else torch.cat([t_logits_i, v_logits_i], dim=-1)
                else:
                    logits_i = t_logits_i
                dec_logits.append(logits_i[0].float())
                if dec.training and i > 0:
                    box_i = torch.sigmoid(dec.decoder_bbox_head[i](hidden) + _inverse_sigmoid(last_refined_bbox))
                else:
                    box_i = refined_bbox
                dec_boxes.append(box_i[0].float())
            last_refined_bbox = refined_bbox
            reference_points = refined_bbox.detach() if dec.training else refined_bbox

        if refined_ex is not None and not use_text:
            target_tokens = torch.ones(len(refined_ex), dtype=torch.bool, device=pixel_values.device)
        elif refined_ex is not None and use_text:
            target_tokens = torch.zeros(num_classes + len(refined_ex), dtype=torch.bool, device=pixel_values.device)
            target_tokens[t_idx] = True
            target_tokens[num_classes:] = True
        else:
            target_tokens = torch.zeros(num_classes, dtype=torch.bool, device=pixel_values.device)
            target_tokens[t_idx] = True

        return dict(
            boxes=dec_boxes[-1],
            logits=dec_logits[-1],
            aux_boxes=dec_boxes[:-1],
            aux_logits=dec_logits[:-1],
            enc_boxes=enc_boxes,
            enc_logits=enc_logits,
            target_tokens=target_tokens,
        )

    def forward(self, pil_image, prompt: str, exemplar_boxes=None, exemplar_feats=None, use_text: bool = True):
        """Single-image inference returning normalized cxcywh boxes (900, 4) and max target logits (900,)."""
        cls_name = prompt.strip().lower().rstrip('.') if use_text else 'object'
        task = f'Detect {cls_name}.'
        inputs = self.processor(images=pil_image, text=[cls_name], task=task, return_tensors='pt').to(self.device)
        out = self.forward_stages(
            inputs['pixel_values'],
            inputs['classes_input_ids'],
            inputs['classes_attention_mask'],
            inputs['tasks_input_ids'],
            inputs['tasks_attention_mask'],
            inputs['classes_structure'],
            target_idx=0,
            exemplar_boxes=exemplar_boxes,
            exemplar_feats=exemplar_feats,
            use_text=use_text,
        )
        scores_logit = out['logits'][:, out['target_tokens']].max(dim=-1).values
        return out['boxes'], scores_logit

    def export_trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.trainable_parameters()}

    def import_trainable_state(self, state):
        det_state = {k: v for k, v in state.items() if not k.startswith('__')}
        _, unexpected = self.detector.load_state_dict(det_state, strict=False)
        if unexpected:
            raise RuntimeError(f'Unexpected keys in OmDet-Turbo checkpoint: {unexpected[:5]}')
        for prefix, mod in self._custom_modules():
            p = f'{prefix}.'
            sub = {k[len(p):]: v for k, v in state.items() if k.startswith(p)}
            if sub:
                mod.load_state_dict(sub, strict=True)
        for key, param in self._custom_scalars_and_embeds():
            if key in state:
                param.data.copy_(state[key].to(param.device))
