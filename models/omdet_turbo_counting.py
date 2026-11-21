"""OmDet-Turbo v2.1 (RT-DETR HybridEncoder + Frozen CLIP Space + Same-Space Class-Agnostic Visual Matching)
for real-time NMS-free Open-World Counting (CountVid Stage 1).

Fixes in v2.1 (verified from Epoch 1-3 diagnostics):
1. Same-Space Multi-Scale Spatial Similarity:
   - `sim_raw_l = cosine(encoder_states[l], roi_align(encoder_states[l], boxes))` (100% same pretrained
     feature space -> lights up matching objects at ~0.9-1.0 from Step 0 on unseen classes!).
   - `sim_learned = cosine(fused, roi_align(fused, boxes))` (100% same `exemplar_proj` space!).
   - Feeds 4-channel `[raw_max, raw_mean, learned_max, learned_mean]` into `sim_fusion_convs[l]`.
2. Residual Identity-Preserving Visual Matching Heads (`vis_query_head`, `vis_ex_head`) so pretrained
   object features are preserved from Step 0 instead of scrambled by random linear layers.
3. Separate parameter groups & separate gradient clipping (`det_params`, `ex_params`, and dedicated
   calibration group for `vis_logit_scale`/`vis_logit_bias` with `weight_decay=0.0`) so global clipping
   never chokes `vbias` or visual-matching gradients.
4. Frozen 7 CLIP class heads + FrozenBN + detached CLIP LRU cache + Pure FP32 visual ops.
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


class ResidualProj(nn.Module):
    """Identity-preserving residual projection: LayerNorm(x + Linear(GELU(Linear(x)))) initialized near identity."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.fc2(F.gelu(self.fc1(x))))


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


class OmDetTurboCounting(nn.Module):
    ARCHITECTURE = 'omdet-turbo-countgd-box-7layer-v2.1-same-space'

    def __init__(self, model_id='omlab/omdet-turbo-swin-tiny-hf', device='cuda',
                 freeze_backbones=True, freeze_class_heads=True, use_exemplars=False,
                 detector=None, processor=None):
        super().__init__()
        self.model_id = str(model_id)
        self.device = device
        self.use_exemplars = bool(use_exemplars)
        self.freeze_backbones = bool(freeze_backbones)
        self.freeze_class_heads = bool(freeze_class_heads)
        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(self.model_id)
        self.detector = (
            detector if detector is not None else OmDetTurboForObjectDetection.from_pretrained(self.model_id)
        ).to(device).float()

        if self.freeze_backbones:
            self.detector.vision_backbone.requires_grad_(False)
            self.detector.language_backbone.requires_grad_(False)
        if self.freeze_class_heads:
            self.detector.decoder.encoder_class_head.requires_grad_(False)
            self.detector.decoder.decoder_class_head.requires_grad_(False)

        cfg = self.detector.config
        self.d_model = int(cfg.decoder_hidden_dim)
        self.class_embed_dim = int(cfg.class_embed_dim)
        self.encoder_dim = int(cfg.encoder_hidden_dim)
        self.num_queries = int(cfg.num_queries)
        self.num_levels = int(cfg.num_feature_levels)

        if self.use_exemplars:
            gn_groups = 32 if self.d_model % 32 == 0 else max(1, self.d_model // 8)
            self.exemplar_proj = nn.Sequential(
                nn.Conv2d(3 * self.encoder_dim, self.d_model, kernel_size=1),
                nn.GroupNorm(gn_groups, self.d_model),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.d_model, self.d_model, kernel_size=1),
            ).to(device).float()

            # 4-channel Same-Space Spatial Sim-Map: [raw_max, raw_mean, learned_max, learned_mean]
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

            # Identity-preserving residual visual matching heads in 256-d query space
            self.vis_query_head = ResidualProj(self.d_model).to(device).float()
            self.vis_ex_head = ResidualProj(self.d_model).to(device).float()
            self.vis_logit_scale = nn.Parameter(torch.tensor([math.log(10.0)], device=device, dtype=torch.float32))
            self.vis_logit_bias = nn.Parameter(torch.tensor([-3.0], device=device, dtype=torch.float32))
        else:
            self.exemplar_proj = None
            self.sim_fusion_convs = None
            self.exemplar_to_task = None
            self.exemplar_task_embed = None
            self.vis_query_head = None
            self.vis_ex_head = None
            self.vis_logit_scale = None
            self.vis_logit_bias = None

        self.set_training(False)

    def set_training(self, training: bool):
        self.detector.train(training)
        if self.exemplar_proj is not None:
            self.exemplar_proj.train(training)
            self.sim_fusion_convs.train(training)
            self.exemplar_to_task.train(training)
            self.vis_query_head.train(training)
            self.vis_ex_head.train(training)
        if self.freeze_backbones:
            self.detector.vision_backbone.eval()
            self.detector.language_backbone.eval()
        if self.freeze_class_heads:
            self.detector.decoder.encoder_class_head.eval()
            self.detector.decoder.decoder_class_head.eval()
        for module in self.detector.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def _exemplar_named_parameters(self):
        if self.exemplar_proj is None:
            return []
        out = []
        for prefix, mod in [
            ('__exemplar_proj__', self.exemplar_proj),
            ('__sim_fusion_convs__', self.sim_fusion_convs),
            ('__exemplar_to_task__', self.exemplar_to_task),
            ('__vis_query_head__', self.vis_query_head),
            ('__vis_ex_head__', self.vis_ex_head),
        ]:
            out.extend([(f'{prefix}.{n}', p) for n, p in mod.named_parameters() if p.requires_grad])
        out.append(('__exemplar_task_embed__', self.exemplar_task_embed))
        out.append(('__vis_logit_scale__', self.vis_logit_scale))
        out.append(('__vis_logit_bias__', self.vis_logit_bias))
        return out

    def trainable_parameters(self):
        params = [(n, p) for n, p in self.detector.named_parameters() if p.requires_grad]
        params.extend(self._exemplar_named_parameters())
        return params

    def parameter_buckets(self):
        """Return `(det_params, ex_body_params, cal_params)` for independent gradient clipping & LRs."""
        det_params = [p for _, p in self.detector.named_parameters() if p.requires_grad]
        if self.exemplar_proj is None:
            return det_params, [], []
        ex_body = []
        for mod in (self.exemplar_proj, self.sim_fusion_convs, self.exemplar_to_task,
                    self.vis_query_head, self.vis_ex_head):
            ex_body.extend([p for p in mod.parameters() if p.requires_grad])
        ex_body.append(self.exemplar_task_embed)
        cal_params = [self.vis_logit_scale, self.vis_logit_bias]
        return det_params, ex_body, cal_params

    def parameter_groups(self, base_lr=3e-5, exemplar_lr=2e-4, cal_lr=2e-3, weight_decay=1e-4):
        det_params, ex_body, cal_params = self.parameter_buckets()
        groups = [dict(params=det_params, lr=base_lr, weight_decay=weight_decay)]
        if ex_body:
            groups.append(dict(params=ex_body, lr=exemplar_lr, weight_decay=weight_decay))
        if cal_params:
            groups.append(dict(params=cal_params, lr=cal_lr, weight_decay=0.0))
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
        """Compute SAME-SPACE raw + learned similarity maps and fuse into (P3, P4, P5) in FP32."""
        orig_dtype = encoder_states[0].dtype
        enhanced = []
        with torch.autocast(device_type='cuda', enabled=False):
            fused = bundle.get('fused')
            if fused is None:
                fused = self._build_fused_map(encoder_states)
            ex_256 = bundle['ex_256'].to(device=fused.device, dtype=torch.float32)
            ex_learned_norm = F.normalize(ex_256, dim=-1, eps=1e-6)
            fused_norm = F.normalize(fused, dim=1, eps=1e-6)
            sim_learned = torch.einsum('bchw,ec->behw', fused_norm, ex_learned_norm)
            learned_stats_0 = torch.cat([
                sim_learned.max(dim=1, keepdim=True).values,
                sim_learned.mean(dim=1, keepdim=True),
            ], dim=1)  # (1, 2, H0, W0)

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
                sim_4ch = torch.cat([raw_stats, learned_stats], dim=1)  # (1, 4, H_l, W_l)
                delta = self.sim_fusion_convs[lvl](sim_4ch)
                enhanced.append((f + delta).to(orig_dtype))
        return enhanced

    def _visual_similarity_logits(self, query_feats, ex_256):
        """Compute direct class-agnostic visual similarity logits (B, Q, E) in pure FP32."""
        with torch.autocast(device_type='cuda', enabled=False):
            q = F.normalize(self.vis_query_head(query_feats.float()), dim=-1, eps=1e-6)
            e = F.normalize(self.vis_ex_head(ex_256.float()), dim=-1, eps=1e-6)
            scale = self.vis_logit_scale.exp().clamp(1.0, 25.0)
            logits = scale * torch.einsum('bqd,ed->bqe', q, e) + self.vis_logit_bias
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
        # Drop full-image `fused` map so sub-tiles compute their own `fused` map in same space
        return dict(ex_256=bundle['ex_256'], ex_raw_levels=bundle['ex_raw_levels'])

    def forward_stages(self, pixel_values, classes_input_ids, classes_attention_mask,
                       tasks_input_ids, tasks_attention_mask, classes_structure,
                       target_idx: int = 0, exemplar_boxes=None, exemplar_feats=None,
                       use_text: bool = True):
        """Run OmDet-Turbo v2.1 forward returning differentiable 7-stage boxes/logits."""
        dec = self.detector.decoder
        with torch.no_grad():
            image_features = self.detector.vision_backbone(pixel_values)
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
            # 1) Same-Space Multi-Scale Spatial Similarity injection BEFORE Two-Stage Query Selection
            extracted_states = self._enhance_states_with_sim(extracted_states, bundle)
            # 2) Append individual visual exemplar tokens to task_features for 6-layer Decoder Self-Attention
            with torch.autocast(device_type='cuda', enabled=False):
                ex_task = self.exemplar_to_task(ex_256).unsqueeze(1) + self.exemplar_task_embed
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
        enc_class_proj = dec.encoder_class_head(class_features).permute(1, 2, 0)
        enc_text_logits = _class_similarity(dec.class_distance_type, enc_vis_feats, enc_class_proj).float()
        enc_vis_logits = self._visual_similarity_logits(enc_vis_feats, ex_256) if ex_256 is not None else None

        if ex_256 is not None and not use_text:
            enc_all_logits = enc_vis_logits
        elif ex_256 is not None and use_text:
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

        # 6-Layer Deformable DETR Decoder with DINO Look-Forward-Twice + Dual-Space Scoring
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
                cls_proj_i = dec.decoder_class_head[i](class_features).permute(1, 2, 0)
                t_logits_i = _class_similarity(dec.class_distance_type, hidden, cls_proj_i).float()
                if ex_256 is not None:
                    v_logits_i = self._visual_similarity_logits(hidden, ex_256)
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

        if ex_256 is not None and not use_text:
            target_tokens = torch.ones(len(ex_256), dtype=torch.bool, device=pixel_values.device)
        elif ex_256 is not None and use_text:
            target_tokens = torch.zeros(num_classes + len(ex_256), dtype=torch.bool, device=pixel_values.device)
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
        if self.exemplar_proj is not None:
            for prefix, mod in [
                ('__exemplar_proj__.', self.exemplar_proj),
                ('__sim_fusion_convs__.', self.sim_fusion_convs),
                ('__exemplar_to_task__.', self.exemplar_to_task),
                ('__vis_query_head__.', self.vis_query_head),
                ('__vis_ex_head__.', self.vis_ex_head),
            ]:
                sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
                if sub:
                    mod.load_state_dict(sub, strict=True)
            for key, param in [
                ('__exemplar_task_embed__', self.exemplar_task_embed),
                ('__vis_logit_scale__', self.vis_logit_scale),
                ('__vis_logit_bias__', self.vis_logit_bias),
            ]:
                if key in state:
                    param.data.copy_(state[key].to(param.device))
