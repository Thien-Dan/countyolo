"""LLMDet (MM-Grounding-DINO) adaptation with CountGD-Box 7-layer deep supervision,
single-forward multi-class negative captioning, and optional ROIAlign visual exemplar injection.

Architecture (`iSEE-Laboratory/llmdet_tiny` / `llmdet_base` / `llmdet_large`):
- Frozen Swin visual backbone + frozen BERT language backbone
- Trainable 6-layer Bi-directional Feature Enhancer Encoder
- Trainable Language-Guided Query Selection (`enc_output`, `encoder_output_class_embed`, `encoder_output_bbox_embed`)
- Trainable 6-layer Deformable DETR Decoder (900 queries, 100% NMS-free) with differentiable iterative
  box refinement (`bbox_embed[0..5]`) and contrastive classification (`class_embed[0..5]`)
- Optional Visual Exemplar Projector (`exemplar_proj` + `exemplar_norm` + `exemplar_type_embed`)
  extracting ROIAlign(1x1) embeddings aligned to Swin `valid_ratios` and injecting them as pseudo-tokens
  directly into `text_features` inside the target phrase's block-diagonal self-attention span.
"""
import random
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

SPECIAL_TOKEN_IDS = (101, 102, 1012, 1029, 0)  # [CLS], [SEP], '.', '?', [PAD]
DELIMITER_TOKEN_IDS = (101, 102, 1012, 1029)
EXEMPLAR_TOKEN_ID = 1008  # '*' token id in BERT vocab (non-special, stays in target phrase block)


def token_span_mask(input_ids: torch.LongTensor) -> torch.BoolTensor:
    """Return boolean mask (seq_len,) selecting non-special word and exemplar tokens."""
    ids = input_ids[0] if input_ids.ndim == 2 else input_ids
    special = torch.tensor(SPECIAL_TOKEN_IDS, device=ids.device)
    mask = ~torch.isin(ids, special)
    if not mask.any():
        mask = ids != 0
    return mask


def generate_text_masks(input_ids: torch.LongTensor):
    """Build block-diagonal self-attention mask (B, L, L) and per-phrase position_ids (B, L).

    Matches HuggingFace `MMGroundingDinoModel`'s `generate_masks_with_special_tokens_and_transfer_map`:
    each phrase (including its trailing '.' delimiter, plus any injected visual exemplar tokens)
    attends bidirectionally within its own block while remaining isolated from all other phrases.
    """
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    special = torch.tensor(DELIMITER_TOKEN_IDS, device=device)
    is_delim = torch.isin(input_ids, special)

    attn_mask = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0).repeat(bsz, 1, 1)
    pos_ids = torch.zeros((bsz, seq_len), dtype=torch.long, device=device)
    for b in range(bsz):
        idxs = torch.nonzero(is_delim[b], as_tuple=False).flatten().tolist()
        prev = 0
        for col in idxs:
            if col == 0 or col == seq_len - 1:
                attn_mask[b, col, col] = True
                pos_ids[b, col] = 0
            else:
                attn_mask[b, prev + 1 : col + 1, prev + 1 : col + 1] = True
                pos_ids[b, prev + 1 : col + 1] = torch.arange(0, col - prev, device=device)
            prev = col
    return attn_mask, pos_ids


def build_multiclass_caption(target_prompt: str, vocabulary: list[str], max_classes: int = 8, rng=None):
    """Build a GroundingDINO/CountGD single-forward multi-class caption (`c1 . c2 . ... .`).

    Returns `(caption, target_phrase_idx)` where `target_phrase_idx` is the 0-based index of
    `target_prompt` among the '.'-separated phrases.
    """
    norm = lambda s: s.strip().lower().rstrip('.')
    target = norm(target_prompt)
    if max_classes <= 1 or not vocabulary:
        return f'{target} .', 0
    pool = [norm(c) for c in vocabulary if norm(c) and norm(c) != target]
    pool = list(dict.fromkeys(pool))
    sampler = rng if rng is not None else random
    k = min(max_classes - 1, len(pool))
    negs = sampler.sample(pool, k) if k > 0 else []
    phrases = [target] + negs
    sampler.shuffle(phrases)
    target_idx = phrases.index(target)
    return ' . '.join(phrases) + ' .', target_idx


def phrase_token_span(input_ids: torch.LongTensor, phrase_idx: int) -> tuple[int, int]:
    """Return `(start, end)` token indices of the `phrase_idx`-th non-special phrase."""
    ids = input_ids[0] if input_ids.ndim == 2 else input_ids
    special = set(SPECIAL_TOKEN_IDS)
    spans = []
    start = None
    for i, tok in enumerate(ids.tolist()):
        if tok in special:
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(ids)))
    if not spans:
        return 0, len(ids)
    idx = min(max(0, phrase_idx), len(spans) - 1)
    return spans[idx]


class LLMDetCounting(nn.Module):
    ARCHITECTURE = 'llmdet-countgd-box-7layer-v3'

    def __init__(self, model_id='iSEE-Laboratory/llmdet_tiny', device='cuda',
                 freeze_backbones=True, use_exemplars=False, detector=None, processor=None):
        super().__init__()
        self.model_id = str(model_id)
        self.device = device
        self.use_exemplars = bool(use_exemplars)
        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(self.model_id)
        self.detector = (detector if detector is not None else AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)).to(device).float()
        self.freeze_backbones = freeze_backbones
        if freeze_backbones:
            self.detector.model.backbone.requires_grad_(False)
            self.detector.model.text_backbone.requires_grad_(False)
        self.d_model = int(self.detector.config.d_model)
        if self.use_exemplars:
            gn_groups = 32 if self.d_model % 32 == 0 else max(1, self.d_model // 8)
            self.exemplar_proj = nn.Sequential(
                nn.Conv2d(3 * self.d_model, self.d_model, kernel_size=1),
                nn.GroupNorm(gn_groups, self.d_model),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.d_model, self.d_model, kernel_size=1),
            ).to(device).float()
            self.exemplar_norm = nn.LayerNorm(self.d_model).to(device).float()
            self.exemplar_type_embed = nn.Parameter(torch.zeros(1, 1, self.d_model, device=device, dtype=torch.float32))
            nn.init.normal_(self.exemplar_type_embed, std=0.02)
        else:
            self.exemplar_proj = None
            self.exemplar_norm = None
            self.exemplar_type_embed = None
        self.set_training(False)

    def set_training(self, training: bool):
        self.detector.train(training)
        if self.exemplar_proj is not None:
            self.exemplar_proj.train(training)
            self.exemplar_norm.train(training)
        if self.freeze_backbones:
            self.detector.model.backbone.eval()
            self.detector.model.text_backbone.eval()

    def trainable_parameters(self):
        params = [(n, p) for n, p in self.detector.named_parameters() if p.requires_grad]
        if self.exemplar_proj is not None:
            params.extend([(f'__exemplar_proj__.{n}', p) for n, p in self.exemplar_proj.named_parameters() if p.requires_grad])
            params.extend([(f'__exemplar_norm__.{n}', p) for n, p in self.exemplar_norm.named_parameters() if p.requires_grad])
            params.append(('__exemplar_type_embed__', self.exemplar_type_embed))
        return params

    def parameter_groups(self, base_lr=2e-5, exemplar_lr=1e-4, weight_decay=1e-4):
        """Differential LR: gentle fine-tuning for pretrained LLMDet, higher LR for new exemplar head."""
        det_params = [p for _, p in self.detector.named_parameters() if p.requires_grad]
        groups = [dict(params=det_params, lr=base_lr, weight_decay=weight_decay)]
        if self.exemplar_proj is not None:
            ex_params = (
                list(self.exemplar_proj.parameters())
                + list(self.exemplar_norm.parameters())
                + [self.exemplar_type_embed]
            )
            groups.append(dict(params=ex_params, lr=exemplar_lr, weight_decay=weight_decay))
        return groups

    def _encode_text_backbone(self, m, input_ids, raw_self_attn, token_type_ids, raw_pos_ids):
        """Call BERT text_backbone compatibly across transformers 4.x (3D mask) and 5.x SDPA (4D mask)."""
        try:
            return m.text_backbone(
                input_ids=input_ids,
                attention_mask=raw_self_attn.unsqueeze(1),
                token_type_ids=token_type_ids,
                position_ids=raw_pos_ids,
                return_dict=True,
            )
        except (RuntimeError, ValueError):
            return m.text_backbone(
                input_ids=input_ids,
                attention_mask=raw_self_attn,
                token_type_ids=token_type_ids,
                position_ids=raw_pos_ids,
                return_dict=True,
            )

    def _extract_vision_features(self, pixel_values, pixel_mask):
        m = self.detector.model
        bsz, _, height, width = pixel_values.shape
        if pixel_mask is None:
            pixel_mask = torch.ones((bsz, height, width), dtype=torch.long, device=pixel_values.device)
        vision_features, position_embeddings_list = m.backbone(pixel_values, pixel_mask)
        feature_maps, masks = [], []
        for level, (source, mask) in enumerate(vision_features):
            feature_maps.append(m.input_proj_vision[level](source))
            masks.append(mask)
        if m.config.num_feature_levels > len(feature_maps):
            _len_sources = len(feature_maps)
            for level in range(_len_sources, m.config.num_feature_levels):
                source = m.input_proj_vision[level](vision_features[-1][0] if level == _len_sources else feature_maps[-1])
                mask = F.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                pos_l = m.backbone.position_embedding(source, mask).to(source.dtype)
                feature_maps.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)
        return feature_maps, masks, position_embeddings_list, pixel_mask

    def _extract_exemplar_tokens(self, feature_maps, masks, exemplar_boxes):
        """Extract (1, E, d_model) normalized exemplar tokens via FP32 ROIAlign(1x1) using Swin valid_ratios."""
        m = self.detector.model
        orig_dtype = feature_maps[0].dtype
        with torch.autocast(device_type='cuda', enabled=False):
            f0 = feature_maps[0].float()
            h0, w0 = f0.shape[-2:]
            lvl1 = (feature_maps[1] if len(feature_maps) > 1 else feature_maps[0]).float()
            lvl2 = (feature_maps[2] if len(feature_maps) > 2 else lvl1).float()
            fused = torch.cat([
                f0,
                F.interpolate(lvl1, size=(h0, w0), mode='bilinear', align_corners=False),
                F.interpolate(lvl2, size=(h0, w0), mode='bilinear', align_corners=False),
            ], dim=1)
            fused = self.exemplar_proj(fused)
            valid_ratio = m.get_valid_ratio(masks[0])[0]  # [valid_w_ratio, valid_h_ratio]
            vx = float(valid_ratio[0].item())
            vy = float(valid_ratio[1].item())
            sx = float(w0) * (vx if vx > 0.0 else 1.0)
            sy = float(h0) * (vy if vy > 0.0 else 1.0)
            boxes = exemplar_boxes.to(device=fused.device, dtype=torch.float32)
            scaled = torch.stack([
                (boxes[:, 0] * sx).clamp(0.0, float(w0)),
                (boxes[:, 1] * sy).clamp(0.0, float(h0)),
                (boxes[:, 2] * sx).clamp(0.0, float(w0)),
                (boxes[:, 3] * sy).clamp(0.0, float(h0)),
            ], dim=-1)
            pooled = roi_align(fused, [scaled], output_size=(1, 1), aligned=True).flatten(1).unsqueeze(0)
            tokens = self.exemplar_norm(pooled) + self.exemplar_type_embed
        return tokens.to(orig_dtype)

    @torch.no_grad()
    def encode_exemplars(self, pil_image, exemplar_boxes):
        """Precompute (1, E, d_model) exemplar embeddings from a full image (used by 2x2 tiling)."""
        if not self.use_exemplars or exemplar_boxes is None or len(exemplar_boxes) == 0:
            return None
        inputs = self.processor(images=pil_image, text='object .', do_resize=False, return_tensors='pt').to(self.device)
        feature_maps, masks, _, _ = self._extract_vision_features(inputs['pixel_values'], inputs.get('pixel_mask'))
        return self._extract_exemplar_tokens(feature_maps, masks, exemplar_boxes)

    def forward_stages(self, pixel_values, pixel_mask, input_ids, attention_mask, token_type_ids=None,
                       target_phrase_idx=0, exemplar_boxes=None, exemplar_feats=None):
        """Run MM-Grounding-DINO forward with differentiable 7-stage boxes/logits & optional exemplar injection."""
        m = self.detector.model
        max_text_len = int(m.config.max_text_len)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        num_ex = 0
        if self.use_exemplars:
            if exemplar_feats is not None:
                num_ex = int(exemplar_feats.shape[1])
            elif exemplar_boxes is not None and len(exemplar_boxes) > 0:
                num_ex = int(len(exemplar_boxes))

        # Reserve token budget for injected exemplars inside max_text_len
        text_budget = max_text_len - num_ex
        if input_ids.shape[1] > text_budget:
            input_ids = input_ids[:, :text_budget]
            attention_mask = attention_mask[:, :text_budget]
            token_type_ids = token_type_ids[:, :text_budget]

        raw_self_attn, raw_pos_ids = generate_text_masks(input_ids)
        text_outputs = self._encode_text_backbone(m, input_ids, raw_self_attn, token_type_ids, raw_pos_ids)
        text_features = m.text_projection(text_outputs.last_hidden_state)

        feature_maps, masks, position_embeddings_list, pixel_mask = self._extract_vision_features(pixel_values, pixel_mask)

        pos_start, pos_end = phrase_token_span(input_ids, target_phrase_idx)
        if self.use_exemplars and num_ex > 0:
            ex_tok = exemplar_feats.to(text_features.device, text_features.dtype) if exemplar_feats is not None else \
                self._extract_exemplar_tokens(feature_maps, masks, exemplar_boxes).to(text_features.dtype)
            ex_ids = torch.full((1, num_ex), EXEMPLAR_TOKEN_ID, dtype=input_ids.dtype, device=input_ids.device)
            ex_mask = torch.ones((1, num_ex), dtype=attention_mask.dtype, device=attention_mask.device)
            input_ids = torch.cat([input_ids[:, :pos_end], ex_ids, input_ids[:, pos_end:]], dim=1)
            attention_mask = torch.cat([attention_mask[:, :pos_end], ex_mask, attention_mask[:, pos_end:]], dim=1)
            text_features = torch.cat([text_features[:, :pos_end], ex_tok, text_features[:, pos_end:]], dim=1)
            pos_end = pos_end + num_ex

        text_self_attention_masks, position_ids = generate_text_masks(input_ids)
        text_token_mask = attention_mask.bool()
        seq_len = input_ids.shape[1]

        query_embeds = m.query_position_embeddings.weight if (m.config.embedding_init_target or m.config.two_stage) else None
        source_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes_list = [], [], [], []
        for level, (source, mask, pos_embed) in enumerate(zip(feature_maps, masks, position_embeddings_list)):
            _, _, h_l, w_l = source.shape
            spatial_shapes_list.append((h_l, w_l))
            source_flatten.append(source.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            lvl_pos_embed_flatten.append(pos_embed.flatten(2).transpose(1, 2) + m.level_embed[level].view(1, 1, -1))
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([m.get_valid_ratio(mk) for mk in masks], 1).float()

        encoder_outputs = m.encoder(
            vision_features=source_flatten,
            vision_attention_mask=~mask_flatten,
            vision_position_embedding=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            text_features=text_features,
            text_attention_mask=~text_token_mask,
            text_position_embedding=None,
            text_self_attention_masks=~text_self_attention_masks,
            text_position_ids=position_ids,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        encoder_pred_boxes = None
        encoder_logits = None
        if m.config.two_stage:
            object_query_embedding, output_proposals = m.generate_encoder_output_proposals(
                encoder_outputs[0], ~mask_flatten, spatial_shapes
            )
            enc_outputs_class = m.encoder_output_class_embed(
                object_query_embedding, encoder_outputs[1], text_token_mask
            )
            delta_bbox = m.encoder_output_bbox_embed(object_query_embedding)
            enc_outputs_coord_logits = delta_bbox + output_proposals

            topk = m.config.num_queries
            topk_logits = enc_outputs_class.max(-1)[0]
            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]
            topk_coords_raw = torch.gather(
                enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )
            topk_class_raw = torch.gather(
                enc_outputs_class, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
            )
            # Differentiable encoder stage outputs (trains enc_output, encoder_output_bbox_embed & encoder_output_class_embed)
            encoder_pred_boxes = topk_coords_raw.sigmoid()[0]
            encoder_logits = topk_class_raw[0]

            # Detached initial reference points fed to decoder (standard two-stage Deformable DETR)
            reference_points = topk_coords_raw.detach().sigmoid()
            init_reference_points = reference_points
            if query_embeds is not None:
                target = query_embeds.unsqueeze(0).repeat(pixel_values.shape[0], 1, 1)
            else:
                target = torch.gather(
                    object_query_embedding, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, m.d_model)
                ).detach()
        else:
            target = query_embeds.unsqueeze(0).repeat(pixel_values.shape[0], 1, 1)
            reference_points = m.reference_points.weight.unsqueeze(0).repeat(pixel_values.shape[0], 1, 1).sigmoid()
            init_reference_points = reference_points

        decoder_outputs = m.decoder(
            inputs_embeds=target,
            vision_encoder_hidden_states=encoder_outputs[0],
            vision_encoder_attention_mask=mask_flatten,
            text_encoder_hidden_states=encoder_outputs[1],
            text_encoder_attention_mask=~text_token_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            self_attn_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        enc_text = encoder_outputs.last_hidden_state_text
        hidden_states = decoder_outputs.intermediate_hidden_states
        inter_refs = decoder_outputs.intermediate_reference_points
        active_mask = token_span_mask(input_ids)
        target_mask_full = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
        target_mask_full[pos_start:pos_end] = True
        target_tokens = target_mask_full[active_mask]
        if not target_tokens.any():
            target_tokens = torch.ones(int(active_mask.sum()), dtype=torch.bool, device=input_ids.device)

        # Recompute differentiable stage boxes via bbox_embed[level] (matches MMGroundingDinoForObjectDetection);
        # inter_refs from decoder are detached, so using inter_refs directly kills all box/center gradients!
        stage_boxes, stage_logits = [], []
        for level in range(hidden_states.shape[1]):
            ref = init_reference_points if level == 0 else inter_refs[:, level - 1]
            ref_logit = torch.special.logit(ref.float(), eps=1e-5)
            delta_b = self.detector.bbox_embed[level](hidden_states[:, level]).float()
            if ref_logit.shape[-1] == 4:
                pred_coord = (delta_b + ref_logit).sigmoid()
            else:
                delta_b = delta_b.clone()
                delta_b[..., :2] = delta_b[..., :2] + ref_logit
                pred_coord = delta_b.sigmoid()
            cls_logits = self.detector.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text,
                text_token_mask=text_token_mask,
            )[0, :, :seq_len][:, active_mask]
            stage_boxes.append(pred_coord[0].float())
            stage_logits.append(cls_logits.float())

        if encoder_logits is not None:
            encoder_logits = encoder_logits[:, :seq_len][:, active_mask].float()
        if encoder_pred_boxes is not None:
            encoder_pred_boxes = encoder_pred_boxes.float()

        return dict(
            boxes=stage_boxes[-1],
            logits=stage_logits[-1],
            aux_boxes=stage_boxes[:-1],
            aux_logits=stage_logits[:-1],
            enc_boxes=encoder_pred_boxes,
            enc_logits=encoder_logits,
            target_tokens=target_tokens,
        )

    def forward(self, pil_image, prompt: str, exemplar_boxes=None, exemplar_feats=None):
        """Single-image inference returning normalized cxcywh boxes (900, 4) and max target logits (900,)."""
        caption = prompt.strip().lower().rstrip('.') + ' .'
        inputs = self.processor(images=pil_image, text=caption, do_resize=False, return_tensors='pt').to(self.device)
        out = self.forward_stages(
            inputs['pixel_values'],
            inputs.get('pixel_mask'),
            inputs['input_ids'],
            inputs['attention_mask'],
            inputs.get('token_type_ids'),
            target_phrase_idx=0,
            exemplar_boxes=exemplar_boxes,
            exemplar_feats=exemplar_feats,
        )
        scores_logit = out['logits'][:, out['target_tokens']].max(dim=-1).values
        return out['boxes'], scores_logit

    def export_trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.trainable_parameters()}

    def import_trainable_state(self, state):
        det_state = {k: v for k, v in state.items() if not k.startswith('__exemplar_')}
        ex_proj = {k[len('__exemplar_proj__.'):]: v for k, v in state.items() if k.startswith('__exemplar_proj__.')}
        ex_norm = {k[len('__exemplar_norm__.'):]: v for k, v in state.items() if k.startswith('__exemplar_norm__.')}
        _, unexpected = self.detector.load_state_dict(det_state, strict=False)
        if unexpected:
            raise RuntimeError(f'Unexpected keys in LLMDet checkpoint: {unexpected[:5]}')
        if self.exemplar_proj is not None and ex_proj:
            self.exemplar_proj.load_state_dict(ex_proj, strict=True)
        if self.exemplar_norm is not None and ex_norm:
            self.exemplar_norm.load_state_dict(ex_norm, strict=True)
        if self.exemplar_type_embed is not None and '__exemplar_type_embed__' in state:
            self.exemplar_type_embed.data.copy_(state['__exemplar_type_embed__'].to(self.exemplar_type_embed.device))
