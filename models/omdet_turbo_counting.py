"""OmDet-Turbo (RT-DETR HybridEncoder + CLIP + Task-Conditioned Deformable Decoder)
adaptation for real-time NMS-free Open-World Counting (CountVid Stage 1).

Key architectural differences from Grounding DINO / CountGD / LLMDet:
- Vision Encoder: Lightweight RT-DETR HybridEncoder (1-layer AIFI on P5 + CSP-FPN/PAN),
  eliminating Grounding DINO's heavy 6-layer bi-directional cross-attention encoder (~5-10x faster).
- Text Backbone: Frozen CLIP with detached LRU embedding cache (encode prompts once per video).
- Decoupled Decoder Fusion: 900 queries + 6-layer Deformable DETR Decoder with Joint Self-Attention
  over `[object queries, task tokens, visual exemplar tokens]` and Cosine Similarity class scoring.
- Dual-Path Visual Exemplar Injection (when `use_exemplars=True`):
  1. Path A (Task Prompt): FP32 ROIAlign(1x1) on fused HybridEncoder (P3, P4, P5) -> `exemplar_proj`
     -> `exemplar_to_embed` -> concatenated into `task_features` so all 900 queries attend to
     individual visual exemplars across all 6 decoder layers.
  2. Path B (Class Prototype): Gated residual visual prototype added to `class_features[target_idx]`
     (`tanh(exemplar_cls_gate) * proto`) so BOTH Two-Stage Encoder Query Selection AND all 6 Decoder
     cosine similarity heads are guided by the visual exemplars.
- 7-Layer Differentiable CountGD-Box Supervision (1 Encoder Two-Stage + 6 Decoder Look-Forward-Twice),
  with BatchNorm2d running stats frozen in `.eval()` mode for `batch_size=1` stability.
"""
import random
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from transformers import AutoProcessor, OmDetTurboForObjectDetection


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
    ARCHITECTURE = 'omdet-turbo-countgd-box-7layer-v1'

    def __init__(self, model_id='omlab/omdet-turbo-swin-tiny-hf', device='cuda',
                 freeze_backbones=True, use_exemplars=False, detector=None, processor=None):
        super().__init__()
        self.model_id = str(model_id)
        self.device = device
        self.use_exemplars = bool(use_exemplars)
        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(self.model_id)
        self.detector = (
            detector if detector is not None else OmDetTurboForObjectDetection.from_pretrained(self.model_id)
        ).to(device).float()
        self.freeze_backbones = freeze_backbones
        if freeze_backbones:
            self.detector.vision_backbone.requires_grad_(False)
            self.detector.language_backbone.requires_grad_(False)

        cfg = self.detector.config
        self.d_model = int(cfg.decoder_hidden_dim)
        self.class_embed_dim = int(cfg.class_embed_dim)
        self.encoder_dim = int(cfg.encoder_hidden_dim)
        self.num_queries = int(cfg.num_queries)

        if self.use_exemplars:
            gn_groups = 32 if self.d_model % 32 == 0 else max(1, self.d_model // 8)
            self.exemplar_proj = nn.Sequential(
                nn.Conv2d(3 * self.encoder_dim, self.d_model, kernel_size=1),
                nn.GroupNorm(gn_groups, self.d_model),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.d_model, self.d_model, kernel_size=1),
            ).to(device).float()
            self.exemplar_to_embed = nn.Sequential(
                nn.Linear(self.d_model, self.class_embed_dim),
                nn.LayerNorm(self.class_embed_dim),
            ).to(device).float()
            self.exemplar_task_embed = nn.Parameter(
                torch.zeros(1, 1, self.class_embed_dim, device=device, dtype=torch.float32)
            )
            nn.init.normal_(self.exemplar_task_embed, std=0.02)
            self.exemplar_proto_head = nn.Sequential(
                nn.Linear(self.class_embed_dim, self.class_embed_dim),
                nn.LayerNorm(self.class_embed_dim),
            ).to(device).float()
            # Small non-zero init (tanh(0.1) ~= 0.1) preserves pretrained CLIP alignment at step 0
            # while guaranteeing immediate non-zero backward gradients to exemplar_proto_head.
            self.exemplar_cls_gate = nn.Parameter(torch.tensor([0.1], device=device, dtype=torch.float32))
        else:
            self.exemplar_proj = None
            self.exemplar_to_embed = None
            self.exemplar_task_embed = None
            self.exemplar_proto_head = None
            self.exemplar_cls_gate = None

        self.set_training(False)

    def set_training(self, training: bool):
        self.detector.train(training)
        if self.exemplar_proj is not None:
            self.exemplar_proj.train(training)
            self.exemplar_to_embed.train(training)
            self.exemplar_proto_head.train(training)
        if self.freeze_backbones:
            self.detector.vision_backbone.eval()
            self.detector.language_backbone.eval()
        # Freeze BatchNorm2d running stats during batch_size=1 training to prevent single-image stat corruption
        for module in self.detector.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def trainable_parameters(self):
        params = [(n, p) for n, p in self.detector.named_parameters() if p.requires_grad]
        if self.exemplar_proj is not None:
            params.extend([(f'__exemplar_proj__.{n}', p) for n, p in self.exemplar_proj.named_parameters() if p.requires_grad])
            params.extend([(f'__exemplar_to_embed__.{n}', p) for n, p in self.exemplar_to_embed.named_parameters() if p.requires_grad])
            params.extend([(f'__exemplar_proto_head__.{n}', p) for n, p in self.exemplar_proto_head.named_parameters() if p.requires_grad])
            params.append(('__exemplar_task_embed__', self.exemplar_task_embed))
            params.append(('__exemplar_cls_gate__', self.exemplar_cls_gate))
        return params

    def parameter_groups(self, base_lr=3e-5, exemplar_lr=1.5e-4, weight_decay=1e-4):
        det_params = [p for _, p in self.detector.named_parameters() if p.requires_grad]
        groups = [dict(params=det_params, lr=base_lr, weight_decay=weight_decay)]
        if self.exemplar_proj is not None:
            ex_params = (
                list(self.exemplar_proj.parameters())
                + list(self.exemplar_to_embed.parameters())
                + list(self.exemplar_proto_head.parameters())
                + [self.exemplar_task_embed, self.exemplar_cls_gate]
            )
            groups.append(dict(params=ex_params, lr=exemplar_lr, weight_decay=weight_decay))
        return groups

    def _extract_exemplar_embeddings(self, encoder_states, exemplar_boxes):
        """Extract (E, class_embed_dim) visual exemplar embeddings in pure FP32 via ROIAlign(1x1)."""
        with torch.autocast(device_type='cuda', enabled=False):
            p3 = encoder_states[0].float()
            h0, w0 = p3.shape[-2:]
            p4 = (encoder_states[1] if len(encoder_states) > 1 else p3).float()
            p5 = (encoder_states[2] if len(encoder_states) > 2 else p4).float()
            fused = torch.cat([
                p3,
                F.interpolate(p4, size=(h0, w0), mode='bilinear', align_corners=False),
                F.interpolate(p5, size=(h0, w0), mode='bilinear', align_corners=False),
            ], dim=1)
            fused = self.exemplar_proj(fused)
            boxes = exemplar_boxes.to(device=fused.device, dtype=torch.float32)
            scaled = torch.stack([
                (boxes[:, 0] * float(w0)).clamp(0.0, float(w0)),
                (boxes[:, 1] * float(h0)).clamp(0.0, float(h0)),
                (boxes[:, 2] * float(w0)).clamp(0.0, float(w0)),
                (boxes[:, 3] * float(h0)).clamp(0.0, float(h0)),
            ], dim=-1)
            pooled = roi_align(fused, [scaled], output_size=(1, 1), aligned=True).flatten(1)  # (E, d_model)
            ex_emb = self.exemplar_to_embed(pooled)  # (E, class_embed_dim)
        return ex_emb

    @torch.no_grad()
    def encode_exemplars(self, pil_image, exemplar_boxes):
        """Precompute (E, class_embed_dim) visual exemplar embeddings from full image for 2x2 tiling."""
        if not self.use_exemplars or exemplar_boxes is None or len(exemplar_boxes) == 0:
            return None
        inputs = self.processor(images=pil_image, text=['object'], task='Detect object.', return_tensors='pt').to(self.device)
        image_features = self.detector.vision_backbone(inputs['pixel_values'])
        enc_out = self.detector.encoder(image_features, return_dict=True)
        return self._extract_exemplar_embeddings(enc_out.extracted_states, exemplar_boxes)

    def forward_stages(self, pixel_values, classes_input_ids, classes_attention_mask,
                       tasks_input_ids, tasks_attention_mask, classes_structure,
                       target_idx: int = 0, exemplar_boxes=None, exemplar_feats=None):
        """Run OmDet-Turbo forward returning differentiable 7-stage outputs (1 Encoder + 6 Decoder)."""
        with torch.no_grad():
            image_features = self.detector.vision_backbone(pixel_values)
            # Compute frozen CLIP language features under no_grad and clone so LRU cache is never mutated
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
        extracted_states = encoder_outputs.extracted_states

        num_classes = int(class_features.shape[0])
        t_idx = min(max(0, int(target_idx)), num_classes - 1)

        if self.use_exemplars:
            ex_emb = None
            if exemplar_feats is not None:
                ex_emb = exemplar_feats.to(device=class_features.device, dtype=torch.float32)
            elif exemplar_boxes is not None and len(exemplar_boxes) > 0:
                ex_emb = self._extract_exemplar_embeddings(extracted_states, exemplar_boxes)
            if ex_emb is not None and len(ex_emb) > 0:
                # Path A: Append individual visual exemplar tokens to task_features for 6-layer Decoder Self-Attention
                ex_task = (ex_emb.unsqueeze(1) + self.exemplar_task_embed).to(task_features.dtype)  # (E, 1, C)
                ex_mask = torch.ones((task_mask.shape[0], ex_task.shape[0]), dtype=task_mask.dtype, device=task_mask.device)
                task_features = torch.cat([task_features, ex_task], dim=0)
                task_mask = torch.cat([task_mask, ex_mask], dim=1)

                # Path B: Gated visual-language prototype fusion into target class embedding
                with torch.autocast(device_type='cuda', enabled=False):
                    proto = self.exemplar_proto_head(ex_emb.mean(dim=0, keepdim=True))  # (1, C)
                    gate = torch.tanh(self.exemplar_cls_gate).view(1, 1)
                    updated_target = class_features[t_idx].float() + gate * proto
                class_features = class_features.clone()
                class_features[t_idx] = updated_target.to(class_features.dtype)

        decoder_outputs = self.detector.decoder(
            extracted_states,
            class_features,
            task_features,
            task_mask,
            return_dict=True,
        )

        # In train mode: decoder_coords has shape (L, 1, Q, 4), decoder_classes has shape (L, 1, Q, K)
        # In eval mode:  decoder_coords has shape (1, 1, Q, 4), decoder_classes has shape (1, 1, Q, K)
        dec_coords = [decoder_outputs.decoder_coords[i, 0].float() for i in range(decoder_outputs.decoder_coords.shape[0])]
        dec_classes = [decoder_outputs.decoder_classes[i, 0].float() for i in range(decoder_outputs.decoder_classes.shape[0])]
        enc_boxes = decoder_outputs.encoder_coord_logits[0].float()
        enc_logits = decoder_outputs.encoder_class_logits[0].float()

        target_tokens = torch.zeros(num_classes, dtype=torch.bool, device=pixel_values.device)
        target_tokens[t_idx] = True

        return dict(
            boxes=dec_coords[-1],
            logits=dec_classes[-1],
            aux_boxes=dec_coords[:-1],
            aux_logits=dec_classes[:-1],
            enc_boxes=enc_boxes,
            enc_logits=enc_logits,
            target_tokens=target_tokens,
        )

    def forward(self, pil_image, prompt: str, exemplar_boxes=None, exemplar_feats=None):
        """Single-image inference returning normalized cxcywh boxes (900, 4) and target class logits (900,)."""
        cls_name = prompt.strip().lower().rstrip('.')
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
        )
        scores_logit = out['logits'][:, out['target_tokens']].max(dim=-1).values
        return out['boxes'], scores_logit

    def export_trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.trainable_parameters()}

    def import_trainable_state(self, state):
        det_state = {k: v for k, v in state.items() if not k.startswith('__exemplar_')}
        ex_proj = {k[len('__exemplar_proj__.'):]: v for k, v in state.items() if k.startswith('__exemplar_proj__.')}
        ex_emb = {k[len('__exemplar_to_embed__.'):]: v for k, v in state.items() if k.startswith('__exemplar_to_embed__.')}
        ex_proto = {k[len('__exemplar_proto_head__.'):]: v for k, v in state.items() if k.startswith('__exemplar_proto_head__.')}
        _, unexpected = self.detector.load_state_dict(det_state, strict=False)
        if unexpected:
            raise RuntimeError(f'Unexpected keys in OmDet-Turbo checkpoint: {unexpected[:5]}')
        if self.exemplar_proj is not None and ex_proj:
            self.exemplar_proj.load_state_dict(ex_proj, strict=True)
        if self.exemplar_to_embed is not None and ex_emb:
            self.exemplar_to_embed.load_state_dict(ex_emb, strict=True)
        if self.exemplar_proto_head is not None and ex_proto:
            self.exemplar_proto_head.load_state_dict(ex_proto, strict=True)
        if self.exemplar_task_embed is not None and '__exemplar_task_embed__' in state:
            self.exemplar_task_embed.data.copy_(state['__exemplar_task_embed__'].to(self.exemplar_task_embed.device))
        if self.exemplar_cls_gate is not None and '__exemplar_cls_gate__' in state:
            self.exemplar_cls_gate.data.copy_(state['__exemplar_cls_gate__'].to(self.exemplar_cls_gate.device))
