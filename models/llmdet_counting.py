"""LLMDet (MM-Grounding-DINO) adaptation with CountGD-Box 7-layer deep supervision.

Uses HuggingFace's MMGroundingDinoForObjectDetection / GroundingDinoForObjectDetection
(`iSEE-Laboratory/llmdet_tiny`, `iSEE-Laboratory/llmdet_base`, `iSEE-Laboratory/llmdet_large`)
with frozen Swin visual backbone and frozen BERT language backbone, while training:
- 6-layer Bi-directional Feature Enhancer Encoder (image <-> text cross-modality fusion)
- Language-Guided Query Selection (`enc_output`, `encoder_output_class_embed`, `encoder_output_bbox_embed`)
- 6-layer Deformable DETR Decoder (900 queries with Self-Attention + Deformable Cross-Attention)
- All 6 shared/per-layer Contrastive Classification & Iterative Box Refinement heads

Supervision applies the exact CountGD-Box Hungarian criterion across all 7 levels:
- 1 Encoder proposal output (`encoder_logits`, `encoder_pred_boxes`)
- 5 Intermediate Decoder layers (`outputs_class[0..4]`, `outputs_coord[0..4]`)
- 1 Final Decoder layer (`outputs_class[5]`, `outputs_coord[5]`)
"""
import torch
from torch import nn
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

SPECIAL_TOKEN_IDS = (101, 102, 1012, 1029, 0)  # [CLS], [SEP], '.', '?', [PAD]


def token_span_mask(input_ids: torch.LongTensor) -> torch.BoolTensor:
    """Return boolean mask (seq_len,) selecting non-special word tokens in a single-class caption."""
    ids = input_ids[0]
    special = torch.tensor(SPECIAL_TOKEN_IDS, device=ids.device)
    mask = ~torch.isin(ids, special)
    if not mask.any():
        mask = ids != 0
    return mask


class LLMDetCounting(nn.Module):
    ARCHITECTURE = 'llmdet-countgd-box-7layer-v1'

    def __init__(self, model_id='iSEE-Laboratory/llmdet_tiny', device='cuda', freeze_backbones=True):
        super().__init__()
        self.model_id = str(model_id)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(device).float()
        self.freeze_backbones = freeze_backbones
        if freeze_backbones:
            self.detector.model.backbone.requires_grad_(False)
            self.detector.model.text_backbone.requires_grad_(False)
        self.set_training(False)

    def set_training(self, training: bool):
        self.detector.train(training)
        if self.freeze_backbones:
            self.detector.model.backbone.eval()
            self.detector.model.text_backbone.eval()

    def _compute_all_stages(self, outputs, attention_mask):
        """Reconstruct all 6 decoder stage (boxes, logits) and the encoder proposal stage."""
        enc_text = outputs.encoder_last_hidden_state_text  # (1, L_txt, D)
        hidden_states = outputs.intermediate_hidden_states  # (1, 6, 900, D)
        init_ref = outputs.init_reference_points  # (1, 900, 4)
        inter_refs = outputs.intermediate_reference_points  # (1, 6, 900, 4)
        text_mask = attention_mask.bool()

        stage_boxes = []
        stage_logits = []
        num_levels = hidden_states.shape[1]
        for level in range(num_levels):
            cls_logits = self.detector.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text,
                text_token_mask=text_mask,
            )  # (1, 900, max_text_len)
            stage_boxes.append(inter_refs[:, level][0])
            stage_logits.append(cls_logits[0])
        enc_boxes = outputs.encoder_pred_boxes[0] if outputs.encoder_pred_boxes is not None else init_ref[0]
        enc_logits = outputs.encoder_logits[0] if outputs.encoder_logits is not None else None
        return stage_boxes, stage_logits, enc_boxes, enc_logits

    def forward_stages(self, pixel_values, pixel_mask, input_ids, attention_mask, token_type_ids=None):
        """Return final + intermediate decoder & encoder stages along with the active text token mask."""
        outputs = self.detector(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_len = input_ids.shape[1]
        word_mask = token_span_mask(input_ids)  # (seq_len,)
        stage_boxes, stage_logits, enc_boxes, enc_logits = self._compute_all_stages(outputs, attention_mask)
        # Slice logits from max_text_len (256) down to active seq_len so target_tokens mask matches
        stage_logits = [lg[:, :seq_len] for lg in stage_logits]
        if enc_logits is not None:
            enc_logits = enc_logits[:, :seq_len]
        return dict(
            boxes=stage_boxes[-1],
            logits=stage_logits[-1],
            aux_boxes=stage_boxes[:-1],
            aux_logits=stage_logits[:-1],
            enc_boxes=enc_boxes,
            enc_logits=enc_logits,
            target_tokens=word_mask,
        )

    def forward(self, pil_image, prompt: str):
        """Single-image inference returning normalized cxcywh boxes (900, 4) and max word-token logits (900,)."""
        caption = prompt.strip().lower().rstrip('.') + '.'
        inputs = self.processor(images=pil_image, text=caption, return_tensors='pt').to(self.device)
        out = self.forward_stages(
            inputs['pixel_values'],
            inputs.get('pixel_mask'),
            inputs['input_ids'],
            inputs['attention_mask'],
            inputs.get('token_type_ids'),
        )
        word_logits = out['logits'][:, out['target_tokens']]
        scores_logit = word_logits.max(dim=-1).values
        return out['boxes'], scores_logit

    def export_trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.detector.named_parameters() if p.requires_grad}

    def import_trainable_state(self, state):
        missing, unexpected = self.detector.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f'Unexpected keys in LLMDet checkpoint: {unexpected[:5]}')
