import random
import unittest
from PIL import Image
import torch
from transformers import MMGroundingDinoConfig, MMGroundingDinoForObjectDetection, BertConfig, SwinConfig
from models.yoloworld_counting import CountingCriterion
from models.llmdet_counting import (
    EXEMPLAR_TOKEN_ID,
    LLMDetCounting,
    token_span_mask,
    generate_text_masks,
    build_multiclass_caption,
    phrase_token_span,
)
from tools.train_llmdet_counting import prepare_llmdet_sample, compute_7layer_loss


def _make_tiny_detector():
    """Create a tiny in-memory MMGroundingDinoForObjectDetection (no downloads, runs on CPU in <0.5s)."""
    cfg = MMGroundingDinoConfig(
        backbone_config=SwinConfig(
            image_size=64, patch_size=4, embed_dim=16, depths=[1, 1, 1],
            num_heads=[2, 2, 2], window_size=4, out_features=['stage1', 'stage2', 'stage3'],
        ),
        text_config=BertConfig(
            vocab_size=5000, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
        ),
        d_model=32,
        encoder_layers=1,
        decoder_layers=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_n_points=2,
        decoder_n_points=2,
        num_queries=12,
        num_feature_levels=3,
        max_text_len=48,
        two_stage=True,
    )
    return MMGroundingDinoForObjectDetection(cfg)


class LLMDetCountingTests(unittest.TestCase):
    def test_token_span_mask_excludes_special_tokens_and_keeps_exemplars(self):
        ids = torch.tensor([[101, 2500, EXEMPLAR_TOKEN_ID, 1012, 3600, 1012, 102, 0]])
        mask = token_span_mask(ids)
        self.assertEqual(mask.tolist(), [False, True, True, False, True, False, False, False])

    def test_generate_text_masks_matches_hf_block_diagonal_and_groups_exemplars(self):
        # [CLS]=101, target=2500, ex1=1008, ex2=1008, '.'=1012, neg1=3600, neg2=3601, '.'=1012, [SEP]=102
        ids = torch.tensor([[101, 2500, EXEMPLAR_TOKEN_ID, EXEMPLAR_TOKEN_ID, 1012, 3600, 3601, 1012, 102]])
        attn, pos = generate_text_masks(ids)
        # Target + 2 exemplars + trailing '.' (indices 1..4) attend together, isolated from neg phrase (5..7)
        self.assertTrue(attn[0, 1:5, 1:5].all())
        self.assertFalse(attn[0, 1:5, 5:8].any())
        self.assertFalse(attn[0, 5:8, 1:5].any())
        self.assertTrue(attn[0, 5:8, 5:8].all())
        self.assertEqual(pos[0].tolist(), [0, 0, 1, 2, 3, 0, 1, 2, 0])

    def test_build_multiclass_caption_and_phrase_token_span(self):
        vocab = ['apples', 'birds', 'cars', 'dogs', 'eggs']
        caption, idx = build_multiclass_caption('birds', vocab, max_classes=4, rng=random.Random(42))
        phrases = [p.strip() for p in caption.rstrip(' .').split(' . ')]
        self.assertEqual(len(phrases), 4)
        self.assertEqual(phrases[idx], 'birds')
        ids = torch.tensor([[101, 2000, 2001, 1012, 3000, 1012, 4000, 4001, 4002, 1012, 102]])
        self.assertEqual(phrase_token_span(ids, 0), (1, 3))
        self.assertEqual(phrase_token_span(ids, 1), (4, 5))
        self.assertEqual(phrase_token_span(ids, 2), (6, 9))

    def test_prepare_llmdet_sample_caps_overflow_and_keeps_exemplars(self):
        image = Image.new('RGB', (400, 300))
        pts = [[float(i % 380 + 10), float((i * 7) % 280 + 10)] for i in range(1200)]
        record = dict(
            points=pts,
            box_examples_coordinates=[[[15, 15], [35, 15], [35, 35], [15, 35]]],
        )
        sample = prepare_llmdet_sample(image, record, short=480, flip=True, max_queries=900)
        self.assertEqual(len(sample['points']), 1200)
        self.assertEqual(len(sample['supervision_points']), 900)
        self.assertEqual(len(sample['exemplar_indices']), 1)
        self.assertTrue((sample['supervision_points'] >= 0).all() and (sample['supervision_points'] <= 1).all())

    def test_balanced_negative_token_loss_stays_bounded(self):
        criterion = CountingCriterion()
        boxes = torch.tensor([[0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]])
        pts = torch.tensor([[0.3, 0.3], [0.7, 0.7]])
        ex_i = torch.tensor([0])
        ex_b = torch.tensor([[0.25, 0.25, 0.35, 0.35]])
        # Compare 1-class (2 tokens) vs 8-class (2 pos + 14 neg tokens) at identical zero logits
        l_1cls, p1 = criterion(boxes, torch.zeros(3, 2), pts, ex_i, ex_b, target_tokens=torch.tensor([True, True]))
        l_8cls, p8 = criterion(boxes, torch.zeros(3, 16), pts, ex_i, ex_b,
                               target_tokens=torch.tensor([True, True] + [False] * 14))
        # Balanced negative scaling guarantees 8-class cls loss is ~2x (not 8x!) of 1-class
        self.assertLess(float(p8['cls']), float(p1['cls']) * 3.0)

    def test_end_to_end_forward_and_backward_gradients_all_modules(self):
        """Verify real gradients reach bbox_embed, encoder proposal heads, class_embed, and exemplar_proj."""
        tiny = _make_tiny_detector()
        model = LLMDetCounting('tiny-test', device='cpu', freeze_backbones=True, use_exemplars=True,
                               detector=tiny, processor=object())
        model.set_training(True)
        pixel_values = torch.randn(1, 3, 64, 64)
        pixel_mask = torch.ones(1, 64, 64, dtype=torch.long)
        # 2 classes: [CLS] w1 w2 '.' w3 '.' [SEP]
        input_ids = torch.tensor([[101, 200, 201, 1012, 300, 1012, 102]])
        attention_mask = torch.ones_like(input_ids)
        ex_boxes = torch.tensor([[0.2, 0.2, 0.4, 0.4], [0.6, 0.6, 0.8, 0.8]])
        st = model.forward_stages(
            pixel_values, pixel_mask, input_ids, attention_mask,
            target_phrase_idx=0, exemplar_boxes=ex_boxes,
        )
        # 3 active text tokens + 2 injected exemplars = 5 active tokens (4 target, 1 negative)
        self.assertEqual(st['logits'].shape, (12, 5))
        self.assertEqual(st['target_tokens'].tolist(), [True, True, True, True, False])
        sample = dict(
            supervision_points=torch.tensor([[0.3, 0.3], [0.7, 0.7]]),
            exemplar_indices=torch.tensor([0, 1]),
            exemplar_boxes=ex_boxes,
        )
        loss, _ = compute_7layer_loss(CountingCriterion(), st, sample, 'cpu')
        loss.backward()

        # CRITICAL: Verify previously detached modules now receive real non-zero gradients!
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.bbox_embed.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.model.encoder_output_bbox_embed.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.model.enc_output.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.class_embed.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.exemplar_proj.parameters()))
        self.assertTrue(model.exemplar_type_embed.grad is not None and model.exemplar_type_embed.grad.abs().sum() > 0)
        # Backbones must stay frozen (no grads)
        self.assertTrue(all(p.grad is None for p in model.detector.model.backbone.parameters()))
        self.assertTrue(all(p.grad is None for p in model.detector.model.text_backbone.parameters()))


if __name__ == '__main__':
    unittest.main()
