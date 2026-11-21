import random
import unittest
from PIL import Image
import torch
from models.yoloworld_counting import CountingCriterion
from models.llmdet_counting import (
    EXEMPLAR_TOKEN_ID,
    token_span_mask,
    generate_text_masks,
    build_multiclass_caption,
    phrase_token_span,
)
from tools.train_llmdet_counting import prepare_llmdet_sample, compute_7layer_loss


class LLMDetCountingTests(unittest.TestCase):
    def test_token_span_mask_excludes_special_tokens_and_keeps_exemplars(self):
        # [CLS]=101, word1=2500, exemplar=1008, '.'=1012, neg=3600, '.'=1012, [SEP]=102, [PAD]=0
        ids = torch.tensor([[101, 2500, EXEMPLAR_TOKEN_ID, 1012, 3600, 1012, 102, 0]])
        mask = token_span_mask(ids)
        self.assertEqual(mask.tolist(), [False, True, True, False, True, False, False, False])

    def test_generate_text_masks_isolates_phrases_and_groups_exemplars(self):
        # [CLS]=101, target=2500, ex1=1008, ex2=1008, '.'=1012, neg1=3600, neg2=3601, '.'=1012, [SEP]=102
        ids = torch.tensor([[101, 2500, EXEMPLAR_TOKEN_ID, EXEMPLAR_TOKEN_ID, 1012, 3600, 3601, 1012, 102]])
        attn, pos = generate_text_masks(ids)
        # Target + 2 exemplars (indices 1..3) attend to each other, NOT to negative phrase (indices 5..6)
        self.assertTrue(attn[0, 1:4, 1:4].all())
        self.assertFalse(attn[0, 1:4, 5:7].any())
        self.assertFalse(attn[0, 5:7, 1:4].any())
        self.assertTrue(attn[0, 5:7, 5:7].all())
        # Position IDs reset per phrase
        self.assertEqual(pos[0].tolist(), [0, 0, 1, 2, 0, 0, 1, 0, 0])

    def test_build_multiclass_caption_and_phrase_token_span(self):
        vocab = ['apples', 'birds', 'cars', 'dogs', 'eggs']
        caption, idx = build_multiclass_caption('birds', vocab, max_classes=4, rng=random.Random(42))
        phrases = [p.strip() for p in caption.rstrip(' .').split(' . ')]
        self.assertEqual(len(phrases), 4)
        self.assertEqual(phrases[idx], 'birds')
        # Simulate tokenized IDs with 3 phrases separated by 1012 ('.')
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

    def test_compute_7layer_loss_multiclass_backpropagates_to_all_stages(self):
        # 5 active tokens: 2 target tokens (True), 3 negative class tokens (False)
        boxes_final = torch.rand(20, 4, requires_grad=True)
        logits_final = torch.randn(20, 5, requires_grad=True)
        aux_boxes = [torch.rand(20, 4, requires_grad=True) for _ in range(5)]
        aux_logits = [torch.randn(20, 5, requires_grad=True) for _ in range(5)]
        enc_boxes = torch.rand(20, 4, requires_grad=True)
        enc_logits = torch.randn(20, 5, requires_grad=True)
        stage_out = dict(
            boxes=boxes_final,
            logits=logits_final,
            aux_boxes=aux_boxes,
            aux_logits=aux_logits,
            enc_boxes=enc_boxes,
            enc_logits=enc_logits,
            target_tokens=torch.tensor([True, True, False, False, False]),
        )
        sample = dict(
            supervision_points=torch.tensor([[0.3, 0.3], [0.7, 0.7]]),
            exemplar_indices=torch.tensor([0]),
            exemplar_boxes=torch.tensor([[0.2, 0.2, 0.4, 0.4]]),
        )
        loss, _ = compute_7layer_loss(CountingCriterion(), stage_out, sample, 'cpu')
        loss.backward()
        self.assertIsNotNone(boxes_final.grad)
        self.assertIsNotNone(enc_boxes.grad)
        self.assertTrue(all(b.grad is not None for b in aux_boxes))
        # Verify gradients flow to negative class token logits too
        self.assertTrue((logits_final.grad[:, 2:].abs().sum() > 0).item())


if __name__ == '__main__':
    unittest.main()
