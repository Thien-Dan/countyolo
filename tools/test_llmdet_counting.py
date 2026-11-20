import unittest
from PIL import Image
import torch
from models.yoloworld_counting import CountingCriterion
from models.llmdet_counting import token_span_mask
from tools.train_llmdet_counting import prepare_llmdet_sample, compute_7layer_loss


class LLMDetCountingTests(unittest.TestCase):
    def test_token_span_mask_excludes_special_tokens(self):
        # [CLS]=101, word1=2500, word2=3600, '.'=1012, [SEP]=102, [PAD]=0
        ids = torch.tensor([[101, 2500, 3600, 1012, 102, 0]])
        mask = token_span_mask(ids)
        self.assertEqual(mask.tolist(), [False, True, True, False, False, False])

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

    def test_compute_7layer_loss_backpropagates_to_all_stages(self):
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
            target_tokens=torch.tensor([False, True, True, False, False]),
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


if __name__ == '__main__':
    unittest.main()
