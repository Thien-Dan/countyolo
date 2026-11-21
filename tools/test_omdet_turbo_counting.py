import random
import unittest
from PIL import Image
import torch
from torch import nn
from transformers import OmDetTurboConfig, OmDetTurboForObjectDetection, SwinConfig, CLIPTextConfig
from models.yoloworld_counting import CountingCriterion
from models.omdet_turbo_counting import OmDetTurboCounting, build_omdet_class_list
from tools.train_omdet_turbo_counting import prepare_omdet_sample
from tools.train_llmdet_counting import compute_7layer_loss


def _make_tiny_omdet():
    """Create a tiny in-memory OmDetTurboForObjectDetection (no downloads, runs on CPU in <0.5s)."""
    cfg = OmDetTurboConfig(
        use_timm_backbone=False,
        backbone=None,
        backbone_config=SwinConfig(
            image_size=64, patch_size=4, embed_dim=16, depths=[1, 1, 1],
            num_heads=[2, 2, 2], window_size=4, out_features=['stage1', 'stage2', 'stage3'],
        ),
        apply_layernorm_after_vision_backbone=False,
        text_config=CLIPTextConfig(
            vocab_size=1000, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32,
        ),
        text_projection_in_dim=64,
        text_projection_out_dim=64,
        class_embed_dim=64,
        task_encoder_hidden_dim=128,
        encoder_in_channels=[16, 32, 64],
        encoder_hidden_dim=32,
        vision_features_channels=[32, 32, 32],
        encoder_projection_indices=[2],
        encoder_attention_heads=4,
        encoder_dim_feedforward=64,
        encoder_layers=1,
        num_feature_levels=3,
        decoder_hidden_dim=32,
        decoder_num_heads=4,
        decoder_num_layers=2,
        decoder_dim_feedforward=64,
        decoder_num_points=2,
        num_queries=12,
        class_distance_type='cosine',
    )
    return OmDetTurboForObjectDetection(cfg)


class OmDetTurboCountingTests(unittest.TestCase):
    def test_build_omdet_class_list(self):
        vocab = ['apples', 'birds', 'cars', 'dogs', 'eggs']
        classes, task, idx = build_omdet_class_list('birds', vocab, max_classes=4, rng=random.Random(7))
        self.assertEqual(len(classes), 4)
        self.assertEqual(classes[idx], 'birds')
        self.assertTrue(task.startswith('Detect ') and task.endswith('.'))

    def test_prepare_omdet_sample_with_crop_and_overflow(self):
        image = Image.new('RGB', (400, 300))
        pts = [[float(i % 380 + 10), float((i * 7) % 280 + 10)] for i in range(1100)]
        record = dict(
            points=pts,
            box_examples_coordinates=[[[40, 40], [70, 40], [70, 70], [40, 70]]],
        )
        sample = prepare_omdet_sample(image, record, flip=True, crop_scale_range=(0.8, 0.9), max_queries=900)
        self.assertLessEqual(len(sample['supervision_points']), 900)
        self.assertEqual(len(sample['exemplar_indices']), 1)
        self.assertTrue((sample['supervision_points'] >= 0).all() and (sample['supervision_points'] <= 1).all())

    def test_all_three_modes_forward_backward_frozen_clip_heads_and_cache_safety(self):
        """Verify Exemplar-Only, Text+Exemplar, and Text-Only forward/backward, FrozenBN, and frozen CLIP heads."""
        tiny = _make_tiny_omdet()
        model = OmDetTurboCounting(
            'tiny-omdet', device='cpu', freeze_backbones=True, freeze_class_heads=True,
            use_exemplars=True, detector=tiny, processor=object(),
        )
        model.set_training(True)
        bns = [m for m in model.detector.modules() if isinstance(m, nn.BatchNorm2d)]
        self.assertTrue(len(bns) > 0 and all(not bn.training for bn in bns))

        pixel_values = torch.randn(1, 3, 64, 64)
        classes_ids = torch.tensor([[10, 20, 30, 0], [10, 40, 50, 0], [10, 60, 70, 0]])
        classes_mask = (classes_ids != 0).long()
        tasks_ids = torch.tensor([[10, 11, 12, 13, 0]])
        tasks_mask = (tasks_ids != 0).long()
        classes_struct = torch.tensor([3], dtype=torch.long)
        ex_boxes = torch.tensor([[0.2, 0.2, 0.4, 0.4], [0.6, 0.6, 0.8, 0.8]])
        sample = dict(
            supervision_points=torch.tensor([[0.3, 0.3], [0.7, 0.7]]),
            exemplar_indices=torch.tensor([0, 1]),
            exemplar_boxes=ex_boxes,
        )

        # 1) Text-Only mode (K=3 columns, 1 True)
        model.zero_grad(set_to_none=True)
        st_txt = model.forward_stages(
            pixel_values, classes_ids, classes_mask, tasks_ids, tasks_mask, classes_struct,
            target_idx=1, exemplar_boxes=None, use_text=True,
        )
        self.assertEqual(st_txt['logits'].shape, (12, 3))
        self.assertEqual(st_txt['target_tokens'].tolist(), [False, True, False])
        loss_txt, _ = compute_7layer_loss(CountingCriterion(), st_txt, sample, 'cpu')
        loss_txt.backward()

        # 2) Exemplar-Only mode (E=2 visual columns, both True)
        model.zero_grad(set_to_none=True)
        st_ex = model.forward_stages(
            pixel_values, classes_ids[:1], classes_mask[:1], tasks_ids, tasks_mask, torch.tensor([1]),
            target_idx=0, exemplar_boxes=ex_boxes, use_text=False,
        )
        self.assertEqual(st_ex['logits'].shape, (12, 2))
        self.assertEqual(st_ex['target_tokens'].tolist(), [True, True])
        loss_ex, _ = compute_7layer_loss(CountingCriterion(), st_ex, sample, 'cpu')
        loss_ex.backward()

        # 3) Multimodal Text+Exemplar mode (K+E=5 columns: target_idx=1 + 2 visual exemplars True)
        model.zero_grad(set_to_none=True)
        st_mm = model.forward_stages(
            pixel_values, classes_ids, classes_mask, tasks_ids, tasks_mask, classes_struct,
            target_idx=1, exemplar_boxes=ex_boxes, use_text=True,
        )
        self.assertEqual(st_mm['logits'].shape, (12, 5))
        self.assertEqual(st_mm['target_tokens'].tolist(), [False, True, False, True, True])
        loss_mm, _ = compute_7layer_loss(CountingCriterion(), st_mm, sample, 'cpu')
        loss_mm.backward()

        # Verify ALL trainable blocks receive real non-zero gradients
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.decoder.encoder_bbox_head.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.decoder.decoder_bbox_head.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.decoder.task_encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.exemplar_proj.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.sim_fusion_convs.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.exemplar_to_task.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vis_query_head.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vis_ex_head.parameters()))
        self.assertTrue(model.vis_logit_scale.grad is not None and model.vis_logit_scale.grad.abs().sum() > 0)
        self.assertTrue(model.vis_logit_bias.grad is not None and model.vis_logit_bias.grad.abs().sum() > 0)
        # Backbones AND all 7 CLIP class projection heads must stay strictly frozen!
        self.assertTrue(all(p.grad is None for p in model.detector.vision_backbone.parameters()))
        self.assertTrue(all(p.grad is None for p in model.detector.language_backbone.parameters()))
        self.assertTrue(all(p.grad is None for p in model.detector.decoder.encoder_class_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.detector.decoder.decoder_class_head.parameters()))


if __name__ == '__main__':
    unittest.main()
