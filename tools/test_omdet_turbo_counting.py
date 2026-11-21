import random
import unittest
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from transformers import OmDetTurboConfig, OmDetTurboForObjectDetection, SwinConfig, CLIPTextConfig
from models.yoloworld_counting import CountingCriterion
from models.omdet_turbo_counting import OmDetTurboCounting, build_omdet_class_list, _is_stem_or_stage0_param
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

    def test_same_space_raw_similarity_and_step0_zero_mean_visual_prior(self):
        """Verify step-0 raw self-similarity > 0.99 AND visual logits start near -3.5 (no background cone explosion)."""
        tiny = _make_tiny_omdet()
        model = OmDetTurboCounting(
            'tiny-omdet', device='cpu', unfreeze_backbone_stages=True, freeze_class_heads=True,
            use_exemplars=True, detector=tiny, processor=object(),
        )
        # 1) Same-space raw similarity
        feat = torch.randn(1, 32, 16, 16)
        feat[:, :, 4:8, 4:8] = torch.ones(1, 32, 1, 1) * 5.0
        boxes = torch.tensor([[0.25, 0.25, 0.50, 0.50]])
        ex_raw = model._roi_pool_level(feat, boxes)
        sim = torch.einsum('bchw,ec->behw', F.normalize(feat, dim=1), F.normalize(ex_raw, dim=-1))
        self.assertGreater(float(sim[0, 0, 5, 5].item()), 0.99)

        # 2) Even if all queries live in a tight positive cone (simulating pretrained objectness cone),
        #    decoupled zero-mean vis_q_proj / vis_e_proj + vis_bias(-3.5) keeps step-0 mean near -3.5 (p < 0.1)!
        cone_q = torch.ones(1, 100, 32) + 0.1 * torch.randn(1, 100, 32)
        cone_e = torch.ones(3, 32) + 0.1 * torch.randn(3, 32)
        v_logits = model._visual_dot_logits(cone_q, cone_e)
        self.assertLess(float(v_logits.mean().item()), -1.5)
        self.assertIsNone(model.detector.encoder.eval_size)

    def test_v3_all_modes_unfrozen_swin_biprompt_fpn_and_bucket_clipping(self):
        """Verify Swin stages 1-3 + Bi-Prompt FPN + 7-stage heads + decoupled visual heads all get grads,
        while Swin stem/stage0, CLIP language backbone, and 7 CLIP class anchors stay strictly frozen."""
        tiny = _make_tiny_omdet()
        model = OmDetTurboCounting(
            'tiny-omdet', device='cpu', unfreeze_backbone_stages=True, freeze_class_heads=True,
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

        # 1) Text-Only
        model.zero_grad(set_to_none=True)
        st_txt = model.forward_stages(
            pixel_values, classes_ids, classes_mask, tasks_ids, tasks_mask, classes_struct,
            target_idx=1, exemplar_boxes=None, use_text=True,
        )
        self.assertEqual(st_txt['logits'].shape, (12, 3))
        compute_7layer_loss(CountingCriterion(), st_txt, sample, 'cpu')[0].backward()

        # 2) Exemplar-Only
        model.zero_grad(set_to_none=True)
        st_ex = model.forward_stages(
            pixel_values, classes_ids[:1], classes_mask[:1], tasks_ids, tasks_mask, torch.tensor([1]),
            target_idx=0, exemplar_boxes=ex_boxes, use_text=False,
        )
        self.assertEqual(st_ex['logits'].shape, (12, 2))
        compute_7layer_loss(CountingCriterion(), st_ex, sample, 'cpu')[0].backward()

        # 3) Multimodal + 4-bucket independent clipping & optimizer step
        opt = torch.optim.AdamW(model.parameter_groups())
        bb_p, det_p, fus_p, cal_p = model.parameter_buckets()
        self.assertTrue(len(bb_p) > 0 and len(det_p) > 0 and len(fus_p) > 0 and len(cal_p) > 0)
        vbias_before = float(model.vis_logit_bias.item())

        opt.zero_grad(set_to_none=True)
        st_mm = model.forward_stages(
            pixel_values, classes_ids, classes_mask, tasks_ids, tasks_mask, classes_struct,
            target_idx=1, exemplar_boxes=ex_boxes, use_text=True,
        )
        self.assertEqual(st_mm['logits'].shape, (12, 5))
        compute_7layer_loss(CountingCriterion(), st_mm, sample, 'cpu')[0].backward()

        # Verify ALL trainable blocks received real non-zero gradients
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in bb_p))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.prompt_fusion_layers.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.decoder.decoder_bbox_head.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.exemplar_proj.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.sim_fusion_convs.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vis_q_proj.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vis_e_proj.parameters()))
        self.assertTrue(model.prompt_fusion_layers[0].null_k.grad is not None and model.prompt_fusion_layers[0].null_k.grad.abs().sum() > 0)

        # Verify frozen blocks stayed strictly frozen
        for name, param in model.detector.vision_backbone.named_parameters():
            if _is_stem_or_stage0_param(name):
                self.assertIsNone(param.grad, f'Stem/stage0 param {name} should stay frozen')
        self.assertTrue(all(p.grad is None for p in model.detector.language_backbone.parameters()))
        self.assertTrue(all(p.grad is None for p in model.detector.decoder.encoder_class_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.detector.decoder.decoder_class_head.parameters()))

        for bucket in (bb_p, det_p, fus_p, cal_p):
            torch.nn.utils.clip_grad_norm_(bucket, 1.0)
        opt.step()
        self.assertGreater(abs(float(model.vis_logit_bias.item()) - vbias_before), 1e-4)


if __name__ == '__main__':
    unittest.main()
