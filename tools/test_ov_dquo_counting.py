"""Comprehensive synthetic CPU verification suite for OV-DQUO v3 Open-World Counting (<3s).

Verifies all 6 v3 architectural fixes, module logic, Step-0 numerical invariants,
forward/backward gradient flow across all 4 training modes (mm, text_only, ex_only, absent),
grad_checkpoint parity, Text-Only variant, official checkpoint loading, and 2x scale-matched tiling.
"""
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from models.yoloworld_counting import CountingCriterion
from models.ov_dquo_counting import OVDQUOCounting, build_ov_dquo_class_list, log_and_logit
from tools.train_ov_dquo_counting import (
    _stage_loss,
    compute_ov_dquo_7layer_loss,
    predict_with_tiling,
    prepare_ov_dquo_sample,
)


def _mock_text_encoder(prompts, device='cpu'):
    vecs = []
    for p in prompts:
        g = torch.Generator().manual_seed(abs(hash(p)) % (2 ** 31))
        vecs.append(torch.randn(32, generator=g))
    return torch.stack(vecs, dim=0).to(device)


def _make_tiny_ov_dquo(use_exemplars=True, grad_checkpoint=False, enc_layers=4):
    tiny_cfg = dict(
        img_size_pretrain=64,
        patch_size=16,
        embed_dim=48,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        text_dim=32,
        pt_hw_seq_len=4,
        out_indices=(1, 2, 3),
        default_resolution=128,
        hidden_dim=32,
        nheads=4,
        dim_feedforward=64,
        enc_layers=enc_layers,
        dec_layers=3,
    )
    return OVDQUOCounting(
        variant='vitb16',
        checkpoint_path=None,
        device='cpu',
        resolution=128,
        num_queries=24,
        use_exemplars=use_exemplars,
        grad_checkpoint=grad_checkpoint,
        text_encoder_fn=_mock_text_encoder,
        _tiny_cfg=tiny_cfg,
    )


def _make_sample(seed=5):
    img = Image.fromarray(np.random.default_rng(seed).integers(0, 255, (150, 220, 3), dtype=np.uint8))
    record = dict(
        points=[[30, 30], [90, 50], [150, 80], [190, 120]],
        box_examples_coordinates=[
            [[20, 20], [40, 20], [40, 40], [20, 40]],
            [[140, 70], [160, 70], [160, 90], [140, 90]],
        ],
    )
    return prepare_ov_dquo_sample(img, record, flip=False, crop_scale_range=None, max_queries=24)


class TestOVDQUOCountingV3(unittest.TestCase):
    def test_1_log_and_logit_stability_and_gradients(self):
        a = torch.tensor([-20.0, -3.0, 0.0, 3.0, 20.0], requires_grad=True)
        b = torch.tensor([-20.0, 3.0, 0.0, -3.0, 20.0], requires_grad=True)
        z = log_and_logit(a, b)
        self.assertTrue(torch.isfinite(z).all().item(), f'Non-finite log_and_logit: {z}')
        self.assertLess(z[3].item(), -2.5)
        self.assertLess(z[1].item(), -2.5)
        self.assertGreater(z[4].item(), 10.0)
        z.sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all().item() and (a.grad >= 0).all().item())
        self.assertTrue(torch.isfinite(b.grad).all().item() and (b.grad >= 0).all().item())

    def test_2_step0_invariants_fix1_fix3_fix4(self):
        """Verify Step-0 mathematical invariants for v3.1:
        - Fix #1: `dec_text_delta` starts near zero (preserving zero-shot EVA02-CLIP) with active 1.5x slope.
        - Fix #3: `vis_mem_gate` has active slope (`1.5 * tanh(0.10) ~= 0.149`) AND `vis_sem > +2.0` at Step 0.
        - Fix #4: `propose_from_memory` NEVER contaminates `enc_class_topk` with `enc_prompt_bias`.
        """
        torch.manual_seed(7)
        model = _make_tiny_ov_dquo(use_exemplars=True)
        model.set_training(False)

        # Fix #3 active-slope visual memory gate (~0.149, slope ~1.49)
        g_vis = (1.5 * torch.tanh(model.vis_mem_gate)).item()
        self.assertGreater(g_vis, 0.10)
        self.assertLess(g_vis, 0.20)

        # Fix #1 & #3 Step-0 check: identical normalized vector as roi_feat, text_embed, ex_clip
        dense_vec = F.normalize(torch.randn(1, 32, 1, 1), dim=1).expand(1, 32, 8, 8).contiguous()
        dense_mask = torch.zeros((1, 8, 8), dtype=torch.bool)
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
        obj_logit = torch.tensor([2.0], dtype=torch.float32)
        qh = torch.randn(1, 32)
        ex_clip = dense_vec[0, :, 0, 0].unsqueeze(0)  # cosine = 1.0!
        ex_mem = torch.randn(1, 32)
        with torch.no_grad():
            st = model._score_stage(
                boxes, obj_logit, qh, dense_vec, dense_mask,
                text_embeds=ex_clip, target_idx=0, ex_clip=ex_clip, ex_mem=ex_mem, is_enc=False,
            )
        # Step-0 text residual norm stays tiny (< 0.05) so zero-shot CLIP cosine ~1.0 is preserved
        self.assertLess(model._last_text_delta_norm, 0.05)
        self.assertGreater(st['text_sem'].item(), 10.0)
        # Step-0 vis_sem must remain strongly positive (> +2.0)!
        self.assertGreater(st['vis_sem'].item(), 2.0, f'Step-0 vis_sem collapsed: {st["vis_sem"].item()}')

        # Fix #4: `enc_class_topk` must equal pure `enc_class` gathered at `topk_inds` (no bias added!)
        mem = torch.randn(1, 20, 32)
        mask_flat = torch.zeros((1, 20), dtype=torch.bool)
        spatial_shapes = torch.tensor([[4, 4], [2, 2]], dtype=torch.long)
        bias = torch.full((1, 20), 5.0)
        with torch.no_grad():
            prop_biased = model.transformer.propose_from_memory(mem, mask_flat, spatial_shapes, enc_prompt_bias=bias)
            prop_plain = model.transformer.propose_from_memory(mem, mask_flat, spatial_shapes, enc_prompt_bias=None)
        # Adding constant +5.0 to all locations leaves topk_inds unchanged AND must leave enc_class_topk UNCHANGED!
        self.assertTrue(torch.equal(prop_biased['topk_inds'], prop_plain['topk_inds']))
        self.assertTrue(
            torch.allclose(prop_biased['enc_class_topk'], prop_plain['enc_class_topk'], atol=1e-6),
            'enc_prompt_bias leaked into enc_class_topk!',
        )

    def test_3_single_matcher_parity_and_absent_loss(self):
        """Verify Fix #2: single-modality loss matches CountingCriterion exactly, and empty-GT absent loss is clean."""
        torch.manual_seed(19)
        criterion = CountingCriterion()
        boxes = torch.rand(16, 4).clamp(0.1, 0.9)
        boxes.requires_grad_(True)
        txt_logits = torch.randn(16, 4, requires_grad=True)
        vis_logits = torch.randn(16, requires_grad=True)
        pts = torch.tensor([[0.2, 0.3], [0.7, 0.8]], dtype=torch.float32)
        ex_idx = torch.tensor([0], dtype=torch.long)
        ex_xyxy = torch.tensor([[0.15, 0.25, 0.25, 0.35]], dtype=torch.float32)

        # Text-only parity with CountingCriterion
        t_mask = torch.tensor([False, True, False, False])
        ref_loss, _ = criterion(boxes, txt_logits, pts, ex_idx, ex_xyxy, target_tokens=t_mask, neg_class_weight=0.5)
        st_txt = dict(boxes=boxes, text_logits=txt_logits, vis_logits=None, mm_logits=None)
        v3_loss, _ = _stage_loss(criterion, st_txt, pts, ex_idx, ex_xyxy, target_idx=1, device='cpu', neg_class_weight=0.5)
        self.assertTrue(torch.allclose(ref_loss, v3_loss, atol=1e-5), f'{ref_loss} vs {v3_loss}')

        # Ex-only parity with CountingCriterion
        ref_v, _ = criterion(boxes, vis_logits, pts, ex_idx, ex_xyxy)
        st_vis = dict(boxes=boxes, text_logits=None, vis_logits=vis_logits, mm_logits=None)
        v3_v, _ = _stage_loss(criterion, st_vis, pts, ex_idx, ex_xyxy, target_idx=0, device='cpu', neg_class_weight=0.5)
        self.assertTrue(torch.allclose(ref_v, v3_v, atol=1e-5), f'{ref_v} vs {v3_v}')

        # Empty-GT absent batch: finite positive focal loss, finite gradients, zero geometry crash
        empty_pts = torch.empty((0, 2), dtype=torch.float32)
        empty_idx = torch.empty((0,), dtype=torch.long)
        empty_box = torch.empty((0, 4), dtype=torch.float32)
        abs_loss, abs_parts = _stage_loss(criterion, st_txt, empty_pts, empty_idx, empty_box, target_idx=1, device='cpu', neg_class_weight=0.5)
        self.assertTrue(torch.isfinite(abs_loss).item() and abs_loss.item() > 0.0)
        self.assertEqual(abs_parts['center'].item(), 0.0)
        abs_loss.backward()
        self.assertTrue(torch.isfinite(txt_logits.grad).all().item())

    def test_4_all_four_modes_7layer_forward_backward_and_grad_checkpoint(self):
        """Verify ALL trainable v3 blocks receive finite non-zero gradients with grad_checkpoint=False AND True."""
        sample = _make_sample(seed=5)
        vocab = ['apples', 'cars', 'coins', 'pills', 'cups']
        criterion = CountingCriterion()

        for use_ckpt in (False, True):
            torch.manual_seed(123)
            model = _make_tiny_ov_dquo(use_exemplars=True, grad_checkpoint=use_ckpt, enc_layers=4)
            model.set_training(True)
            model.zero_grad(set_to_none=True)

            for mode in ('multimodal', 'text_only', 'ex_only', 'absent'):
                if mode == 'absent':
                    use_txt, ex = True, None
                    classes, tidx = build_ov_dquo_class_list('cars', ['cars', 'coins', 'pills', 'cups'], max_classes=4)
                    step_sample = dict(
                        sample,
                        supervision_points=torch.empty((0, 2), dtype=torch.float32),
                        exemplar_indices=torch.empty((0,), dtype=torch.long),
                        exemplar_boxes=torch.empty((0, 4), dtype=torch.float32),
                    )
                else:
                    use_txt = mode != 'ex_only'
                    ex = sample['exemplar_boxes'] if mode != 'text_only' else None
                    classes, tidx = build_ov_dquo_class_list('apples', vocab, max_classes=4) if use_txt else (['object'], 0)
                    step_sample = sample

                out = model.forward_stages(
                    step_sample['pil_image'], classes, target_idx=tidx, exemplar_boxes=ex, use_text=use_txt,
                )
                loss, parts = compute_ov_dquo_7layer_loss(criterion, out, step_sample, device='cpu', neg_class_weight=0.5, prior_weight=0.10)
                self.assertTrue(torch.isfinite(loss).item(), f'Non-finite loss in {mode} (ckpt={use_ckpt}): {loss}')
                self.assertTrue(torch.isfinite(parts['prior']).item())
                (loss / 4.0).backward()

            # Verify parameter buckets partition trainable_parameters cleanly
            bb_p, tr_p, ad_p, cal_p = model.parameter_buckets()
            self.assertEqual(
                {id(p) for b in (bb_p, tr_p, ad_p, cal_p) for p in b},
                {id(p) for _, p in model.trainable_parameters()},
            )

            # Step 0 check: EVERY trainable parameter has finite non-None grad
            for n, p in model.trainable_parameters():
                self.assertIsNotNone(p.grad, f'Missing grad on {n} (ckpt={use_ckpt})')
                self.assertTrue(torch.isfinite(p.grad).all().item(), f'Non-finite grad on {n} (ckpt={use_ckpt})')

            # Step 1 check: after 1 optimizer step (unsticking zero-init DETR layers like
            # MSDeformAttn.sampling_offsets.weight and bbox_embed.layers[-1].weight),
            # 100% of ALL trainable parameters (ZERO exceptions!) must get finite non-zero grad across the 4 modes.
            opt = torch.optim.SGD(model.parameter_groups(backbone_lr=1e-2, lr=1e-2, adapter_lr=1e-2, cal_lr=1e-2))
            opt.step()
            model.zero_grad(set_to_none=True)
            for mode in ('multimodal', 'text_only', 'ex_only'):
                use_txt = mode != 'ex_only'
                ex = sample['exemplar_boxes'] if mode != 'text_only' else None
                classes, tidx = build_ov_dquo_class_list('apples', vocab, max_classes=4) if use_txt else (['object'], 0)
                out2 = model.forward_stages(sample['pil_image'], classes, target_idx=tidx, exemplar_boxes=ex, use_text=use_txt)
                l2, _ = compute_ov_dquo_7layer_loss(criterion, out2, sample, device='cpu')
                (l2 / 3.0).backward()
            for n, p in model.trainable_parameters():
                self.assertIsNotNone(p.grad, f'Step-1 missing grad on {n} (ckpt={use_ckpt})')
                self.assertTrue(torch.isfinite(p.grad).all().item(), f'Step-1 non-finite grad on {n} (ckpt={use_ckpt})')
                self.assertGreater(p.grad.abs().sum().item(), 0.0, f'Step-1 zero grad on {n} (ckpt={use_ckpt})')

            # Verify frozen EVA02-CLIP visual backbone strictly received NO gradient
            self.assertTrue(all(p.grad is None for p in model.backbone[0].visual.parameters()))

        # Verify bit-exact forward & backward parity between grad_checkpoint=False and grad_checkpoint=True
        torch.manual_seed(999)
        m_no = _make_tiny_ov_dquo(use_exemplars=True, grad_checkpoint=False, enc_layers=4)
        torch.manual_seed(999)
        m_ck = _make_tiny_ov_dquo(use_exemplars=True, grad_checkpoint=True, enc_layers=4)
        m_no.set_training(True)
        m_ck.set_training(True)
        classes = ['apples', 'cars', 'coins', 'pills']
        o_no = m_no.forward_stages(sample['pil_image'], classes, target_idx=0, exemplar_boxes=sample['exemplar_boxes'], use_text=True)
        o_ck = m_ck.forward_stages(sample['pil_image'], classes, target_idx=0, exemplar_boxes=sample['exemplar_boxes'], use_text=True)
        l_no, _ = compute_ov_dquo_7layer_loss(criterion, o_no, sample, device='cpu')
        l_ck, _ = compute_ov_dquo_7layer_loss(criterion, o_ck, sample, device='cpu')
        l_no.backward()
        l_ck.backward()
        self.assertTrue(torch.allclose(o_no['final']['mm_logits'], o_ck['final']['mm_logits'], atol=1e-5))
        for (n1, p1), (n2, p2) in zip(m_no.trainable_parameters(), m_ck.trainable_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.allclose(p1.grad, p2.grad, atol=1e-4), f'grad_checkpoint mismatch on {n1}')

    def test_5_text_only_variant_forward_backward(self):
        """Verify `use_exemplars=False` variant has clean forward/backward across text_only + absent."""
        torch.manual_seed(42)
        model = _make_tiny_ov_dquo(use_exemplars=False, grad_checkpoint=False, enc_layers=2)
        model.set_training(True)
        sample = _make_sample(seed=9)
        criterion = CountingCriterion()
        vocab = ['apples', 'cars', 'coins', 'pills']

        model.zero_grad(set_to_none=True)
        classes, tidx = build_ov_dquo_class_list('apples', vocab, max_classes=4)
        out = model.forward_stages(sample['pil_image'], classes, target_idx=tidx, exemplar_boxes=None, use_text=True)
        loss, _ = compute_ov_dquo_7layer_loss(criterion, out, sample, device='cpu')
        loss.backward()
        opt = torch.optim.SGD(model.parameter_groups(backbone_lr=1e-2, lr=1e-2, adapter_lr=1e-2, cal_lr=1e-2))
        opt.step()
        model.zero_grad(set_to_none=True)
        out2 = model.forward_stages(sample['pil_image'], classes, target_idx=tidx, exemplar_boxes=None, use_text=True)
        loss2, _ = compute_ov_dquo_7layer_loss(criterion, out2, sample, device='cpu')
        loss2.backward()

        for n, p in model.trainable_parameters():
            self.assertIsNotNone(p.grad, f'Text-only variant missing grad on {n}')
            self.assertTrue(torch.isfinite(p.grad).all().item(), f'Non-finite grad on {n}')
            self.assertGreater(p.grad.abs().sum().item(), 0.0, f'Zero grad on {n}')
        self.assertTrue(all(p.grad is None for p in model.backbone[0].visual.parameters()))

    def test_6_official_checkpoint_loader_and_scale_matched_2x_tiling(self):
        """Verify official OV-DQUO checkpoint compatibility + Fix #6 scale-matched 2x tiling exemplars."""
        model = _make_tiny_ov_dquo(use_exemplars=True, enc_layers=4)
        model.set_training(False)

        v3_prefixes = (
            'mem_fusion.', 'enc_prompt_prior', 'enc_prior_bias',
            'enc_tgt_proj.', 'enc_tgt_gate', 'sim_inject.',
            'clip_text_', 'dec_text_delta.', 'enc_text_delta.',
            'vis_proj.', 'clip_vis_', 'dec_vis_proj.', 'enc_vis_proj.', 'e_mem_vis_proj.',
            'mem_vis_', 'vis_mem_gate', 'mm_alpha', 'mm_bias',
        )
        raw_sd = {}
        for k, v in model.state_dict().items():
            if k.startswith(v3_prefixes):
                continue
            if k.startswith('class_embed.') and not k.startswith('class_embed.0.'):
                continue
            raw_sd['module.' + k] = v.clone()
        raw_sd['module.label_enc.weight'] = torch.randn(101, 32)
        raw_sd['module.classifier'] = torch.randn(1204, 32)

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            ckpt_path = f.name
        torch.save(dict(ema_model=raw_sd), ckpt_path)
        model.load_official_ov_dquo_checkpoint(ckpt_path)

        # Test scale-matched 2x exemplar extraction (2 far-apart exemplars -> 2 half-size windows) + full 2x2 tiling
        img = Image.fromarray(np.random.default_rng(11).integers(0, 255, (160, 240, 3), dtype=np.uint8))
        ex = torch.tensor([[0.05, 0.05, 0.15, 0.15], [0.80, 0.75, 0.92, 0.90]], dtype=torch.float32)
        ex_feats_2x = model.encode_exemplars(img, ex, prompt='apples', use_text=True, scale_zoom=2.0)
        self.assertIsInstance(ex_feats_2x, dict)
        self.assertEqual(set(ex_feats_2x.keys()), {'ex_clip', 'ex_mem'})
        self.assertEqual(tuple(ex_feats_2x['ex_clip'].shape), (2, 32))
        self.assertEqual(tuple(ex_feats_2x['ex_mem'].shape), (2, 32))
        for k, t in ex_feats_2x.items():
            self.assertTrue(torch.isfinite(t).all().item(), f'Non-finite {k}')

        # Force 2x2 tiling trigger (`tile_trigger=1` with threshold `-1.0`)
        tb, ts = predict_with_tiling(model, img, 'apples', exemplar_boxes=ex, use_text=True, threshold=-1.0, tile_trigger=1)
        self.assertEqual(tuple(tb.shape), (4 * 24, 4))
        self.assertEqual(tuple(ts.shape), (4 * 24,))
        self.assertTrue(torch.isfinite(tb).all().item() and torch.isfinite(ts).all().item())

        # Trainable state export/import roundtrip
        model.import_trainable_state(model.export_trainable_state())


if __name__ == '__main__':
    unittest.main()
