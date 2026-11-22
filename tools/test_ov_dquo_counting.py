"""Fast synthetic CPU unit tests for OV-DQUO Open-World Counting (zero downloads, <2s)."""
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch

from models.yoloworld_counting import CountingCriterion
from models.ov_dquo_counting import OVDQUOCounting, build_ov_dquo_class_list, log_and_logit
from tools.train_ov_dquo_counting import compute_ov_dquo_7layer_loss, prepare_ov_dquo_sample


def _mock_text_encoder(prompts, device='cpu'):
    vecs = []
    for p in prompts:
        g = torch.Generator().manual_seed(abs(hash(p)) % (2 ** 31))
        vecs.append(torch.randn(32, generator=g))
    return torch.stack(vecs, dim=0).to(device)


def _make_tiny_ov_dquo(use_exemplars=True):
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
        enc_layers=2,
        dec_layers=3,
    )
    return OVDQUOCounting(
        variant='vitb16',
        checkpoint_path=None,
        device='cpu',
        resolution=128,
        num_queries=24,
        use_exemplars=use_exemplars,
        grad_checkpoint=False,
        text_encoder_fn=_mock_text_encoder,
        _tiny_cfg=tiny_cfg,
    )


class TestOVDQUOCounting(unittest.TestCase):
    def test_log_and_logit_numerical_stability_and_gradients(self):
        a = torch.tensor([-20.0, -3.0, 0.0, 3.0, 20.0], requires_grad=True)
        b = torch.tensor([-20.0, 3.0, 0.0, -3.0, 20.0], requires_grad=True)
        z = log_and_logit(a, b)
        self.assertTrue(torch.isfinite(z).all().item(), f'Non-finite log_and_logit: {z}')
        # Product of probabilities checks:
        # (3, -3) -> sigmoid(3)*sigmoid(-3) ~ 0.95*0.047 = 0.045 -> z < -2.5 (rejects wrong class!)
        self.assertLess(z[3].item(), -2.5)
        self.assertLess(z[1].item(), -2.5)
        # (20, 20) -> high positive
        self.assertGreater(z[4].item(), 10.0)
        z.sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all().item() and (a.grad >= 0).all().item())
        self.assertTrue(torch.isfinite(b.grad).all().item() and (b.grad >= 0).all().item())

    def test_rectangular_padding_valid_ratios_and_dense_roi(self):
        model = _make_tiny_ov_dquo(use_exemplars=True)
        model.set_training(False)
        # 200x100 landscape image -> resizes to 128x64 inside 128x128 (bottom half masked True)
        img = Image.fromarray(np.full((100, 200, 3), 128, dtype=np.uint8))
        pv, mask = model.preprocess_image(img, device=torch.device('cpu'))
        self.assertEqual(tuple(pv.shape), (1, 3, 128, 128))
        self.assertFalse(mask[0, :64, :128].any().item())
        self.assertTrue(mask[0, 64:, :].all().item())

        ex = torch.tensor([[0.1, 0.2, 0.4, 0.8]], dtype=torch.float32)
        with torch.no_grad():
            boxes, logits = model(img, 'apples', exemplar_boxes=ex, use_text=True)
        self.assertEqual(tuple(boxes.shape), (24, 4))
        self.assertEqual(tuple(logits.shape), (24,))
        self.assertTrue(torch.isfinite(boxes).all().item())
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_all_three_modes_7layer_forward_backward_and_freezing(self):
        torch.manual_seed(123)
        model = _make_tiny_ov_dquo(use_exemplars=True)
        model.set_training(True)
        criterion = CountingCriterion()

        img = Image.fromarray(np.random.default_rng(5).integers(0, 255, (150, 220, 3), dtype=np.uint8))
        record = dict(
            points=[[30, 30], [90, 50], [150, 80], [190, 120]],
            box_examples_coordinates=[
                [[20, 20], [40, 20], [40, 40], [20, 40]],
                [[80, 40], [100, 40], [100, 60], [80, 60]],
            ],
        )
        sample = prepare_ov_dquo_sample(img, record, flip=False, crop_scale_range=None, max_queries=24)
        vocab = ['apples', 'cars', 'coins', 'pills', 'cups']

        model.zero_grad(set_to_none=True)
        for mode in ('multimodal', 'text_only', 'ex_only'):
            use_txt = mode != 'ex_only'
            ex = sample['exemplar_boxes'] if mode != 'text_only' else None
            classes, tidx = build_ov_dquo_class_list('apples', vocab, max_classes=4) if use_txt else (['object'], 0)
            out = model.forward_stages(
                sample['pil_image'], classes, target_idx=tidx, exemplar_boxes=ex, use_text=use_txt,
            )
            loss, _ = compute_ov_dquo_7layer_loss(criterion, out, sample, device='cpu', neg_class_weight=0.5)
            self.assertTrue(torch.isfinite(loss).item(), f'Non-finite loss in {mode}: {loss}')
            (loss / 3.0).backward()

        # Verify parameter buckets partition trainable_parameters cleanly
        bb_p, tr_p, ad_p, cal_p = model.parameter_buckets()
        self.assertEqual(
            {id(p) for b in (bb_p, tr_p, ad_p, cal_p) for p in b},
            {id(p) for _, p in model.trainable_parameters()},
        )

        # Verify ALL trainable blocks received finite non-zero gradients
        for label, params in [
            ('interpolate1', list(model.backbone[0].interpolate1.parameters())),
            ('input_proj', list(model.input_proj.parameters())),
            ('encoder', list(model.transformer.encoder.parameters())),
            ('decoder', list(model.transformer.decoder.parameters())),
            ('enc_out_bbox', list(model.transformer.enc_out_bbox_embed.parameters())),
            ('enc_out_cls', list(model.transformer.enc_out_class_embed.parameters())),
            ('text_proj', list(model.transformer.text_proj.parameters())),
            ('mem_fusion', list(model.mem_fusion.parameters())),
            ('enc_prompt_prior', [model.enc_prompt_prior]),
            ('sim_inject', list(model.sim_inject.parameters())),
            ('vis_proj', list(model.vis_proj.parameters())),
            ('dec_vis_proj', list(model.dec_vis_proj.parameters())),
            ('cal_bucket', cal_p),
        ]:
            gsum = sum(p.grad.abs().sum().item() for p in params if p.grad is not None)
            self.assertTrue(np.isfinite(gsum) and gsum > 0.0, f'{label} bad grad: {gsum}')

        for lvl in range(model.num_dec_layers):
            bg = sum(p.grad.abs().sum().item() for p in model.bbox_embed[lvl].parameters() if p.grad is not None)
            cg = sum(p.grad.abs().sum().item() for p in model.class_embed[lvl].parameters() if p.grad is not None)
            self.assertTrue(np.isfinite(bg) and bg > 0.0, f'bbox_embed[{lvl}] bad grad: {bg}')
            self.assertTrue(np.isfinite(cg) and cg > 0.0, f'class_embed[{lvl}] bad grad: {cg}')

        # Verify frozen EVA02-CLIP visual backbone strictly received NO gradient
        self.assertTrue(all(p.grad is None for p in model.backbone[0].visual.parameters()))

    def test_official_checkpoint_loader_and_tiling_exemplars(self):
        model = _make_tiny_ov_dquo(use_exemplars=True)
        model.set_training(False)

        # Simulate an official OV-DQUO checkpoint (with DDP prefix, label_enc, shared class_embed.0)
        raw_sd = {}
        for k, v in model.state_dict().items():
            if any(k.startswith(p) for p in (
                'mem_fusion.', 'enc_prompt_prior', 'sim_inject.', 'clip_text_',
                'vis_proj.', 'dec_vis_proj.', 'vis_mix', 'clip_vis_', 'mm_alpha', 'mm_bias',
            )):
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

        # Test encode_exemplars + quadrant crop inference + trainable state roundtrip
        img = Image.fromarray(np.random.default_rng(11).integers(0, 255, (128, 192, 3), dtype=np.uint8))
        ex = torch.tensor([[0.1, 0.1, 0.4, 0.4]], dtype=torch.float32)
        ex_feats = model.encode_exemplars(img, ex)
        self.assertIsInstance(ex_feats, dict)
        self.assertEqual(tuple(ex_feats['ex_clip'].shape), (1, 32))
        self.assertEqual(tuple(ex_feats['ex_mem'].shape), (1, 32))
        crop = img.crop((0, 0, 96, 64))
        with torch.no_grad():
            qb, ql = model(crop, 'apples', exemplar_feats=ex_feats, use_text=True)
        self.assertEqual(tuple(qb.shape), (24, 4))
        self.assertTrue(torch.isfinite(ql).all().item())
        model.import_trainable_state(model.export_trainable_state())


if __name__ == '__main__':
    unittest.main()
