"""Fast CPU unit tests for OV-DQUO v1 + SAM2 + Tracking + TemporalIdentity video pipeline (<2s)."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch

from models.ov_dquo_v1_infer import OVDQUOV1Detector
from models.ov_dquo_video_detector import OVDQUOV2Detector, OVDQUOV31Detector
from video.identity import IdentityConfig
from video.ov_dquo_sam2_pipeline import count_video_ov_dquo_sam2


def _mock_text_encoder(prompts, device='cpu'):
    vecs = []
    for p in prompts:
        g = torch.Generator().manual_seed(abs(hash(p)) % (2 ** 31))
        vecs.append(torch.randn(32, generator=g))
    return torch.stack(vecs, dim=0).to(device)


def _make_tiny_v1():
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
    return OVDQUOV1Detector(
        variant='vitb16',
        device='cpu',
        resolution=128,
        num_queries=16,
        use_exemplars=True,
        text_encoder_fn=_mock_text_encoder,
        _tiny_cfg=tiny_cfg,
    )


class _MockSAM2ImagePredictor:
    def __init__(self):
        self.h = 0
        self.w = 0

    def set_image(self, arr):
        self.h, self.w = arr.shape[:2]

    def reset_predictor(self):
        pass

    def predict(self, box, multimask_output=False):
        boxes = np.asarray(box, dtype=np.float32).reshape(-1, 4)
        masks = np.zeros((len(boxes), 1, self.h, self.w), dtype=bool)
        scores = np.ones((len(boxes), 1), dtype=np.float32) * 0.9
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(self.w, max(ix1 + 2, int(x2))), min(self.h, max(iy1 + 2, int(y2)))
            masks[i, 0, iy1:iy2, ix1:ix2] = True
        return masks, scores, None


class _MockSAM2VideoPredictor:
    """Deterministic mock SAM2VideoPredictor that propagates seeded boxes linearly across frames."""
    def __init__(self, height: int, width: int, n_frames: int):
        self.h = height
        self.w = width
        self.n_frames = n_frames
        self.seeds = {}

    def init_state(self, video_path, offload_video_to_cpu=True, offload_state_to_cpu=True):
        return dict(video_path=video_path)

    def reset_state(self, state):
        self.seeds = {}

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box):
        self.seeds[int(obj_id)] = (int(frame_idx), np.asarray(box, dtype=np.float32))

    def propagate_in_video(self, state, start_frame_idx=0, reverse=False, max_frame_num_to_track=None):
        step = -1 if reverse else 1
        end = -1 if reverse else self.n_frames
        frames = list(range(start_frame_idx, end, step))
        if max_frame_num_to_track is not None:
            frames = frames[:max_frame_num_to_track + 1]
        for f in frames:
            ids = list(self.seeds.keys())
            logits = []
            for oid in ids:
                _, (x1, y1, x2, y2) = self.seeds[oid]
                m = torch.full((1, self.h, self.w), -5.0)
                ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                ix2, iy2 = min(self.w, max(ix1 + 2, int(x2))), min(self.h, max(iy1 + 2, int(y2)))
                m[0, iy1:iy2, ix1:ix2] = 5.0
                logits.append(m)
            yield f, ids, logits


class _MockDetectorForPipeline:
    """Yields 2 stable moving/persistent objects across 6 frames plus 1 single-frame false positive on frame 2."""
    def __init__(self):
        self.frame_idx = 0

    def detect_with_features(self, pil_image, prompt, threshold=0.26, tile_trigger=650):
        idx = self.frame_idx
        self.frame_idx += 1
        boxes = [
            [10.0, 10.0, 30.0, 30.0],   # persistent object A (frames 0..5)
            [60.0, 20.0, 82.0, 44.0],   # persistent object B (frames 0..5)
        ]
        if idx == 2:
            boxes.append([90.0, 60.0, 110.0, 80.0])  # 1-frame FP -> must be filtered by Stage 2 (window=3)!
        dm = torch.ones((1, 32, 8, 8), dtype=torch.float16)
        dm[:, :, :, :4] = 2.0
        dmask = torch.zeros((1, 8, 8), dtype=torch.bool)
        return dict(
            boxes_xyxy=np.asarray(boxes, dtype=np.float32),
            scores=np.full((len(boxes),), 0.8, dtype=np.float32),
            dense_map=dm,
            dense_mask=dmask,
            tiled=False,
        )


class TestOVDQUOV1VideoPipeline(unittest.TestCase):
    def test_1_v1_detector_detect_and_appearance_features(self):
        det = _make_tiny_v1()
        img = Image.fromarray(np.random.default_rng(3).integers(0, 255, (100, 160, 3), dtype=np.uint8))
        res = det.detect_with_features(img, 'penguin', threshold=-1.0, tile_trigger=0)
        self.assertGreater(len(res['boxes_xyxy']), 0)
        self.assertEqual(res['boxes_xyxy'].shape[1], 4)
        self.assertTrue((res['boxes_xyxy'][:, 2] >= res['boxes_xyxy'][:, 0]).all())
        self.assertTrue((res['boxes_xyxy'][:, 3] >= res['boxes_xyxy'][:, 1]).all())
        feats = OVDQUOV1Detector.extract_box_features(res['dense_map'], res['dense_mask'], res['boxes_xyxy'][:4], img.size)
        self.assertEqual(len(feats), min(4, len(res['boxes_xyxy'])))
        for f in feats:
            self.assertEqual(f.shape, (32,))
            self.assertAlmostEqual(float(np.linalg.norm(f)), 1.0, places=4)

    def test_2_real_checkpoints_schema_if_present(self):
        """Verify 100% strict key match (zero unexpected/missing) for ALL real archived v1, v2, v3.1 checkpoints."""
        checks = [
            ('checkpoints/ov_dquo_v1_vitb16_mm_ep4_fullval13.32.pth', OVDQUOV1Detector, 'vitb16', 768, 'load_tuned_v1'),
            ('checkpoints/ov_dquo_v2_vitl14_mm_ep12_fullval10.85.pth', OVDQUOV2Detector, 'vitl14', 672, 'load_tuned_v2'),
            ('checkpoints/ov_dquo_v31_vitl14_mm_ep8_fullval11.08.pth', OVDQUOV31Detector, 'vitl14', 672, 'load_tuned_v31'),
        ]
        for path_str, cls, variant, res, loader_name in checks:
            ckpt_path = Path(path_str)
            if not ckpt_path.exists():
                continue
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            self.assertEqual(ckpt.get('architecture'), cls.ARCHITECTURE)
            model = cls(variant=variant, device='cpu', resolution=res, use_exemplars=True)
            getattr(model, loader_name)(str(ckpt_path))

    def test_2b_tiny_v2_and_v31_detect_and_features(self):
        """Verify tiny v2 & v3.1 detectors run detect_with_features + extract_box_features cleanly."""
        tiny_cfg = dict(
            img_size_pretrain=64, patch_size=16, embed_dim=48, depth=4, num_heads=4,
            mlp_ratio=2.0, text_dim=32, pt_hw_seq_len=4, out_indices=(1, 2, 3),
            default_resolution=128, hidden_dim=32, nheads=4, dim_feedforward=64,
            enc_layers=2, dec_layers=3,
        )
        img = Image.fromarray(np.random.default_rng(7).integers(0, 255, (100, 160, 3), dtype=np.uint8))
        for cls in (OVDQUOV2Detector, OVDQUOV31Detector):
            det = cls(variant='vitb16', device='cpu', resolution=128, num_queries=16,
                      use_exemplars=True, text_encoder_fn=_mock_text_encoder, _tiny_cfg=tiny_cfg)
            res = det.detect_with_features(img, 'penguin', threshold=-1.0, tile_trigger=0)
            self.assertGreater(len(res['boxes_xyxy']), 0)
            feats = cls.extract_box_features(res['dense_map'], res['dense_mask'], res['boxes_xyxy'][:3], img.size)
            self.assertEqual(len(feats), min(3, len(res['boxes_xyxy'])))
            for f in feats:
                self.assertAlmostEqual(float(np.linalg.norm(f)), 1.0, places=4)

    def test_3_end_to_end_4stage_pipeline_filters_fp_and_tracks_ids(self):
        frames = [Image.fromarray(np.full((96, 128, 3), 120 + i, dtype=np.uint8)) for i in range(6)]
        mock_det = _MockDetectorForPipeline()
        mock_sam = (_MockSAM2VideoPredictor(96, 128, 6), _MockSAM2ImagePredictor())
        with tempfile.TemporaryDirectory() as tmp:
            out = count_video_ov_dquo_sam2(
                detector=mock_det,
                frames=frames,
                text='penguin',
                device='cpu',
                threshold=0.26,
                tile_trigger=0,
                window=3,
                batch_size=8,
                box_chunk=2,  # explicitly exercise chunking!
                sam_predictors=mock_sam,
                identity_config=IdentityConfig(),
                preview_dir=Path(tmp),
                preview_prefix='test_vid',
            )
            # Frame 2 had 3 proposals (1 transient FP); Stage 2 must drop it -> exactly 2 persistent tracked IDs!
            self.assertEqual(out['proposal_counts'][2], 3)
            self.assertEqual(out['filtered_counts'][2], 2)
            self.assertEqual(out['count'], 2)
            self.assertEqual(out['stitched_count'], 2)
            self.assertEqual(out['frame_counts'], [2, 2, 2, 2, 2, 2])
            self.assertEqual(len(out['preview_files']), 3)
            for pf in out['preview_files']:
                self.assertTrue(Path(pf).is_file())


if __name__ == '__main__':
    unittest.main()
