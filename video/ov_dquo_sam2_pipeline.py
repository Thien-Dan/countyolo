"""Complete 4-Stage OV-DQUO v1 + SAM 2.1 + Long-Term Tracking + Temporal Identity Video Counting Pipeline.

Stages:
1. Per-frame OV-DQUO v1 open-world detection (`threshold=0.26`, optional 2x2 dense tiling) + chunked SAM2
   independent mask segmentation (`dedup_iou=0.85`) + zero-extra-cost EVA02-CLIP `dense_map` caching.
2. Short-term SAM2 temporal persistence filter (`window=3`, IoU > 0.5 across >= 3 consecutive frames).
3. Long-term SAM2 forward propagation & uncovered-object discovery (global track IDs `1..N`, CountVid protocol).
4. Conservative offline `TemporalIdentity` tracklet stitching (`video/identity.py`) using EVA02-CLIP `dense_map`
   appearance embeddings + velocity/boundary re-entry consistency (recovers IDs fragmented by occlusion/re-entry).
"""
from contextlib import nullcontext
import hashlib
from pathlib import Path
import tempfile
import time
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
import torch

from video.count_sam2 import (
    SAM2_CHECKPOINT,
    SAM2_CONFIG,
    mask_indices,
    mask_iou,
    temporal_filter,
    track_long_term,
)
from video.identity import IdentityConfig, TemporalIdentity
from models.ov_dquo_v1_infer import OVDQUOV1Detector

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def independent_masks_chunked(image_predictor, image: Image.Image, boxes: np.ndarray,
                              dedup_iou: float = 0.85, box_chunk: int = 64) -> list[dict]:
    """Memory-safe chunked SAM2 box-prompted segmentation with mask deduplication (bit-identical to `count_sam2`)."""
    if len(boxes) == 0:
        return []
    image_predictor.set_image(np.array(image))
    all_masks, all_scores = [], []
    for start in range(0, len(boxes), box_chunk):
        sub_boxes = boxes[start:start + box_chunk]
        masks, scores, _ = image_predictor.predict(box=sub_boxes, multimask_output=False)
        all_masks.append(np.asarray(masks).reshape(len(sub_boxes), image.height, image.width))
        all_scores.append(np.asarray(scores).reshape(len(sub_boxes), -1).max(axis=1))
    masks = np.concatenate(all_masks, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    kept = []
    for j in np.argsort(-scores, kind='stable'):
        pixels = mask_indices(masks[j])
        if not pixels.size:
            continue
        if any(mask_iou(pixels, obj['mask']) >= dedup_iou for obj in kept):
            continue
        kept.append(dict(box=np.asarray(boxes[j], dtype=np.float32), mask=pixels))
    return kept


def mask_pixels_to_xyxy(flat_pixels: np.ndarray, width: int) -> np.ndarray:
    """Convert flattened pixel indices to tight `[x1, y1, x2, y2]` box."""
    ys = flat_pixels // int(width)
    xs = flat_pixels % int(width)
    return np.array([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=np.float64)


def render_tracked_preview(image: Image.Image, frame_objects: list[tuple[int, np.ndarray]], title: str) -> Image.Image:
    """Draw tracked boxes and stitched IDs on a frame copy for visual inspection."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for obj_id, box in frame_objects:
        r = (obj_id * 67) % 200 + 55
        g = (obj_id * 131) % 200 + 55
        b = (obj_id * 197) % 200 + 55
        color = (int(r), int(g), int(b))
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, max(0.0, y1 - 11.0)), str(obj_id), fill=color)
    draw.rectangle([4, 4, min(image.width - 4, 460), 26], fill=(0, 0, 0))
    draw.text((10, 8), title, fill=(255, 255, 255))
    return canvas


def build_sam2_predictors(sam_checkpoint, device):
    if not Path(sam_checkpoint).is_file():
        raise FileNotFoundError(f'Missing SAM2 checkpoint: {sam_checkpoint}')
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = build_sam2_video_predictor(SAM2_CONFIG, str(sam_checkpoint), device=device)
    image_predictor = SAM2ImagePredictor(predictor)
    return predictor, image_predictor


@torch.inference_mode()
def count_video_ov_dquo_sam2(
    detector: OVDQUOV1Detector,
    frames: list,
    text: str,
    device: str = 'cuda',
    threshold: float = 0.26,
    tile_trigger: int = 650,
    window: int = 3,
    batch_size: int = 30,
    box_chunk: int = 64,
    dedup_iou: float = 0.85,
    sam_checkpoint=SAM2_CHECKPOINT,
    sam_predictors: Optional[tuple] = None,
    identity_config: Optional[IdentityConfig] = None,
    preview_dir: Optional[Path] = None,
    preview_prefix: str = '',
    progress=None,
) -> dict:
    """Run complete 4-stage OV-DQUO v1 + SAM2 + Tracking + TemporalIdentity video counting."""
    if not frames:
        raise ValueError('No frames provided')
    if window < 2 or batch_size < 1:
        raise ValueError('window must be >= 2 and batch_size >= 1')
    if len(frames) < window:
        raise ValueError(f'Need at least {window} frames for short-term temporal filter')

    dev = torch.device(device)
    started = time.perf_counter()
    stage_times = {}

    if sam_predictors is not None:
        predictor, image_predictor = sam_predictors
    else:
        predictor, image_predictor = build_sam2_predictors(sam_checkpoint, dev)

    if dev.type == 'cuda':
        if torch.cuda.is_bf16_supported():
            sam_dtype = torch.bfloat16
            sam_prec_name = 'bfloat16'
        else:
            # T4 Turing Tensor Cores: float16 gives ~2.2x faster SAM2 propagation than FP32
            sam_dtype = torch.float16
            sam_prec_name = 'float16'
        sam_ctx = torch.autocast('cuda', dtype=sam_dtype)
    else:
        sam_prec_name = 'float32'
        sam_ctx = nullcontext()

    with tempfile.TemporaryDirectory(prefix='ovdquo_sam2_') as folder, sam_ctx:
        images = []
        for i, f in enumerate(frames):
            img = (Image.open(f) if isinstance(f, (str, Path)) else f).convert('RGB')
            if images and img.size != images[0].size:
                raise ValueError('All frames must share identical dimensions')
            images.append(img)
            img.save(Path(folder) / f'{i:06d}.jpg', quality=95)

        img_w, img_h = images[0].size
        independent = []
        proposal_counts = []
        tiled_frames = 0
        dense_cache = []

        # Stage 1: OV-DQUO v1 proposals + chunked SAM2 independent segmentation + EVA02-CLIP feature caching
        t0 = time.perf_counter()
        for i, image in enumerate(images):
            with torch.autocast(dev.type, enabled=False):
                det = detector.detect_with_features(image, text, threshold=threshold, tile_trigger=tile_trigger)
            boxes_px = det['boxes_xyxy']
            proposal_counts.append(int(len(boxes_px)))
            if det['tiled']:
                tiled_frames += 1
            dense_cache.append((det['dense_map'], det['dense_mask']))
            independent.append(
                independent_masks_chunked(image_predictor, image, boxes_px, dedup_iou=dedup_iou, box_chunk=box_chunk)
            )
            if progress:
                progress('Stage 1: OV-DQUO v1 + SAM2 masks', i + 1, len(images))
        stage_times['stage1_detect_and_segment'] = time.perf_counter() - t0
        image_predictor.reset_predictor()

        # Stage 2: Short-term SAM2 persistence filter (keep state on GPU for <= 120 frames to avoid PCIe stalls!)
        offload_state = len(images) > 120
        state = predictor.init_state(video_path=folder, offload_video_to_cpu=True, offload_state_to_cpu=offload_state)
        t0 = time.perf_counter()
        filtered = temporal_filter(predictor, state, independent, window=window, batch_size=batch_size, progress=progress)
        stage_times['stage2_temporal_filter'] = time.perf_counter() - t0

        # Stage 3: Long-term SAM2 tracking & unique track ID assignment
        t0 = time.perf_counter()
        birth_trace = []
        tracks = track_long_term(predictor, state, filtered, batch_size=batch_size, progress=progress, trace=birth_trace)
        stage_times['stage3_long_term_tracking'] = time.perf_counter() - t0
        predictor.reset_state(state)

        # Stage 4: Conservative offline TemporalIdentity stitching using cached EVA02-CLIP dense_map features
        t0 = time.perf_counter()
        identity = TemporalIdentity(identity_config or IdentityConfig())
        frame_counts = []
        frame_visible_boxes = {}
        for i in range(len(images)):
            raw_ids, boxes_list = [], []
            for raw_id in sorted(tracks):
                traj = tracks[raw_id]
                if i in traj and traj[i].size > 0:
                    raw_ids.append(int(raw_id))
                    boxes_list.append(mask_pixels_to_xyxy(traj[i], img_w))
            frame_counts.append(len(raw_ids))
            if raw_ids:
                boxes_arr = np.stack(boxes_list, axis=0)
                dm, dmask = dense_cache[i]
                feats = OVDQUOV1Detector.extract_box_features(dm, dmask, boxes_arr, (img_w, img_h))
                identity.observe(i, raw_ids, boxes_arr, feats, (img_h, img_w))
                frame_visible_boxes[i] = list(zip(raw_ids, boxes_arr))
            else:
                identity.observe(i, [], [], [], (img_h, img_w))
                frame_visible_boxes[i] = []
        id_result = identity.finalize(min_hits=1)
        stage_times['stage4_temporal_identity'] = time.perf_counter() - t0

        # Optional preview frames (first, middle, last) with stitched global IDs drawn
        preview_files = []
        if preview_dir is not None:
            preview_dir = Path(preview_dir)
            preview_dir.mkdir(parents=True, exist_ok=True)
            mapping = id_result['raw_to_identity']
            sample_idxs = sorted({0, len(images) // 2, len(images) - 1})
            for idx_p in sample_idxs:
                objs = [(mapping.get(rid, rid), box) for rid, box in frame_visible_boxes[idx_p]]
                title = (
                    f'{preview_prefix} f={idx_p} "{text}" | visible={len(objs)} '
                    f'| sam2={len(tracks)} stitched={id_result["count"]}'
                )
                out_img = render_tracked_preview(images[idx_p], objs, title)
                fname = f'{preview_prefix}_f{idx_p:04d}.jpg' if preview_prefix else f'f{idx_p:04d}.jpg'
                out_path = preview_dir / fname
                out_img.save(out_path, quality=90)
                preview_files.append(str(out_path))

    if dev.type == 'cuda':
        torch.cuda.synchronize()
    total_sec = time.perf_counter() - started
    return dict(
        count=int(len(tracks)),
        stitched_count=int(id_result['count']),
        identity_links_count=int(len(id_result['links'])),
        identity_links=id_result['links'],
        identity_uncertain_count=int(len(id_result['uncertain'])),
        raw_to_identity={str(k): int(v) for k, v in id_result['raw_to_identity'].items()},
        identity_hits={str(k): int(v) for k, v in id_result['identity_hits'].items()},
        frame_counts=frame_counts,
        frames=len(frames),
        sec=total_sec,
        fps=float(len(frames)) / total_sec if total_sec > 0 else 0.0,
        stage_times=stage_times,
        proposal_counts=proposal_counts,
        independent_counts=[len(x) for x in independent],
        filtered_counts=[len(x) for x in filtered],
        tiled_frames=tiled_frames,
        birth_trace=birth_trace,
        preview_files=preview_files,
        threshold=float(threshold),
        tile_trigger=int(tile_trigger),
        window=int(window),
        batch_size=int(batch_size),
        dedup_iou=float(dedup_iou),
        tracker='OV-DQUO v1 + SAM2 (3 stages) + TemporalIdentity',
        protocol='ov-dquo-v1-sam2-identity-v1',
        sam2_config=SAM2_CONFIG,
        sam2_precision=sam_prec_name,
        pipeline_code_sha256=IMPLEMENTATION_SHA256,
    )
