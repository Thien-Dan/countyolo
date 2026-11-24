"""CountYOLO + SAM2, following CountVid's three-stage video protocol.

Reference: niki-amini-naieni/CountVid, commit
690ca3e48369c35f5e23a3ab5d2b04b56881a343 (MIT, Niki Amini-Naieni).
Stage 1: independent box-prompted masks. Stage 2: consecutive mask-IoU
persistence. Stage 3: forward mask propagation and uncovered-object discovery.
Unlike the upstream script, this module is reentrant and has no CountGD imports
or multiprocessing globals. It also removes duplicate independent masks before
tracking, skips empty masks, and handles initially empty videos/batched births.
"""
from contextlib import nullcontext
import hashlib
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SAM2_CHECKPOINT = ROOT / "checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def mask_indices(mask):
    return np.flatnonzero(np.asarray(mask).reshape(-1)).astype(np.int32)


def mask_iou(a, b):
    intersection = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - intersection
    return intersection / union if union else 0.0


def independent_masks(image_predictor, image, boxes, dedup_iou=0.85):
    """One independent SAM2 mask per proposal, removing empty/duplicate masks."""
    if not len(boxes):
        return []
    image_predictor.set_image(np.array(image))
    masks, scores, _ = image_predictor.predict(box=boxes, multimask_output=False)
    masks = np.asarray(masks).reshape(len(boxes), image.height, image.width)
    scores = np.asarray(scores).reshape(len(boxes), -1).max(axis=1)
    kept = []
    for j in np.argsort(-scores, kind="stable"):
        pixels = mask_indices(masks[j])
        if not pixels.size:
            continue
        if any(mask_iou(pixels, obj["mask"]) >= dedup_iou for obj in kept):
            continue
        kept.append(dict(box=np.asarray(boxes[j], dtype=np.float32), mask=pixels))
    return kept


def propagate(predictor, state, frame, seeds, reverse=False, steps=None):
    """Reset a bounded object batch, seed it, then yield masks with global IDs."""
    predictor.reset_state(state)
    for obj_id, box in seeds:
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=frame, obj_id=obj_id, box=box,
        )
    kwargs = dict(start_frame_idx=frame, reverse=reverse)
    if steps is not None:
        kwargs["max_frame_num_to_track"] = steps
    for index, ids, logits in predictor.propagate_in_video(state, **kwargs):
        masks = {int(obj_id): mask_indices((logits[j] > 0).cpu().numpy())
                 for j, obj_id in enumerate(ids)}
        yield int(index), masks


def persistent(seed_frame, obj_id, tracked, independent, window):
    """CountVid: >=window consecutive independent matches, IoU strictly >0.5."""
    support = 1
    for direction in (-1, 1):
        for offset in range(1, window):
            frame = seed_frame + direction * offset
            if frame < 0 or frame >= len(independent):
                break
            mask = tracked.get(frame, {}).get(obj_id, np.empty(0, dtype=np.int32))
            if not any(mask_iou(mask, obj["mask"]) > 0.5 for obj in independent[frame]):
                break
            support += 1
    return support >= window


def temporal_filter(predictor, state, independent, window=3, batch_size=8, progress=None):
    kept = []
    for frame, objects in enumerate(independent):
        accepted = []
        for start in range(0, len(objects), batch_size):
            batch = objects[start:start + batch_size]
            seeds = [(j + 1, obj["box"]) for j, obj in enumerate(batch)]
            tracked = {}
            for reverse in (True, False):
                for index, masks in propagate(predictor, state, frame, seeds, reverse, window - 1):
                    tracked[index] = masks
            accepted.extend(obj for j, obj in enumerate(batch)
                            if persistent(frame, j + 1, tracked, independent, window))
        kept.append(accepted)
        if progress:
            progress("Lọc ngắn hạn SAM2", frame + 1, len(independent))
    return kept


def discover(objects, covered, min_area=1):
    """CountVid new-object rule: nonempty mask, zero overlap with tracked union.

    Extend coverage within this frame too, so duplicate births cannot enter the
    same batch. IDs are assigned only after a candidate passes this check.
    """
    accepted = []
    for obj in objects:
        pixels = obj["mask"]
        if pixels.size < min_area or np.intersect1d(pixels, covered, assume_unique=True).size:
            continue
        accepted.append(obj)
        covered = np.union1d(covered, pixels)
    return accepted


def track_long_term(predictor, state, independent, batch_size=8, progress=None, trace=None):
    tracks = {}
    next_id = 1
    for frame, objects in enumerate(independent):
        present = [trajectory[frame] for trajectory in tracks.values() if frame in trajectory]
        covered = np.unique(np.concatenate(present)) if present else np.empty(0, dtype=np.int32)
        births = discover(objects, covered)
        first_id = next_id
        for start in range(0, len(births), batch_size):
            batch = births[start:start + batch_size]
            seeds = [(next_id + j, obj["box"]) for j, obj in enumerate(batch)]
            next_id += len(batch)
            for index, masks in propagate(predictor, state, frame, seeds):
                for obj_id, pixels in masks.items():
                    tracks.setdefault(obj_id, {})[index] = pixels
        if trace is not None:
            trace.append(dict(frame=frame, new_ids=list(range(first_id, next_id)),
                              cumulative_count=next_id - 1))
        if progress:
            progress("Theo dõi dài hạn SAM2", frame + 1, len(independent))
    return tracks


@torch.inference_mode()
def count_video(model, frames, text, device, min_mass=0.2, window=3, batch_size=8,
                sam_checkpoint=SAM2_CHECKPOINT, on_frame=None, progress=None, proposal_fn=None):
    if not frames:
        raise ValueError("Không có frame để đếm")
    if window < 2 or batch_size < 1:
        raise ValueError("window phải >=2 và batch_size phải >=1")
    if len(frames) < window:
        raise ValueError(f"Cần ít nhất {window} frame cho bộ lọc ngắn hạn")
    if not Path(sam_checkpoint).is_file():
        raise FileNotFoundError("Thiếu SAM2. Chạy: python tools/download_checkpoints.py --group countvid")
    try:
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise RuntimeError("Chưa cài SAM2. Cài requirements-video.txt vào môi trường hiện có.") from exc
    from utils.scale_zoom import density_tiled_count

    device = torch.device(device)
    started = time.perf_counter()
    stage_times = {}
    predictor = build_sam2_video_predictor(SAM2_CONFIG, str(sam_checkpoint), device=device)
    # Image/video predictors share the same SAM2 model rather than loading twice.
    image_predictor = SAM2ImagePredictor(predictor)
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()
    with tempfile.TemporaryDirectory(prefix="countyolo_sam2_") as folder, context:
        images = []
        for i, f in enumerate(frames):
            image = (Image.open(f) if isinstance(f, (str, Path)) else f).convert("RGB")
            if images and image.size != images[0].size:
                raise ValueError("Các frame phải có cùng kích thước")
            images.append(image)
            image.save(Path(folder) / f"{i:06d}.jpg", quality=95)
        independent, proposal_counts, proposal_trace = [], [], []
        t0 = time.perf_counter()
        for i, image in enumerate(images):
            # CountYOLO remains in its trained precision; autocast is for SAM2.
            with torch.autocast(device.type, enabled=False):
                frame_started = time.perf_counter()
                if proposal_fn is None:
                    density_count, info, boxes, _ = density_tiled_count(
                        model, image, text, device, count_thr=0, return_boxes=True, min_mass=min_mass,
                    )
                else:
                    density_count, info, boxes, _ = proposal_fn(image, text)
                proposal_trace.append(dict(frame=i, density_count=float(density_count),
                    proposal_sec=time.perf_counter()-frame_started, exemplar=info.get('exemplar')))
            proposal_counts.append(len(boxes))
            independent.append(independent_masks(image_predictor, image, boxes.cpu().numpy()))
            if progress:
                progress("Proposal CountYOLO + phân đoạn SAM2", i + 1, len(images))
        stage_times["independent"] = time.perf_counter() - t0
        image_predictor.reset_predictor()
        state = predictor.init_state(video_path=folder, offload_video_to_cpu=True, offload_state_to_cpu=True)
        t0 = time.perf_counter()
        filtered = temporal_filter(predictor, state, independent, window, batch_size, progress)
        stage_times["short_term"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        birth_trace = []
        tracks = track_long_term(predictor, state, filtered, batch_size, progress, trace=birth_trace)
        stage_times["long_term"] = time.perf_counter() - t0
        frame_counts = []
        for i, image in enumerate(images):
            visible = {k: trajectory[i] for k, trajectory in tracks.items()
                       if i in trajectory and trajectory[i].size}
            frame_counts.append(len(visible))
            if on_frame:
                on_frame(i, image, visible, len(visible))
        predictor.reset_state(state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return dict(count=len(tracks), frame_counts=frame_counts, frames=len(frames),
                sec=time.perf_counter() - started, stage_times=stage_times,
                proposal_counts=proposal_counts, independent_counts=[len(x) for x in independent],
                filtered_counts=[len(x) for x in filtered], tracker="CountVid SAM2 (3 stages)",
                proposal_trace=proposal_trace, birth_trace=birth_trace,
                window=window, batch_size=batch_size, mask_dedup_iou=0.85,
                protocol="countyolo-sam2-ui-v1", sam2_config=SAM2_CONFIG,
                sam2_precision="bfloat16" if use_bfloat16 else "float32",
                pipeline_code_sha256=IMPLEMENTATION_SHA256, torch_version=torch.__version__)
