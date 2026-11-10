"""
Engine đếm bằng SAM 3 (Meta, qua Ultralytics): text -> tìm + tách mask mọi đối tượng của khái niệm đó.
  - Ảnh : số đếm = số đối tượng (conf >= ngưỡng). SAM 3 chỉ có 200 query/lượt, nên ảnh có số đối tượng gần
          mức trần được chia lưới g x g ô chồng lấn, đếm từng ô (phóng lên 1008px) rồi gộp box bằng NMS.
  - Video: SAM3VideoSemanticPredictor vừa phát hiện vừa theo dõi -> số đếm = số ID track khác nhau.

Trọng số `sam3.pt` phải tải từ https://huggingface.co/facebook/sam3 (cần được Meta duyệt quyền truy cập).
"""
import os
import collections
import shutil
import tempfile
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

QUERY_CAP = 200          # số query của decoder SAM 3 (build_sam3: num_queries=200)
TILE_TRIGGER = 0.8       # chia ô khi số đối tượng >= 80% mức trần
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# weights/sam3.pt hoặc sam3.pt ở thư mục gốc dự án
DEFAULT_WEIGHTS = next((p for p in (os.path.join(_ROOT, "weights", "sam3.pt"), os.path.join(_ROOT, "sam3.pt"))
                        if os.path.exists(p)), os.path.join(_ROOT, "weights", "sam3.pt"))


def _windows(W, H, g, overlap=0.15):
    tw, th = min(W, W / g * (1 + overlap)), min(H, H / g * (1 + overlap))
    xs = [0.0] if g == 1 else [i * (W - tw) / (g - 1) for i in range(g)]
    ys = [0.0] if g == 1 else [i * (H - th) / (g - 1) for i in range(g)]
    return [(round(x), round(y), round(x + tw), round(y + th)) for y in ys for x in xs]


class SAM3Counter:
    def __init__(self, weights=DEFAULT_WEIGHTS, conf=0.3, half=True, max_grid=4):
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"Không thấy {weights}. Tải sam3.pt từ https://huggingface.co/facebook/sam3 (cần quyền truy cập) "
                "và đặt vào thư mục weights/.")
        self.weights, self.conf, self.half, self.max_grid = weights, conf, half, max_grid
        self._img = None

    def _overrides(self, **kw):
        return dict(conf=self.conf, task="segment", mode="predict", model=self.weights,
                    quantize=16 if self.half else 32, save=False, verbose=False, **kw)

    def _image_predictor(self):
        if self._img is None:
            from ultralytics.models.sam import SAM3SemanticPredictor
            # imgsz 1008 = độ phân giải SAM 3 được train; mặc định 640 của Ultralytics làm đếm thiếu nặng
            # (FSC-147: táo 46 -> 81/82, kính râm 2 -> 150/253 ở conf 0.3)
            class CheckedSemanticPredictor(SAM3SemanticPredictor):
                def inference(inner, *args, **kwargs):
                    out = super().inference(*args, **kwargs)
                    inner.last_numerics = {}
                    for key in ("pred_logits", "presence_logit_dec", "pred_boxes"):
                        tensor = out[key]
                        finite = bool(torch.isfinite(tensor).all())
                        inner.last_numerics[key] = dict(finite=finite,
                            min=float(tensor.min()), max=float(tensor.max()))
                        if not finite:
                            raise FloatingPointError(f"SAM3 produced non-finite {key}; check precision/checkpoint")
                    return out
            self._img = CheckedSemanticPredictor(overrides=self._overrides(imgsz=1008))
        return self._img

    def _detect(self, rgb, text):
        """Một lượt SAM 3 trên ảnh RGB (H, W, 3) -> box xyxy (N, 4), score (N,)."""
        pred = self._image_predictor()
        pred.set_image(rgb[..., ::-1].copy())  # Ultralytics nhận BGR như cv2
        r = pred(text=[text])[0]
        if r.boxes is None or len(r.boxes) == 0:
            return torch.zeros(0, 4), torch.zeros(0)
        return r.boxes.xyxy.float().cpu(), r.boxes.conf.float().cpu()

    @torch.no_grad()
    def count_image(self, image, text):
        """image: PIL.Image / đường dẫn / mảng RGB. Trả về {"count", "boxes", "scores", "grid", "ms"}."""
        if isinstance(image, str):
            image = Image.open(image)
        rgb = np.array(image.convert("RGB")) if isinstance(image, Image.Image) else np.asarray(image)
        H, W = rgb.shape[:2]
        t0 = time.perf_counter()
        boxes, scores = self._detect(rgb, text)
        g = 1
        # Gần chạm trần 200 query -> có thể còn đối tượng bị bỏ sót: chia ô dày dần tới khi mỗi ô còn dư chỗ
        while len(boxes) >= TILE_TRIGGER * QUERY_CAP * g * g and g < self.max_grid:
            g += 1
            all_b, all_s = [], []
            for x0, y0, x1, y1 in _windows(W, H, g):
                b, s = self._detect(rgb[y0:y1, x0:x1], text)
                b = b + torch.tensor([x0, y0, x0, y0], dtype=b.dtype)
                # bỏ box chạm cạnh ô nằm trong ảnh (vật bị cắt; ô bên cạnh thấy trọn vật)
                m = 2.0
                keep = torch.ones(len(b), dtype=torch.bool)
                if x0 > 0: keep &= b[:, 0] > x0 + m
                if y0 > 0: keep &= b[:, 1] > y0 + m
                if x1 < W: keep &= b[:, 2] < x1 - m
                if y1 < H: keep &= b[:, 3] < y1 - m
                all_b.append(b[keep])
                all_s.append(s[keep])
            boxes, scores = torch.cat(all_b), torch.cat(all_s)
            if len(boxes):
                k = nms(boxes, scores, 0.5)
                boxes, scores = boxes[k], scores[k]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {"count": int(len(boxes)), "boxes": boxes.numpy(), "scores": scores.numpy(), "grid": g,
                "ms": (time.perf_counter() - t0) * 1000,
                "numerics": getattr(self._img, "last_numerics", {})}

    @torch.no_grad()
    def count_video(self, frames, text, fps=3, on_frame=None, min_hits=1, trace_path=None, detection_conf=0.5, tracker_profile="legacy", object_batch_size=0):
        """Count real track IDs; report raw and persistence-filtered counts separately.

        Frame lists are streamed to lossless FFV1, avoiding lossy MP4 conversion
        and loading an entire video into RAM. Missing IDs fail explicitly.
        """
        from ultralytics.models.sam import SAM3VideoSemanticPredictor
        from video.sam3_policy import META_ALIGNED, DelayedTrackFrames, add_object_batches
        if object_batch_size < 0:
            raise ValueError("object_batch_size must be nonnegative")
        if tracker_profile not in ("legacy", "aligned"):
            raise ValueError(f"Unknown SAM3 tracker profile: {tracker_profile}")
        # Lifecycle fields are an internal API: pin the audited implementation.
        if tracker_profile == "aligned":
            import ultralytics
            if ultralytics.__version__ != "8.4.204":
                raise RuntimeError("Aligned SAM3 requires audited ultralytics==8.4.204")
        class LifecyclePredictor(SAM3VideoSemanticPredictor):
            def _tracker_add_new_objects(inner, frame_idx, num_frames, new_obj_ids, new_obj_masks, tracker_states_local):
                return add_object_batches(super()._tracker_add_new_objects, frame_idx, num_frames,
                                          new_obj_ids, new_obj_masks, tracker_states_local, object_batch_size)

            def setup_model(inner, *args, **kwargs):
                # Upstream uses strict=False when mapping Meta checkpoint keys.
                # Audit every required parameter instead of silently using random weights.
                from unittest.mock import patch
                original_load = torch.nn.Module.load_state_dict
                inner.weight_load_reports = []
                def audited_load(module, *load_args, **load_kwargs):
                    incompatible = original_load(module, *load_args, **load_kwargs)
                    missing_params = sorted(set(incompatible.missing_keys) & dict(module.named_parameters()).keys())
                    report = dict(model=type(module).__name__, missing_parameters=missing_params,
                                  missing_buffers=sorted(set(incompatible.missing_keys)-set(missing_params)),
                                  unexpected_count=len(incompatible.unexpected_keys))
                    inner.weight_load_reports.append(report)
                    if missing_params:
                        raise RuntimeError(f"Incomplete SAM3 checkpoint mapping: {missing_params[:12]}")
                    return incompatible
                with patch.object(torch.nn.Module, "load_state_dict", audited_load):
                    super().setup_model(*args, **kwargs)
                print("SAM3 checkpoint audit:", inner.weight_load_reports, flush=True)

            def postprocess(inner, preds, img, orig_imgs):
                inner.removed_ids = tuple(preds["removed_obj_ids"])
                results = super().postprocess(preds, img, orig_imgs)
                # Upstream builds boxes before enforcing mask exclusivity.
                # Remove masks that became empty, then rebuild their geometry.
                from ultralytics.models.sam.amg import batched_mask_to_box
                for j, result in enumerate(results):
                    if result.masks is not None:
                        keep = result.masks.data.bool().flatten(1).any(1)
                        result = result[keep]
                        if len(result):
                            result.boxes.data[:, :4] = batched_mask_to_box(result.masks.data.bool())
                        results[j] = result
                return results
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        t0 = time.perf_counter()
        src, tmp_dir = frames, None
        trace = None
        writer = None
        try:
            if not isinstance(frames, str):
                if not len(frames):
                    raise ValueError("Video has no frames")
                tmp_dir = tempfile.mkdtemp(prefix="sam3_video_")
                src = os.path.join(tmp_dir, "frames.avi")
                for i, f in enumerate(frames):
                    if isinstance(f, str):
                        with Image.open(f) as im:
                            rgb = np.array(im.convert("RGB"))
                    else:
                        rgb = np.array(f.convert("RGB"))
                    if i == 0:
                        h, w = rgb.shape[:2]
                        writer = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError("FFV1 encoder unavailable; refusing lossy evaluation")
                    if rgb.shape[:2] != (h, w):
                        raise ValueError("Frame dimensions changed within one video")
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                writer.release()
                writer = None
            encode_sec = time.perf_counter() - t0
            predictor_cls = LifecyclePredictor if tracker_profile == "aligned" else SAM3VideoSemanticPredictor
            settings = dict(META_ALIGNED) if tracker_profile == "aligned" else {}
            pred = predictor_cls(overrides=self._overrides(imgsz=1008),
                                 score_threshold_detection=detection_conf, **settings)
            delayed = DelayedTrackFrames(settings.get("hotstart_delay", 0))
            hits, raw_hits, frame_counts = collections.Counter(), collections.Counter(), []
            def emit(record):
                ids, xyxy = record["ids"], record["boxes"]
                hits.update(set(ids))
                frame_counts.append(len(ids))
                if trace:
                    trace.write(json.dumps({k: v for k, v in record.items() if k != "image"}) + "\n")
                    trace.flush()
                if on_frame is not None:
                    on_frame(record["frame"], record["image"], np.asarray(xyxy).reshape(-1, 4), ids, len(ids))
            if trace_path:
                os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
                trace = open(trace_path, "w", encoding="utf-8")
            import json
            for i, r in enumerate(pred(source=src, text=[text], stream=True)):
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    ids, xyxy = [], np.zeros((0, 4))
                else:
                    if not boxes.is_track or boxes.id is None:
                        raise RuntimeError("SAM 3 returned detections without persistent track IDs")
                    ids = boxes.id.int().cpu().tolist()
                    xyxy = boxes.xyxy.float().cpu().numpy()
                raw_hits.update(set(ids))
                record = dict(frame=i, ids=ids, boxes=xyxy.tolist())
                if on_frame is not None:
                    record["image"] = r.orig_img[..., ::-1].copy()
                for ready in delayed.push(record, getattr(pred, "removed_ids", ())):
                    emit(ready)
            for ready in delayed.finish():
                emit(ready)
            if not frame_counts:
                raise RuntimeError("SAM 3 processed no frames")
            if not isinstance(frames, str) and len(frame_counts) != len(frames):
                raise RuntimeError(f"Frame count mismatch: {len(frame_counts)} != {len(frames)}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return {"count": sum(n >= min_hits for n in hits.values()), "raw_count": len(raw_hits), "tracker_profile": tracker_profile,
                    "tracker_settings": settings, "precision": "fp16" if self.half else "fp32",
                    "hotstart_delay": delayed.delay, "object_batch_size": object_batch_size, "weight_load_reports": getattr(pred, "weight_load_reports", []),
                    "min_hits": min_hits, "detection_conf": detection_conf, "track_hits": dict(hits), "frame_counts": frame_counts,
                    "frames": len(frame_counts), "encode_sec": encode_sec,
                    "sec": time.perf_counter() - t0}
        finally:
            if writer is not None:
                writer.release()
            if trace:
                trace.close()
            if tmp_dir:
                shutil.rmtree(tmp_dir)
