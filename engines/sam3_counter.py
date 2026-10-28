"""
Engine đếm bằng SAM 3 (Meta, qua Ultralytics): text -> tìm + tách mask mọi đối tượng của khái niệm đó.
  - Ảnh : số đếm = số đối tượng (conf >= ngưỡng). SAM 3 chỉ có 200 query/lượt, nên ảnh có số đối tượng gần
          mức trần được chia lưới g x g ô chồng lấn, đếm từng ô (phóng lên 1008px) rồi gộp box bằng NMS.
  - Video: SAM3VideoSemanticPredictor vừa phát hiện vừa theo dõi -> số đếm = số ID track khác nhau.

Trọng số `sam3.pt` phải tải từ https://huggingface.co/facebook/sam3 (cần được Meta duyệt quyền truy cập).
"""
import os
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
            self._img = SAM3SemanticPredictor(overrides=self._overrides(imgsz=1008))
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
                "ms": (time.perf_counter() - t0) * 1000}

    @torch.no_grad()
    def count_video(self, frames, text, fps=3, on_frame=None):
        """frames: list PIL.Image / đường dẫn ảnh, hoặc đường dẫn file video. Trả về {"count", "frame_counts",
        "frames", "sec"}; on_frame(i, image_rgb, boxes, ids, n) sau mỗi frame (dùng cho UI)."""
        from ultralytics.models.sam import SAM3VideoSemanticPredictor
        src, tmp = frames, None
        if not isinstance(frames, str):  # ghép frame thành mp4 tạm cho predictor video
            imgs = [np.array((Image.open(f) if isinstance(f, str) else f).convert("RGB")) for f in frames]
            h, w = imgs[0].shape[:2]
            tmp = os.path.join(tempfile.mkdtemp(), "frames.mp4")
            vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for im in imgs:
                vw.write(cv2.cvtColor(cv2.resize(im, (w, h)), cv2.COLOR_RGB2BGR))
            vw.release()
            src = tmp
        pred = SAM3VideoSemanticPredictor(overrides=self._overrides(imgsz=1008))
        ids_seen, frame_counts = set(), []
        t0 = time.perf_counter()
        for i, r in enumerate(pred(source=src, text=[text], stream=True)):
            data = r.boxes.data.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 7))
            ids = data[:, 4].astype(int).tolist() if data.shape[1] >= 7 else list(range(len(data)))
            ids_seen.update(ids)
            frame_counts.append(len(data))
            if on_frame is not None:
                on_frame(i, r.orig_img[..., ::-1], data[:, :4], ids, len(data))
        return {"count": len(ids_seen), "frame_counts": frame_counts, "frames": len(frame_counts),
                "sec": time.perf_counter() - t0}
