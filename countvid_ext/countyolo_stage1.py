"""
Stage 1 của CountVid bằng CountYOLO (YOLO-World + DINOv2) thay cho CountGD-Box: mỗi frame -> box prompt cho SAM 2.

Box = hình vuông tại round(count) đỉnh mạnh nhất của density, cạnh = kích thước vật Scale Head dự đoán; có
test-time tiling theo số đếm (utils/scale_zoom.density_tiled_count). Code CountYOLO được nạp theo đường dẫn file
vì CountVid cũng có package `models` / `datasets` trùng tên.

Tuỳ chọn frame chính SAM 3 (sam3_weights, kf_every > 0): cứ kf_every frame chạy SAM 3 một lượt trên frame đó
  - không thấy vật nào của lớp được hỏi -> các frame tới frame chính sau không có box (chống đếm nhầm lớp)
  - có -> tâm các box SAM 3 điểm cao nhất làm exemplar, lấy prototype của CountYOLO tại đó và dùng cho mọi frame
    tới frame chính sau (CountYOLO đếm "vật giống exemplar" thay vì đỉnh density của chính frame).
Chi phí SAM 3 (~0.2 s / lượt trên T4) chia đều cho kf_every frame.
"""
import importlib.util
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CountYOLOStage1:
    def __init__(self, countyolo_dir, checkpoint, device="cuda", ttn_thr=0.0, ttn_grid=2, min_mass=0.0,
                 sam3_weights="", kf_every=0, sam3_conf=0.2, n_exemplars=5):
        dy = _load("countyolo_density_yolo", os.path.join(countyolo_dir, "models", "density_yolo.py"))
        self._tiled = _load("countyolo_scale_zoom", os.path.join(countyolo_dir, "utils", "scale_zoom.py")) \
            .density_tiled_count
        self.model = dy.DensityYOLOWorld("yolov8s-world.pt")
        ckpt = self.model.load_head(checkpoint, map_location="cpu")
        self.model.to(device).eval_mode()
        self.device, self.ttn_thr, self.ttn_grid, self.min_mass = device, ttn_thr, ttn_grid, min_mass
        self.sam3, self.kf_every, self.n_exemplars = None, kf_every, n_exemplars
        if sam3_weights and kf_every > 0:
            self.sam3 = _load("countyolo_sam3_counter", os.path.join(countyolo_dir, "engines", "sam3_counter.py"))                 .SAM3Counter(sam3_weights, conf=sam3_conf)
        self.calls, self.protos, self.present, self.kf_log = 0, None, True, []
        print(f"CountYOLO Stage 1: arch={self.model.arch}, epoch {ckpt.get('epoch')}, "
              f"ttn_thr={ttn_thr}, ttn_grid={ttn_grid}, min_mass={min_mass}, "
              f"SAM 3 keyframe every {kf_every if self.sam3 else 0}", flush=True)

    @torch.no_grad()
    def keyframe(self, image, text, size=640):
        """SAM 3 trên một frame -> cờ có mặt + prototype exemplar của CountYOLO tại tâm các box điểm cao nhất."""
        boxes, scores = self.sam3._detect(np.array(image), text)
        self.present = len(boxes) > 0
        self.protos = None
        if self.present:
            top = scores.argsort(descending=True)[:self.n_exemplars]
            w, h = image.size
            pts = (boxes[top, :2] + boxes[top, 2:]) / 2 * torch.tensor([size / w, size / h])
            x = TF.to_tensor(image.resize((size, size)))[None].to(self.device)
            self.protos = self.model(x, [text], exemplar_points=[pts], return_aux=True)[2]["protos"]
        self.kf_log.append((self.calls, int(len(boxes))))

    @torch.no_grad()
    def boxes(self, image, text):
        """PIL.Image (RGB) + text -> (box xyxy (N, 4) toạ độ ảnh gốc, score (N,), count)."""
        if self.sam3 is not None and self.calls % self.kf_every == 0:
            self.keyframe(image, text)
        self.calls += 1
        if not self.present:
            return torch.zeros(0, 4), torch.zeros(0), 0.0
        count, _, boxes, scores = self._tiled(self.model, image, text, self.device, count_thr=self.ttn_thr,
                                              grid=self.ttn_grid, return_boxes=True, min_mass=self.min_mass,
                                              protos=self.protos)
        w, h = image.size
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        return boxes.float(), scores.float(), count
