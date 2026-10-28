"""
Engine đếm đối tượng bằng Grounding DINO (IDEA-Research, qua HuggingFace Transformers).
Phương pháp phát hiện đối tượng Zero-Shot / Open-Vocabulary dựa trên Transformer (DETR).

Ưu điểm nổi bật cho bài toán đếm:
- NMS-Free: Sử dụng Object Queries và Bipartite Matching (Hungarian Loss), không dùng NMS truyền thống
  nên phát hiện cực tốt các đối tượng dày đặc, chồng lấn hoặc đè lên nhau.
- Cross-Modality: Tương tác sâu giữa đặc trưng ngôn ngữ (BERT) và đặc trưng thị giác (Swin Transformer).
"""
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COUNTGD_HF = os.path.join(_ROOT, "checkpoints", "countgd_box_hf")
DEFAULT_MODEL_ID = _COUNTGD_HF if os.path.exists(_COUNTGD_HF) else "IDEA-Research/grounding-dino-tiny"


class GroundingDinoCounter:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        print(f"Loading Grounding DINO model ({model_id}) on {self.device}...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(self.device).eval()
        print(f"Grounding DINO loaded successfully!", flush=True)

    @torch.no_grad()
    def count_image(
        self,
        image: Image.Image,
        text: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Đếm và định vị các đối tượng trong ảnh theo text query bằng Grounding DINO.

        Args:
            image: PIL.Image (RGB)
            text: Tên hoặc mô tả đối tượng cần đếm (ví dụ: 'tomatoes', 'apples')
            box_threshold: Ngưỡng tự tin của bounding box (mặc định: 0.25)
            text_threshold: Ngưỡng tự tin liên kết text và vùng ảnh (mặc định: 0.25)

        Returns:
            dict chứa:
                - count: Tổng số lượng đối tượng đếm được (int)
                - boxes: Toạ độ bounding box xyxy trên ảnh gốc (np.ndarray N x 4)
                - scores: Điểm tin cậy của từng box (np.ndarray N)
                - labels: Nhãn văn bản tương ứng cho từng box (list[str])
                - ms: Thời gian suy luận (float, tính bằng ms)
        """
        box_thr = box_threshold if box_threshold is not None else self.box_threshold
        text_thr = text_threshold if text_threshold is not None else self.text_threshold

        # Grounding DINO trong HuggingFace yêu cầu prompt kết thúc bằng dấu chấm
        prompt = text.strip()
        if not prompt.endswith("."):
            prompt = prompt + "."

        t0 = time.perf_counter()

        # Chuẩn bị inputs cho model
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        # Suy luận (Inference)
        outputs = self.model(**inputs)

        # Hậu xử lý (Post-processing): lấy toạ độ box trên kích thước ảnh gốc
        w, h = image.size
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_thr,
            text_threshold=text_thr,
            target_sizes=[(h, w)],
        )[0]

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000

        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        labels = results["labels"]

        return {
            "count": int(len(boxes)),
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "ms": ms,
        }
