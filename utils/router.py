import torch

class DualModeRouter:
    """
    Module định tuyến động: Chuyển đổi Box Prompt thành Point Prompt cho vật thể nhỏ.
    Nguyên lý:
    - Nếu Box có max(width, height) < size_threshold: Tìm cực đại địa phương trên Density Map trong vùng box đó để sinh Point Prompt.
    - Ngược lại: Giữ nguyên Box Prompt.
    """
    def __init__(self, size_threshold: int = 15):
        self.size_threshold = size_threshold

    def route(self, boxes: torch.Tensor, density_map: torch.Tensor):
        """
        Args:
            boxes: Tensor shape (N, 4) định dạng [x1, y1, x2, y2].
            density_map: Tensor shape (1, H, W) bản đồ mật độ đã dự đoán.
        Returns:
            points: Tensor shape (M, 2) chứa [x, y] của các vật nhỏ (M <= N).
            point_labels: Tensor shape (M,) chứa nhãn 1 (foreground) cho SAM.
            large_boxes: Tensor shape (K, 4) chứa các box lớn (K = N - M).
            indices_point: Index của các vật thể được chuyển thành point.
            indices_box: Index của các vật thể giữ nguyên box.
        """
        if len(boxes) == 0:
            return torch.zeros((0, 2)), torch.zeros((0,)), torch.zeros((0, 4)), [], []

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        max_sizes = torch.max(widths, heights)

        # Phân loại Box vs Point
        mask_small = max_sizes < self.size_threshold
        indices_point = torch.nonzero(mask_small).squeeze(1).tolist()
        indices_box = torch.nonzero(~mask_small).squeeze(1).tolist()

        large_boxes = boxes[indices_box]
        small_boxes = boxes[indices_point]

        points = []
        point_labels = []

        # Với mỗi box nhỏ, tìm điểm có mật độ cực đại
        H, W = density_map.shape[1], density_map.shape[2]
        
        for box in small_boxes:
            x1, y1, x2, y2 = box.int().tolist()
            
            # Clip vào giới hạn ảnh
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            
            if x2 <= x1 or y2 <= y1:
                # Nếu box lỗi (diện tích <= 0), lấy tâm box
                points.append([(x1+x2)/2.0, (y1+y2)/2.0])
                point_labels.append(1)
                continue

            # Crop vùng density
            crop_density = density_map[0, y1:y2, x1:x2]
            
            # Tìm local maximum trong crop
            # unravel_index để tìm tọa độ 2D từ flattened argmax
            flat_idx = torch.argmax(crop_density)
            local_y = flat_idx // crop_density.shape[1]
            local_x = flat_idx % crop_density.shape[1]
            
            # Đổi về tọa độ global ảnh gốc
            global_x = x1 + local_x.item()
            global_y = y1 + local_y.item()
            
            points.append([global_x, global_y])
            point_labels.append(1)

        points = torch.tensor(points, dtype=torch.float32, device=boxes.device) if points else torch.zeros((0, 2), device=boxes.device)
        point_labels = torch.tensor(point_labels, dtype=torch.int32, device=boxes.device) if point_labels else torch.zeros((0,), dtype=torch.int32, device=boxes.device)

        return points, point_labels, large_boxes, indices_point, indices_box
