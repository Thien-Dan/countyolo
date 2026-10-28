import os
import json
import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from .density_generator import generate_density_map


def load_class_names(data_dir: str) -> dict:
    """Đọc ImageClasses_FSC147.txt -> {image_name: class_name}."""
    img_to_class = {}
    with open(os.path.join(data_dir, "ImageClasses_FSC147.txt"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_to_class[parts[0]] = parts[1]
    return img_to_class


class FSC147Dataset(Dataset):
    """
    Dataset loader cho FSC-147 (images_384_VarV2), ảnh được resize (kéo giãn) về image_size x image_size
    để khớp với đầu vào YOLO-World; density map GT ở lưới stride (80x80 với 640/8).

    Quy ước tọa độ của FSC-147 (đã kiểm chứng bằng cách đối chiếu với gt_density_map_adaptive_384_VarV2:
    mật độ tại điểm cao hơn ~20-30 lần so với cách đọc [y, x]):
    - annotation['points']       : [x, y] trong không gian ảnh 384 (images_384_VarV2), KHÔNG cần scale thêm.
    - 'box_examples_coordinates' : mỗi box là 4 điểm [[x, y], ...] trong không gian 384.
    - ratio_h / ratio_w          : tỉ lệ ảnh 384 so với ảnh gốc, không dùng cho points.

    Density GT: ưu tiên map adaptive chính thức (gt_density_map_adaptive_384_VarV2), sum-pool về lưới stride
    (bảo toàn số lượng). Nếu thiếu thư mục đó, sinh Gaussian từ points ngay trên lưới stride.
    """

    def __init__(self, data_dir: str, split: str = 'train', image_size: int = 640, stride: int = 8,
                 hflip: bool = False, sigma: float = 1.0, mosaic_prob: float = 0.0, mosaic_grids=(2,),
                 neg_prob: float = 0.0, mixed_prob: float = 0.0, pseudo: str = None, hard_prob: float = 0.0):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.grid = image_size // stride
        self.stride = stride
        self.hflip = hflip
        self.sigma = sigma  # chỉ dùng khi không có GT density chính thức (đơn vị: ô lưới)
        # Mosaic cùng lớp: ghép g x g ảnh cùng lớp (g ngẫu nhiên trong mosaic_grids) -> ảnh đông giả,
        # count = tổng g*g ảnh (chỉ dùng khi train). g = 4 tạo cảnh dày đặc vật nhỏ mà train thiếu so với test.
        self.mosaic_prob = mosaic_prob
        self.mosaic_grids = tuple(mosaic_grids)
        for g in self.mosaic_grids:
            if image_size % g or self.grid % g:
                raise ValueError(f"mosaic grid {g} phải chia hết ảnh {image_size} và lưới {self.grid}")
        # Dạy mô hình đếm ĐÚNG lớp được hỏi (FSC-147 chỉ gán nhãn 1 lớp/ảnh nên text luôn khớp ảnh và mô hình học
        # cách bỏ qua text): neg_prob = xác suất thay prompt bằng lớp KHÁC (đáp án density 0); mixed_prob = tỉ lệ
        # mosaic trộn ô của lớp khác (chỉ ô đúng lớp có density).
        self.neg_prob = neg_prob
        self.mixed_prob = mixed_prob
        # Nhãn giả SAM 3 (tools/sam3_pseudo.py): ảnh -> {prompt: (mask 80x80 uint8, số instance)}. Cho cổng text
        # của vitgate một đích "chỗ nào là khái niệm được hỏi"; hard_prob = xác suất đổi prompt sang một lớp KHÁC
        # mà SAM 3 thấy có thật trong ảnh (negative khó: đích = mask của lớp đó, không có density GT).
        self.pseudo, self.hard_prob = {}, hard_prob
        if pseudo:
            d = np.load(pseudo)
            for k, m, c in zip(d["keys"], d["masks"], d["counts"]):
                img, prompt = str(k).split("|", 1)
                self.pseudo.setdefault(img, {})[prompt] = (m, int(c))

        with open(os.path.join(data_dir, "annotation_FSC147_384.json"), 'r') as f:
            self.annotations = json.load(f)
        with open(os.path.join(data_dir, "Train_Test_Val_FSC_147.json"), 'r') as f:
            self.image_names = json.load(f)[split]
        self.img_to_class = load_class_names(data_dir)
        self.class_to_idx = {}
        for i, name in enumerate(self.image_names):
            self.class_to_idx.setdefault(self.img_to_class.get(name, "object"), []).append(i)

        self.img_dir = os.path.join(data_dir, "images_384_VarV2")
        self.gt_dir = os.path.join(data_dir, "gt_density_map_adaptive_384_VarV2")
        self.use_official_gt = os.path.isdir(self.gt_dir)

    def __len__(self):
        return len(self.image_names)

    def _density(self, img_name, points_grid, count):
        if self.use_official_gt:
            d = np.load(os.path.join(self.gt_dir, os.path.splitext(img_name)[0] + ".npy")).astype(np.float32)
            # INTER_AREA lấy trung bình -> nhân diện tích để thành sum-pool, rồi chỉnh lại tổng cho khớp chính xác
            d = cv2.resize(d, (self.grid, self.grid), interpolation=cv2.INTER_AREA)
            if d.sum() > 0:
                d = d / d.sum() * count
            return torch.from_numpy(d).unsqueeze(0)
        return generate_density_map(self.grid, self.grid, points_grid, sigma=self.sigma)

    def __getitem__(self, idx):
        if torch.rand(1).item() < self.mosaic_prob:
            sample = self._mosaic(idx, mixed=torch.rand(1).item() < self.mixed_prob)
        else:
            sample = self._load(idx)
        if torch.rand(1).item() < self.neg_prob:
            sample = self._negative(sample)
        elif self.pseudo and torch.rand(1).item() < self.hard_prob and 'mosaic' not in sample:
            sample = self._hard_negative(sample)

        if self.hflip and torch.rand(1).item() < 0.5:
            sample['image'] = sample['image'].flip(-1)
            sample['density_map'] = sample['density_map'].flip(-1)
            sample['size_map'] = sample['size_map'].flip(-1)
            sample['gate_mask'] = sample['gate_mask'].flip(-1)
            sample['points'][:, 0] = self.image_size - sample['points'][:, 0]
            if 'other_points' in sample:
                sample['other_points'][:, 0] = self.image_size - sample['other_points'][:, 0]
            sample['boxes'][:, [0, 2]] = self.image_size - sample['boxes'][:, [2, 0]]
        return sample

    def _other_class(self, cls):
        others = [c for c in self.class_to_idx if c != cls]
        return others[torch.randint(len(others), (1,)).item()]

    def _negative(self, sample):
        """Giữ ảnh, đổi prompt sang một lớp khác -> đáp án: không có vật nào."""
        return {**sample, 'class_name': self._other_class(sample['class_name']),
                'density_map': torch.zeros_like(sample['density_map']), 'points': torch.zeros(0, 2), 'count': 0.0,
                'gate_mask': torch.zeros_like(sample['gate_mask'])}

    def _hard_negative(self, sample):
        """Đổi prompt sang lớp khác mà SAM 3 thấy có thật trong ảnh; đích cổng = mask SAM 3 của lớp đó.
        Không có density GT cho lớp này -> density_valid = 0 (chỉ học cổng)."""
        own = sample['class_name']
        cands = [(pr, m, c) for pr, (m, c) in self.pseudo.get(sample['image_name'], {}).items() if pr != own and c > 0]
        if not cands:
            return self._negative(sample)
        pr, m, c = cands[torch.randint(len(cands), (1,)).item()]
        return {**sample, 'class_name': pr, 'density_map': torch.zeros_like(sample['density_map']),
                'points': torch.zeros(0, 2), 'gate_mask': torch.from_numpy(m.astype(np.float32) / 255)[None],
                'density_valid': 0.0, 'count': float(c)}

    def _mosaic(self, idx, mixed=False):
        """Ghép g x g ảnh cùng lớp (ảnh idx + g*g-1 ảnh ngẫu nhiên cùng lớp, g chọn trong mosaic_grids),
        mỗi ô thu nhỏ g lần. Density sum-pool g x g nên tổng được bảo toàn; points/boxes co lại và dời theo ô.
        mixed: 1..g*g/2 ô (không phải ô đầu) lấy từ một lớp khác, density/points của các ô đó bị bỏ."""
        g = self.mosaic_grids[torch.randint(len(self.mosaic_grids), (1,)).item()]
        cls = self.img_to_class.get(self.image_names[idx], "object")
        pool = self.class_to_idx[cls]
        others = [pool[j] for j in torch.randint(len(pool), (g * g - 1,)).tolist()]
        negative = set()
        if mixed:
            other_pool = self.class_to_idx[self._other_class(cls)]
            n_neg = torch.randint(1, g * g // 2 + 1, (1,)).item()
            negative = {k + 1 for k in torch.randperm(g * g - 1)[:n_neg].tolist()}
            others = [other_pool[torch.randint(len(other_pool), (1,)).item()] if k + 1 in negative else j
                      for k, j in enumerate(others)]
        cell, gcell = self.image_size // g, self.grid // g
        image = torch.zeros(3, self.image_size, self.image_size)
        density = torch.zeros(1, self.grid, self.grid)
        size_map = torch.zeros(1, self.grid, self.grid)
        gate = torch.zeros(1, self.grid, self.grid)
        points, boxes, other_points = [], [], []
        tiles = [self._load(j) for j in [idx] + others]
        for k, s in enumerate(tiles):
            oy, ox = divmod(k, g)
            image[:, oy * cell:(oy + 1) * cell, ox * cell:(ox + 1) * cell] = \
                torch.nn.functional.interpolate(s['image'][None], size=(cell, cell), mode='area')[0]
            if k not in negative:
                density[:, oy * gcell:(oy + 1) * gcell, ox * gcell:(ox + 1) * gcell] = \
                    torch.nn.functional.avg_pool2d(s['density_map'][None], g)[0] * g * g
                gate[:, oy * gcell:(oy + 1) * gcell, ox * gcell:(ox + 1) * gcell] = \
                    torch.nn.functional.avg_pool2d(s['gate_mask'][None], g)[0]
            # vật co lại g lần -> log2 kích thước giảm log2(g)
            size_map[:, oy * gcell:(oy + 1) * gcell, ox * gcell:(ox + 1) * gcell] = \
                s['size_map'][0, 0, 0] - math.log2(g)
            offset = torch.tensor([ox * cell, oy * cell], dtype=torch.float32)
            if k not in negative:
                points.append(s['points'] / g + offset)
            else:  # vật lớp khác trong mosaic trộn lớp (exemplar nhiễu "chọn nhầm lớp" khi train)
                other_points.append(s['points'] / g + offset)
            boxes.append(s['boxes'] / g + offset.repeat(2))
        points = torch.cat(points)
        return {**tiles[0], 'image': image, 'density_map': density, 'size_map': size_map, 'points': points,
                'boxes': boxes[0], 'count': float(len(points)), 'gate_mask': gate, 'mosaic': True,
                'other_points': torch.cat(other_points) if other_points else torch.zeros(0, 2)}

    def _load(self, idx):
        img_name = self.image_names[idx]
        anno = self.annotations[img_name]

        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        W, H = image.size
        image = image.resize((self.image_size, self.image_size))

        # Points [x, y] (không gian 384) -> không gian image_size
        pts = torch.tensor(anno.get('points', []), dtype=torch.float32).reshape(-1, 2)
        pts[:, 0] *= self.image_size / W
        pts[:, 1] *= self.image_size / H

        # Exemplar boxes: 4 điểm [[x, y], ...] -> [x1, y1, x2, y2] trong không gian image_size
        boxes = []
        for box in anno.get('box_examples_coordinates', []):
            xs = [p[0] * self.image_size / W for p in box]
            ys = [p[1] * self.image_size / H for p in box]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        density = self._density(img_name, pts / self.stride, len(pts))
        image = TF.to_tensor(image)
        # Kích thước vật = trung vị sqrt(w*h) của exemplar box (px trong ảnh S), nhãn cho Scale Head
        side = torch.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).median() \
            if len(boxes) else torch.tensor(float(self.stride * 4))
        size_map = torch.full((1, self.grid, self.grid), torch.log2(side.clamp(min=1.0)).item())

        return {
            'image':       image,                # (3, S, S)
            'density_map': density,              # (1, S/stride, S/stride), sum ≈ count
            'size_map':    size_map,             # (1, S/stride, S/stride), log2 kích thước vật
            'count':       float(len(pts)),
            'class_name':  self.img_to_class.get(img_name, "object"),
            'points':      pts,                  # [x, y] trong không gian S
            'boxes':       boxes,                # exemplar [x1, y1, x2, y2] trong không gian S
            'image_name':  img_name,
            # Đích cổng text: mask SAM 3 của chính lớp này nếu có nhãn giả, không thì vùng có density GT
            'gate_mask':   self._own_gate(img_name, density),
            'density_valid': 1.0,
        }

    def _own_gate(self, img_name, density):
        m = self.pseudo.get(img_name, {}).get(self.img_to_class.get(img_name, "object"))
        if m is not None:
            return torch.from_numpy(m[0].astype(np.float32) / 255)[None]
        return (torch.nn.functional.max_pool2d(density[None], 3, 1, 1)[0] > 1e-3).float()


def collate_fsc147(batch):
    """points/boxes có độ dài khác nhau giữa các ảnh -> giữ dạng list."""
    return {
        'image':       torch.stack([b['image'] for b in batch]),
        'density_map': torch.stack([b['density_map'] for b in batch]),
        'size_map':    torch.stack([b['size_map'] for b in batch]),
        'count':       torch.tensor([b['count'] for b in batch]),
        'class_name':  [b['class_name'] for b in batch],
        'points':      [b['points'] for b in batch],
        'other_points': [b.get('other_points', torch.zeros(0, 2)) for b in batch],
        'boxes':       [b['boxes'] for b in batch],
        'image_name':  [b['image_name'] for b in batch],
        'gate_mask':   torch.stack([b['gate_mask'] for b in batch]),
        'density_valid': torch.tensor([b.get('density_valid', 1.0) for b in batch]),
    }
