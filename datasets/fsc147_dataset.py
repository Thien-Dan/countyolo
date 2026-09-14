import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from .density_generator import generate_density_map

class FSC147Dataset(Dataset):
    """
    Dataset loader cho FSC-147 (images_384_VarV2 variant).

    Quy ước tọa độ của FSC-147 (xác nhận bằng thực nghiệm đối chiếu với GT density maps):
    - annotation['points']          : danh sách [y_orig, x_orig] trong ảnh GỐC (resolution cao).
    - 'box_examples_coordinates'    : mỗi box là 4 điểm [[y,x], ...] trong ảnh GỐC.
    - Images trong images_384_VarV2 : đã rescale về h=384, w tỉ lệ.
    - ratio_h = 384 / H_orig, ratio_w = W_384 / W_orig (có trong annotation).
    => Để sinh density map khớp với ảnh 384-scaled, ta phải scale:
       y_scaled = y_orig * ratio_h, x_scaled = x_orig * ratio_w
    """

    def __init__(self, data_dir: str, split: str = 'train', transform=None, sigma: float = 4.0):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.sigma = sigma

        anno_path  = os.path.join(data_dir, "annotation_FSC147_384.json")
        split_path = os.path.join(data_dir, "Train_Test_Val_FSC_147.json")

        with open(anno_path, 'r') as f:
            self.annotations = json.load(f)

        with open(split_path, 'r') as f:
            splits = json.load(f)
            if split == 'val':
                self.image_names = splits['val']
            elif split == 'test':
                self.image_names = splits['test']
            else:
                self.image_names = splits['train']

        self.img_dir = os.path.join(data_dir, "images_384_VarV2")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        anno     = self.annotations[img_name]

        # 1. Đọc ảnh đã rescale (H=384, W_scaled tỉ lệ)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            image = Image.new("RGB", (384, 384))

        W_scaled, H_scaled = image.size  # PIL trả về (W, H)

        # 2. Tỉ lệ scale từ annotation
        ratio_h = anno.get('ratio_h', H_scaled / anno['H'])
        ratio_w = anno.get('ratio_w', W_scaled / anno['W'])

        # 3. Points: annotation lưu [y_orig, x_orig] → scale sang không gian 384
        points_list = anno.get('points', [])
        if points_list:
            pts_orig   = torch.tensor(points_list, dtype=torch.float32)  # (N, 2): [y, x]
            pts_scaled = torch.stack([
                pts_orig[:, 0] * ratio_h,  # y_scaled
                pts_orig[:, 1] * ratio_w,  # x_scaled
            ], dim=1)                       # (N, 2): [y_scaled, x_scaled]
        else:
            pts_scaled = torch.zeros((0, 2))

        # 4. Exemplar boxes: mỗi box là 4 điểm [[y_orig, x_orig], ...]
        #    Đổi sang [x1, y1, x2, y2] chuẩn trong không gian 384
        boxes_list = anno.get('box_examples_coordinates', [])
        boxes = []
        for box in boxes_list:
            y_min = min(p[0] for p in box) * ratio_h
            y_max = max(p[0] for p in box) * ratio_h
            x_min = min(p[1] for p in box) * ratio_w
            x_max = max(p[1] for p in box) * ratio_w
            boxes.append([x_min, y_min, x_max, y_max])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))

        # 5. Sinh Density Map D_gt tại scaled resolution
        #    generate_density_map nhận points theo (N, 2): [x, y]
        #    => hoán đổi cột: pts_xy[:,0] = x_scaled, pts_xy[:,1] = y_scaled
        if len(pts_scaled) > 0:
            pts_xy = pts_scaled[:, [1, 0]]  # [x_scaled, y_scaled]
        else:
            pts_xy = pts_scaled
        density_map = generate_density_map(
            image_h=H_scaled,
            image_w=W_scaled,
            points=pts_xy,
            sigma=self.sigma
        )

        # 6. Chuyển ảnh sang tensor (3, H, W)
        image_tensor = TF.to_tensor(image)

        return {
            'image':       image_tensor,
            'density_map': density_map,
            'points':      pts_scaled,  # [y_scaled, x_scaled] trong không gian 384
            'boxes':       boxes,       # [x1, y1, x2, y2] trong không gian 384
            'image_name':  img_name
        }
