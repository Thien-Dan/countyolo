"""Frozen DINOv3 ROI appearance, independently loaded from detector weights."""
import numpy as np
import torch
from torchvision.ops import roi_align
from models.density_yolo import FrozenViT, DINOV3_SMALL_REVISION


class DINOIdentityFeatures:
    def __init__(self, device='cpu'):
        self.device = device
        self.metadata = dict(model='facebook/dinov3-vits16-pretrain-lvd1689m',
                             revision=DINOV3_SMALL_REVISION, input_size=640,
                             pooling='box ROIAlign3x3 mean; untrained instance appearance')
        self.model = FrozenViT('facebook/dinov3-vits16-pretrain-lvd1689m', 640,
                               DINOV3_SMALL_REVISION).to(device).eval()

    @torch.inference_mode()
    def __call__(self, rgb, boxes):
        if not len(boxes):
            return []
        h, w = rgb.shape[:2]
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).float().div(255).to(self.device)
        grid = self.model(image[None]).float()
        boxes = np.asarray(boxes)
        valid = np.flatnonzero(np.all(boxes[:,2:] > boxes[:,:2], axis=1))
        if not len(valid):
            return [None]*len(boxes)
        regions = torch.as_tensor(boxes[valid], dtype=torch.float32, device=self.device).clone()
        regions *= torch.tensor([640/w,640/h,640/w,640/h], device=self.device)
        pooled = roi_align(grid, [regions], 3, spatial_scale=grid.shape[-1]/640, aligned=True).mean((-1,-2))
        features = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy().tolist()
        result = [None]*len(boxes)
        for i, feature in zip(valid, features):result[int(i)] = feature
        return result
