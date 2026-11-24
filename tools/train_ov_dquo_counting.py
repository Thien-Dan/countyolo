"""Train & evaluate OV-DQUO (EVA02-CLIP + CLIPSelf + 3-Scale Multi-Scale Deformable Attention) on FSC-147.

Key features:
- Loads official `OVDQUO_ViT-B16_LVIS.pth` / `OVDQUO_ViT-L14_LVIS.pth` with strict weight verification.
- Pure-PyTorch Multi-Scale Deformable Attention (3 scales stride 8/16/32) + 1,000 queries (100% NMS-free).
- Differentiable 7-layer deep supervision (6 decoder layers with Look-Forward-Twice + Stage-1 encoder proposals)
  with probabilistic Log-AND scoring (`logit(sigmoid(dec_obj) * sigmoid(clip_sem))`).
- Balanced Modality Dropout (40% Text-Only / 35% Multimodal / 25% Exemplar-Only) when `--use-exemplars` is set.
- 4 disjoint gradient-clipping buckets, Warmup+Cosine LR schedule, GradScaler(init_scale=256.0), and 2x2 tiling.
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from models.yoloworld_counting import CountingCriterion, xywh_to_xyxy, exemplar_point_indices
from models.ov_dquo_counting import OVDQUOCounting, build_ov_dquo_class_list
from datasets.yoloworld_counting import load_fsc147d_class_names
from tools.train_yoloworld_counting import THRESHOLDS, save, totals


def make_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int = 400, min_lr_ratio: float = 0.1):
    """Linear warmup from min_lr_ratio -> 1.0, then smooth cosine decay back to min_lr_ratio."""
    total_steps = max(1, total_steps)
    warmup_steps = min(warmup_steps, max(1, total_steps // 4))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * (step / max(1, warmup_steps))
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def prepare_ov_dquo_sample(image: Image.Image, record: dict, flip=False, crop_scale_range=None, max_queries=1000):
    """Prepare PIL image + image-normalized [0,1] points/exemplars (OVDQUOCounting preserves aspect ratio + pads cleanly)."""
    points = np.asarray(record['points'], dtype=np.float32).reshape(-1, 2).copy()
    boxes = np.asarray([np.concatenate((np.min(b, axis=0), np.max(b, axis=0)))
                        for b in record.get('box_examples_coordinates', [])], dtype=np.float32).reshape(-1, 4)
    raw_exemplar_count = len(boxes)
    supervision_points = points.copy()
    indices, matched_boxes = exemplar_point_indices(points, boxes)
    for idx, box in zip(indices, matched_boxes):
        supervision_points[idx] = (box[:2] + box[2:]) / 2.0

    w, h = image.size
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        points[:, 0] = w - points[:, 0]
        supervision_points[:, 0] = w - supervision_points[:, 0]
        if len(boxes):
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

    if crop_scale_range is not None and len(supervision_points) > 1:
        lo, hi = crop_scale_range
        scale = random.uniform(lo, hi)
        cw, ch = max(64, int(round(w * scale))), max(64, int(round(h * scale)))
        if len(boxes):
            bx0, by0 = float(boxes[:, 0].min()), float(boxes[:, 1].min())
            bx1, by1 = float(boxes[:, 2].max()), float(boxes[:, 3].max())
            cw = max(cw, min(w, int(math.ceil(bx1 - bx0)) + 8))
            ch = max(ch, min(h, int(math.ceil(by1 - by0)) + 8))
            x0_min = max(0, int(math.ceil(bx1)) - cw)
            x0_max = min(w - cw, int(math.floor(bx0)))
            y0_min = max(0, int(math.ceil(by1)) - ch)
            y0_max = min(h - ch, int(math.floor(by0)))
        else:
            x0_min, x0_max = 0, w - cw
            y0_min, y0_max = 0, h - ch
        if x0_min <= x0_max and y0_min <= y0_max and (cw < w or ch < h):
            x0 = random.randint(x0_min, x0_max)
            y0 = random.randint(y0_min, y0_max)
            keep_pts = (
                (supervision_points[:, 0] >= x0) & (supervision_points[:, 0] <= x0 + cw) &
                (supervision_points[:, 1] >= y0) & (supervision_points[:, 1] <= y0 + ch)
            )
            if int(keep_pts.sum()) >= max(1, len(boxes)):
                image = image.crop((x0, y0, x0 + cw, y0 + ch))
                supervision_points = supervision_points[keep_pts] - np.array([x0, y0], dtype=np.float32)
                keep_raw = (
                    (points[:, 0] >= x0) & (points[:, 0] <= x0 + cw) &
                    (points[:, 1] >= y0) & (points[:, 1] <= y0 + ch)
                )
                points = points[keep_raw] - np.array([x0, y0], dtype=np.float32)
                if len(boxes):
                    boxes = boxes - np.array([x0, y0, x0, y0], dtype=np.float32)
                    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(cw))
                    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(ch))
                w, h = cw, ch

    if len(points):
        points = np.clip(points / np.array([w, h], dtype=np.float32), 0.0, 1.0)
        supervision_points = np.clip(supervision_points / np.array([w, h], dtype=np.float32), 0.0, 1.0)
    if len(boxes):
        boxes = np.clip(boxes / np.array([w, h, w, h], dtype=np.float32), 0.0, 1.0)
        valid_b = ((boxes[:, 2] - boxes[:, 0]) > 1e-4) & ((boxes[:, 3] - boxes[:, 1]) > 1e-4)
        boxes = boxes[valid_b]

    if len(supervision_points) > max_queries:
        ex_idx, _ = exemplar_point_indices(supervision_points, boxes)
        keep_set = set(ex_idx.tolist())
        remaining = [i for i in range(len(supervision_points)) if i not in keep_set]
        random.shuffle(remaining)
        chosen = sorted(list(keep_set) + remaining[:max_queries - len(keep_set)])
        supervision_points = supervision_points[chosen]

    exemplar_indices, exemplar_boxes = exemplar_point_indices(supervision_points, boxes)
    return dict(
        pil_image=image,
        points=torch.from_numpy(points),
        supervision_points=torch.from_numpy(supervision_points),
        exemplar_indices=torch.from_numpy(exemplar_indices),
        exemplar_boxes=torch.from_numpy(exemplar_boxes),
        raw_exemplar_count=raw_exemplar_count,
    )


class OVDQUOData:
    def __init__(self, root: Path, split: str, exclude=(), max_queries=1000):
        self.root = Path(root)
        self.annotations = json.loads((self.root / 'annotation_FSC147_384.json').read_text())
        self.names = [n for n in json.loads((self.root / 'Train_Test_Val_FSC_147.json').read_text())[split]
                      if n not in set(exclude)]
        self.classes = load_fsc147d_class_names(str(self.root))
        self.split = split
        self.max_queries = max_queries

    def get(self, name: str, augment=False):
        image = Image.open(self.root / 'images_384_VarV2' / name).convert('RGB')
        record = self.annotations[name]
        if name == '7171.jpg':
            record = dict(record, box_examples_coordinates=[])
        crop_range = (0.70, 1.0) if (augment and random.random() < 0.5) else None
        sample = prepare_ov_dquo_sample(
            image, record, flip=augment and random.random() < 0.5,
            crop_scale_range=crop_range, max_queries=self.max_queries,
        )
        sample.update(name=name, prompt=self.classes[name])
        return sample


def _focal_cls_from_match(
    logits: torch.Tensor,
    pred_ids: torch.Tensor,
    normalizer: float,
    target_tokens: torch.Tensor = None,
    neg_class_weight: float = 1.0,
) -> torch.Tensor:
    """Exact CountGD focal classification loss given precomputed matched `pred_ids`."""
    z = logits.float()
    p = z.sigmoid()
    positive = 0.25 * (1.0 - p).square() * (-F.logsigmoid(z))
    negative = 0.75 * p.square() * (-F.logsigmoid(-z))
    neg_weight = 1.0
    if z.ndim == 2 and target_tokens is not None and (~target_tokens).any():
        num_pos_tok = max(1, int(target_tokens.sum().item()))
        num_neg_tok = max(1, int((~target_tokens).sum().item()))
        tok_neg_scale = torch.ones((1, z.shape[1]), device=z.device, dtype=torch.float32)
        tok_neg_scale[:, ~target_tokens] = float(neg_class_weight) * min(1.0, num_pos_tok / num_neg_tok)
        neg_weight = tok_neg_scale
    target = torch.zeros_like(z, dtype=torch.float32)
    if len(pred_ids):
        if z.ndim == 1:
            target[pred_ids] = 1.0
        else:
            target[pred_ids[:, None], torch.where(target_tokens)[0][None, :]] = 1.0
    return torch.where(target.bool(), positive, negative * neg_weight).sum() / float(normalizer)


def _hungarian_match_stage(
    criterion: CountingCriterion,
    boxes: torch.Tensor,
    logits: torch.Tensor,
    points: torch.Tensor,
    target_tokens: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a SINGLE Hungarian matching for a stage (handles empty GT for absent-prompt batches cleanly)."""
    if len(points) == 0:
        empty = torch.empty((0,), device=boxes.device, dtype=torch.long)
        return empty, empty
    if len(points) > len(boxes):
        raise ValueError(f'More targets ({len(points)}) than candidate slots ({len(boxes)}); never silently drop GT')
    with torch.no_grad():
        z = logits.float()
        p = z.sigmoid()
        positive = 0.25 * (1.0 - p).square() * (-F.logsigmoid(z))
        negative = 0.75 * p.square() * (-F.logsigmoid(-z))
        cost = criterion.loc * torch.cdist(boxes[:, :2].float(), points.float(), p=1)
        class_cost = positive - negative
        if z.ndim == 2:
            class_cost = class_cost[:, target_tokens].mean(-1)
        cost = cost + criterion.cls * class_cost[:, None]
        pred_ids, target_ids = linear_sum_assignment(cost.detach().cpu().numpy())
        return (
            torch.as_tensor(pred_ids, device=boxes.device, dtype=torch.long),
            torch.as_tensor(target_ids, device=boxes.device, dtype=torch.long),
        )


def _geometry_loss_from_match(
    boxes: torch.Tensor,
    points: torch.Tensor,
    ex_idx: torch.Tensor,
    ex_xyxy: torch.Tensor,
    pred_ids: torch.Tensor,
    target_ids: torch.Tensor,
    normalizer: float,
):
    zero = boxes.sum() * 0.0
    if len(points) == 0:
        return zero, zero, zero
    loss_center = F.l1_loss(boxes[pred_ids, :2].float(), points[target_ids].float(), reduction='sum') / float(normalizer)
    loss_wh, loss_giou = zero, zero
    if len(ex_idx):
        inverse = torch.full((len(points),), -1, device=boxes.device, dtype=torch.long)
        inverse[target_ids] = pred_ids
        matched = boxes[inverse[ex_idx]].float()
        if (inverse[ex_idx] < 0).any():
            raise AssertionError('Exemplar target unmatched')
        target_wh = ex_xyxy[:, 2:] - ex_xyxy[:, :2]
        loss_wh = F.l1_loss(matched[:, 2:], target_wh.float(), reduction='sum') / float(normalizer)
        loss_giou = (1.0 - generalized_box_iou(xywh_to_xyxy(matched), ex_xyxy.float()).diag()).sum() / float(normalizer)
    return loss_center, loss_wh, loss_giou


def _stage_loss(criterion: CountingCriterion, st: dict, pts, ex_idx, ex_xyxy, target_idx: int,
                device: str, neg_class_weight: float, absent_normalizer: float = 30.0):
    """v3.1: Unified SINGLE Hungarian matching per stage + calibrated absent normalizer (prevents recall collapse)."""
    boxes = st['boxes']
    has_text = st['text_logits'] is not None
    has_vis = st['vis_logits'] is not None
    is_absent = (len(pts) == 0)
    normalizer = float(absent_normalizer) if is_absent else float(len(pts))

    if has_text and has_vis:
        k_cls = st['text_logits'].shape[1]
        t_mask = torch.zeros(k_cls, device=device, dtype=torch.bool)
        t_mask[target_idx] = True
        pred_ids, target_ids = _hungarian_match_stage(criterion, boxes, st['mm_logits'], pts)
        loss_center, loss_wh, loss_giou = _geometry_loss_from_match(
            boxes, pts, ex_idx, ex_xyxy, pred_ids, target_ids, normalizer,
        )
        cls_mm = _focal_cls_from_match(st['mm_logits'], pred_ids, normalizer)
        cls_txt = _focal_cls_from_match(
            st['text_logits'], pred_ids, normalizer, target_tokens=t_mask, neg_class_weight=neg_class_weight,
        )
        cls_vis = _focal_cls_from_match(st['vis_logits'], pred_ids, normalizer)
        loss_cls = 0.50 * cls_mm + 0.25 * cls_txt + 0.25 * cls_vis
    elif has_text:
        if is_absent:
            # Penalize ONLY the queried absent class column normalized by realistic average count (30.0)
            empty = torch.empty((0,), device=boxes.device, dtype=torch.long)
            loss_center = boxes.sum() * 0.0
            loss_wh, loss_giou = loss_center, loss_center
            loss_cls = _focal_cls_from_match(st['text_logits'][:, target_idx], empty, normalizer)
        else:
            k_cls = st['text_logits'].shape[1]
            t_mask = torch.zeros(k_cls, device=device, dtype=torch.bool)
            t_mask[target_idx] = True
            pred_ids, target_ids = _hungarian_match_stage(criterion, boxes, st['text_logits'], pts, target_tokens=t_mask)
            loss_center, loss_wh, loss_giou = _geometry_loss_from_match(
                boxes, pts, ex_idx, ex_xyxy, pred_ids, target_ids, normalizer,
            )
            loss_cls = _focal_cls_from_match(
                st['text_logits'], pred_ids, normalizer, target_tokens=t_mask, neg_class_weight=neg_class_weight,
            )
    else:
        pred_ids, target_ids = _hungarian_match_stage(criterion, boxes, st['vis_logits'], pts)
        loss_center, loss_wh, loss_giou = _geometry_loss_from_match(
            boxes, pts, ex_idx, ex_xyxy, pred_ids, target_ids, normalizer,
        )
        loss_cls = _focal_cls_from_match(st['vis_logits'], pred_ids, normalizer)

    total = criterion.loc * (loss_center + loss_wh) + criterion.giou * loss_giou + criterion.cls * loss_cls
    return total, dict(
        cls=loss_cls.detach(),
        center=loss_center.detach(),
        wh=loss_wh.detach(),
        giou=loss_giou.detach(),
    )


def _spatial_prior_loss(out: dict, pts: torch.Tensor) -> torch.Tensor:
    """Supervise Stage-1 spatial prior heatmap across L0/L1/L2 (trains `enc_prompt_prior` without leaking into `enc_class_topk`)."""
    prior_logits = out.get('prior_logits')
    if prior_logits is None:
        return out['final']['boxes'].sum() * 0.0
    shapes = out['spatial_shapes'].tolist()
    valid_ratios = out['valid_ratios']
    mask_flat = out['mask_flatten'][0]
    dev = prior_logits.device

    tgt_levels = []
    for lvl, (h, w) in enumerate(shapes):
        grid = torch.zeros((h, w), device=dev, dtype=torch.float32)
        if len(pts) > 0:
            vr_w = float(valid_ratios[0, lvl, 0].item()) * float(w)
            vr_h = float(valid_ratios[0, lvl, 1].item()) * float(h)
            max_ix = max(0, min(w - 1, int(math.ceil(vr_w)) - 1))
            max_iy = max(0, min(h - 1, int(math.ceil(vr_h)) - 1))
            ix = (pts[:, 0].float() * vr_w).long().clamp(0, max_ix)
            iy = (pts[:, 1].float() * vr_h).long().clamp(0, max_iy)
            grid[iy, ix] = 1.0
        tgt_levels.append(grid.reshape(-1))
    tgt_flat = torch.cat(tgt_levels, dim=0)
    valid = ~mask_flat
    z = prior_logits[0, valid].float()
    t = tgt_flat[valid]
    p = z.sigmoid()
    pos = 0.25 * (1.0 - p).square() * (-F.logsigmoid(z))
    neg = 0.75 * p.square() * (-F.logsigmoid(-z))
    norm = 30.0 if len(pts) == 0 else max(3.0, float(t.sum().item()))
    return torch.where(t > 0.5, pos, neg).sum() / norm


def compute_ov_dquo_7layer_loss(criterion: CountingCriterion, out: dict, sample: dict, device: str,
                                aux_weight: float = 1.0, neg_class_weight: float = 0.5,
                                prior_weight: float = 0.10):
    """Differentiable 7-layer CountGD deep supervision + auxiliary Stage-1 spatial prior heatmap loss."""
    pts = sample['supervision_points'].to(device)
    ex_idx = sample['exemplar_indices'].to(device)
    ex_xyxy = sample['exemplar_boxes'].to(device)
    tidx = out['target_idx']

    loss_final, parts_final = _stage_loss(criterion, out['final'], pts, ex_idx, ex_xyxy, tidx, device, neg_class_weight)
    total = loss_final
    count = 1.0
    for aux_st in out['aux']:
        l_aux, _ = _stage_loss(criterion, aux_st, pts, ex_idx, ex_xyxy, tidx, device, neg_class_weight)
        total = total + aux_weight * l_aux
        count += aux_weight
    l_enc, _ = _stage_loss(criterion, out['enc'], pts, ex_idx, ex_xyxy, tidx, device, neg_class_weight)
    total = total + aux_weight * l_enc
    count += aux_weight

    l_prior = _spatial_prior_loss(out, pts)
    parts_final['prior'] = l_prior.detach()
    return (total / count) + float(prior_weight) * l_prior, parts_final


@torch.no_grad()
def predict_with_tiling(model: OVDQUOCounting, pil_image: Image.Image, prompt: str,
                        exemplar_boxes=None, use_text: bool = True,
                        threshold: float = 0.23, tile_trigger: int = 650):
    use_amp = torch.cuda.is_available()
    with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
        boxes, logits = model(pil_image, prompt, exemplar_boxes=exemplar_boxes, use_text=use_text)
    boxes = boxes.float().cpu()
    scores = logits.float().sigmoid().cpu()
    if tile_trigger <= 0 or int((scores > threshold).sum().item()) < tile_trigger:
        return boxes, scores

    ex_feats = None
    if model.use_exemplars and exemplar_boxes is not None and len(exemplar_boxes) > 0:
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            # v3 Fix #6: Extract scale-matched 2x exemplar features (`ex_clip`, `ex_mem_mid`, `ex_mem`)
            ex_feats = model.encode_exemplars(
                pil_image, exemplar_boxes, prompt=prompt, use_text=use_text, scale_zoom=2.0,
            )

    w, h = pil_image.size
    mx, my = w // 2, h // 2
    quads = [
        (0, 0, mx, my, 0.0, 0.0, mx / w, my / h),
        (mx, 0, w, my, mx / w, 0.0, (w - mx) / w, my / h),
        (0, my, mx, h, 0.0, my / h, mx / w, (h - my) / h),
        (mx, my, w, h, mx / w, my / h, (w - mx) / w, (h - my) / h),
    ]
    all_boxes, all_scores = [], []
    for x0, y0, x1, y1, ox, oy, sw, sh in quads:
        crop = pil_image.crop((x0, y0, x1, y1))
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            qb, ql = model(crop, prompt, exemplar_feats=ex_feats, use_text=use_text)
        qb = qb.float().cpu()
        qs = ql.float().sigmoid().cpu()
        qb[:, 0] = ox + qb[:, 0] * sw
        qb[:, 1] = oy + qb[:, 1] * sh
        qb[:, 2] = qb[:, 2] * sw
        qb[:, 3] = qb[:, 3] * sh
        all_boxes.append(qb)
        all_scores.append(qs)
    return torch.cat(all_boxes, dim=0), torch.cat(all_scores, dim=0)


@torch.no_grad()
def evaluate_ov_dquo(model: OVDQUOCounting, dataset: OVDQUOData, names, device: str, output: Path,
                     absent: bool = False, use_exemplars: bool = False, use_text: bool = True, tile_trigger: int = 0):
    model.set_training(False)
    rows = []
    for i, name in enumerate(names):
        sample = dataset.get(name, augment=False)
        prompt = 'elephant' if absent else sample['prompt']
        ex_boxes = sample['exemplar_boxes'] if (use_exemplars and not absent and len(sample['exemplar_boxes'])) else None
        effective_use_text = True if (absent or ex_boxes is None) else use_text
        boxes, scores = predict_with_tiling(
            model, sample['pil_image'], prompt, exemplar_boxes=ex_boxes,
            use_text=effective_use_text, tile_trigger=tile_trigger,
        )
        scale = boxes.new_tensor([640.0, 640.0, 640.0, 640.0])
        xyxy = xywh_to_xyxy(boxes) * scale
        centers = ((xyxy[:, :2] + xyxy[:, 2:]) / 2.0).numpy()
        points = np.empty((0, 2)) if absent else sample['points'].numpy() * 640.0
        methods = {}
        for threshold in THRESHOLDS:
            chosen = centers[scores.numpy() > threshold]
            tp = int((maximum_bipartite_matching(csr_matrix(cdist(chosen, points) <= 16), perm_type='column') >= 0).sum()) if len(chosen) and len(points) else 0
            methods[str(threshold)] = dict(count=len(chosen), tp=tp, fp=len(chosen) - tp, fn=len(points) - tp)
        rows.append(dict(image=name, prompt=prompt, gt=len(points), methods=methods))
        if (i + 1) % 100 == 0:
            print('eval', output.name, i + 1, len(names), flush=True)
    value = dict(n=len(rows), summary=totals(rows), rows=rows)
    save(output, value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--variant', choices=['vitb16', 'vitl14'], default='vitl14')
    p.add_argument('--checkpoint', type=Path, required=True, help='Official OVDQUO_ViT-B16_LVIS.pth / ViT-L14_LVIS.pth path')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resolution', type=int, default=672, help='Square input resolution (multiple of 2*patch_size: 672/896 for vitl14, 768/1024 for vitb16)')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--accumulate', type=int, default=4)
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--lr', type=float, default=2.0e-5)
    p.add_argument('--adapter-lr', type=float, default=1.5e-4)
    p.add_argument('--cal-lr', type=float, default=1.5e-3)
    p.add_argument('--max-classes', type=int, default=8)
    p.add_argument('--neg-class-weight', type=float, default=0.5)
    p.add_argument('--absent-prob', type=float, default=0.05, help='Probability of calibrated absent-prompt negative step')
    p.add_argument('--prior-weight', type=float, default=0.10, help='Weight of auxiliary Stage-1 spatial prior loss')
    p.add_argument('--use-exemplars', action='store_true')
    p.add_argument('--grad-checkpoint', action='store_true')
    p.add_argument('--tile-trigger', type=int, default=400)
    p.add_argument('--init', type=Path)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--dev-per-class', type=int, default=8)
    p.add_argument('--pilot-images', type=Path)
    p.add_argument('--exclude-train', type=Path)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / 'manifest.json').exists():
        raise RuntimeError('Use a fresh output directory')

    random.seed(20261122)
    np.random.seed(20261122)
    torch.manual_seed(20261122)
    torch.set_num_threads(4)

    excluded = json.loads(a.exclude_train.read_text()) if a.exclude_train and a.exclude_train.exists() else []
    pilot = json.loads(a.pilot_images.read_text()) if a.pilot_images and a.pilot_images.exists() else []
    train = OVDQUOData(a.data_root, 'train', excluded)
    val = OVDQUOData(a.data_root, 'val')
    groups = {}
    for n in val.names:
        if n not in pilot:
            groups.setdefault(val.classes[n], []).append(n)
    rng = random.Random(5199)
    dev = []
    for c in sorted(groups):
        g = sorted(groups[c])
        rng.shuffle(g)
        dev.extend(g[:a.dev_per_class])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OVDQUOCounting(
        variant=a.variant,
        checkpoint_path=str(a.checkpoint),
        device=device,
        resolution=a.resolution,
        use_exemplars=a.use_exemplars,
        grad_checkpoint=a.grad_checkpoint,
    )
    if a.init and a.init.exists():
        ckpt = torch.load(a.init, map_location='cpu', weights_only=False)
        model.import_trainable_state(ckpt['trainable'] if 'trainable' in ckpt else ckpt)
        print('loaded init checkpoint:', a.init, flush=True)

    trainable = model.trainable_parameters()
    assert trainable, 'No trainable parameters in OVDQUOCounting'
    train_vocab = sorted({train.classes[n] for n in train.names})
    val_vocab = sorted({val.classes[n] for n in val.names})
    # Pre-warm frozen EVA02-CLIP 12-template text cache once before training
    model.encode_text_queries(sorted(set(train_vocab + val_vocab + ['elephant'])), device=device)
    print(f'Pre-cached {len(model._text_cache)} EVA02-CLIP text embeddings', flush=True)

    manifest = dict(
        architecture=model.ARCHITECTURE,
        variant=a.variant,
        resolution=a.resolution,
        num_queries=model.num_queries,
        epochs=a.epochs,
        accumulate=a.accumulate,
        backbone_lr=a.backbone_lr,
        lr=a.lr,
        adapter_lr=a.adapter_lr,
        cal_lr=a.cal_lr,
        max_classes=a.max_classes,
        neg_class_weight=a.neg_class_weight,
        absent_prob=a.absent_prob,
        prior_weight=a.prior_weight,
        use_exemplars=a.use_exemplars,
        modality_dropout='12% absent + 35% text-only / 31% multimodal / 22% exemplar-only' if a.use_exemplars else '12% absent + 88% text-only',
        tile_trigger=a.tile_trigger,
        init=str(a.init) if a.init else None,
        eval_only=a.eval_only,
        trainable_parameters=sum(p.numel() for _, p in trainable),
        nms_free=True,
    )
    save(a.output / 'manifest.json', manifest)

    if a.eval_only:
        res = dict(state='complete', mode='eval_only', init=str(a.init) if a.init else None)
        res['fullval_text_raw'] = evaluate_ov_dquo(
            model, val, val.names, device, a.output / 'eval_fullval_text_raw.json',
            use_exemplars=False, use_text=True, tile_trigger=0)['summary']
        if a.use_exemplars:
            res['fullval_mm_raw'] = evaluate_ov_dquo(
                model, val, val.names, device, a.output / 'eval_fullval_mm_raw.json',
                use_exemplars=True, use_text=True, tile_trigger=0)['summary']
            res['fullval_ex_only_raw'] = evaluate_ov_dquo(
                model, val, val.names, device, a.output / 'eval_fullval_ex_only_raw.json',
                use_exemplars=True, use_text=False, tile_trigger=0)['summary']
            if a.tile_trigger > 0:
                res['fullval_mm_tiled'] = evaluate_ov_dquo(
                    model, val, val.names, device, a.output / 'eval_fullval_mm_tiled.json',
                    use_exemplars=True, use_text=True, tile_trigger=a.tile_trigger)['summary']
        elif a.tile_trigger > 0:
            res['fullval_text_tiled'] = evaluate_ov_dquo(
                model, val, val.names, device, a.output / 'eval_fullval_text_tiled.json',
                use_exemplars=False, use_text=True, tile_trigger=a.tile_trigger)['summary']
        if pilot:
            res['pilot58'] = evaluate_ov_dquo(
                model, val, pilot, device, a.output / 'eval_pilot58.json',
                use_exemplars=a.use_exemplars, use_text=True, tile_trigger=a.tile_trigger)['summary']
        res['absent'] = evaluate_ov_dquo(
            model, val, ['1919.jpg', '3436.jpg'], device, a.output / 'eval_absent.json',
            absent=True, use_exemplars=False, tile_trigger=0)['summary']
        save(a.output / 'result.json', res)
        print('eval_only complete', flush=True)
        return

    criterion = CountingCriterion()
    bb_p, tr_p, ad_p, cal_p = model.parameter_buckets()
    optimizer = torch.optim.AdamW(model.parameter_groups(
        backbone_lr=a.backbone_lr, lr=a.lr, adapter_lr=a.adapter_lr, cal_lr=a.cal_lr, weight_decay=1e-4))
    steps_per_epoch = math.ceil(len(train.names) / a.accumulate)
    scheduler = make_warmup_cosine_scheduler(
        optimizer, total_steps=a.epochs * steps_per_epoch, warmup_steps=min(450, max(1, steps_per_epoch // 2)))
    use_amp = (device == 'cuda') and not a.no_amp
    scaler = torch.amp.GradScaler('cuda', init_scale=256.0, enabled=use_amp)

    def clip_all_buckets():
        for bucket in (bb_p, tr_p, ad_p):
            if bucket:
                torch.nn.utils.clip_grad_norm_(bucket, 1.0)
        if cal_p:
            # Elementwise clip for calibration scalars so huge `clip_text_bias` grads never starve gates!
            torch.nn.utils.clip_grad_value_(cal_p, 1.0)

    if a.smoke:
        model.set_training(True)
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        modes = (
            [('mm', True, True, False), ('ex', False, True, False), ('txt', True, False, False), ('absent', True, False, True)]
            if a.use_exemplars
            else [('txt', True, False, False), ('absent', True, False, True)]
        )
        last_loss = None
        for idx_s, (m_tag, use_txt, want_ex, is_abs) in enumerate(modes):
            sample = train.get(train.names[idx_s], augment=True)
            if is_abs:
                negs = [c for c in train_vocab if c != sample['prompt']]
                abs_p = negs[0] if negs else 'elephant'
                classes, tidx = build_ov_dquo_class_list(abs_p, negs, max_classes=a.max_classes)
                ex_boxes = None
                step_sample = dict(
                    sample,
                    supervision_points=torch.empty((0, 2), dtype=torch.float32),
                    exemplar_indices=torch.empty((0,), dtype=torch.long),
                    exemplar_boxes=torch.empty((0, 4), dtype=torch.float32),
                )
            else:
                classes, tidx = build_ov_dquo_class_list(sample['prompt'], train_vocab, max_classes=a.max_classes) if use_txt else (['object'], 0)
                ex_boxes = sample['exemplar_boxes'] if (want_ex and len(sample['exemplar_boxes'])) else None
                if not use_txt and ex_boxes is None:
                    use_txt = True
                step_sample = sample
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                out = model.forward_stages(
                    step_sample['pil_image'], classes, target_idx=tidx, exemplar_boxes=ex_boxes, use_text=use_txt,
                )
            loss, _ = compute_ov_dquo_7layer_loss(
                criterion, out, step_sample, device, neg_class_weight=a.neg_class_weight, prior_weight=a.prior_weight,
            )
            if is_abs:
                loss = 0.5 * loss
            assert torch.isfinite(loss), f'Non-finite loss in mode {m_tag}: {loss}'
            scaler.scale(loss / float(len(modes))).backward()
            last_loss = float(loss.detach())
        scaler.unscale_(optimizer)

        def _gsum(params):
            return sum(p.grad.abs().sum().item() for p in params if p.grad is not None)

        bb_g = _gsum(bb_p)
        enc_g = _gsum(model.transformer.encoder.parameters())
        dec_g = _gsum(model.transformer.decoder.parameters())
        eb_g = _gsum(model.transformer.enc_out_bbox_embed.parameters())
        ec_g = _gsum(model.transformer.enc_out_class_embed.parameters())
        tp_g = _gsum(model.transformer.text_proj.parameters())
        mf_g = _gsum(model.mem_fusion.parameters())
        etp_g = _gsum(model.enc_tgt_proj.parameters())
        dtp_g = _gsum(model.dec_text_delta.parameters())
        entp_g = _gsum(model.enc_text_delta.parameters())
        ep_g = _gsum([model.enc_prompt_prior, model.enc_prior_bias])
        cal_g = _gsum(cal_p)
        for lbl, val_g in [
            ('bb_neck', bb_g), ('encoder', enc_g), ('decoder', dec_g), ('enc_bbox', eb_g),
            ('enc_cls', ec_g), ('text_proj', tp_g), ('mem_fusion', mf_g),
            ('enc_tgt_proj', etp_g), ('dec_text_delta', dtp_g), ('enc_text_delta', entp_g),
            ('enc_prior', ep_g), ('cal', cal_g),
        ]:
            assert math.isfinite(val_g) and val_g > 0, f'{lbl} bad grad: {val_g}'
        for lvl in range(model.num_dec_layers):
            bg_l = _gsum(model.bbox_embed[lvl].parameters())
            cg_l = _gsum(model.class_embed[lvl].parameters())
            assert math.isfinite(bg_l) and bg_l > 0, f'bbox_embed[{lvl}] bad grad: {bg_l}'
            assert math.isfinite(cg_l) and cg_l > 0, f'class_embed[{lvl}] bad grad: {cg_l}'
        assert all(p.grad is None for p in model.backbone[0].visual.parameters()), 'EVA02 visual must stay frozen'
        ex_msg = ''
        if a.use_exemplars:
            vp_g = _gsum(model.vis_proj.parameters())
            dv_g = _gsum(model.dec_vis_proj.parameters())
            ev_g = _gsum(model.enc_vis_proj.parameters())
            emv_g = _gsum(model.e_mem_vis_proj.parameters())
            si_g = _gsum(model.sim_inject.parameters())
            for lbl, val_g in [('vis_proj', vp_g), ('dec_vis_proj', dv_g), ('enc_vis_proj', ev_g), ('e_mem_vis_proj', emv_g), ('sim_inject', si_g)]:
                assert math.isfinite(val_g) and val_g > 0, f'{lbl} bad grad: {val_g}'
            ex_msg = f' vis_proj={vp_g:.4f} dec_vis={dv_g:.4f} enc_vis={ev_g:.4f} em_vis={emv_g:.4f}'
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        print(
            f'ov-dquo v3.1 smoke grad sums: bb={bb_g:.4f} enc={enc_g:.4f} dec={dec_g:.4f} '
            f'mf={mf_g:.4f} dtxt={dtp_g:.4f} etxt={entp_g:.4f} '
            f'ep={ep_g:.4f} cal={cal_g:.4f}{ex_msg} peak_vram_mb={peak_mb:.1f}',
            flush=True,
        )
        clip_all_buckets()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        model.set_training(False)
        with torch.no_grad():
            b, s = predict_with_tiling(
                model, sample['pil_image'], sample['prompt'],
                exemplar_boxes=sample['exemplar_boxes'] if a.use_exemplars else None, tile_trigger=0,
            )
        assert b.shape == (model.num_queries, 4) and s.shape == (model.num_queries,)
        save(a.output / 'smoke.json', dict(
            passed=True, loss=last_loss, queries=len(b), stages=7,
            resolution=a.resolution, peak_vram_mb=round(peak_mb, 1), use_exemplars=a.use_exemplars,
        ))
        print('ov-dquo smoke passed', last_loss, f'peak_vram_mb={peak_mb:.1f}', flush=True)
        return

    base_dev = evaluate_ov_dquo(model, val, dev, device, a.output / 'baseline_dev.json', use_exemplars=a.use_exemplars, use_text=True)
    print(
        f"zero-shot baseline: dev_mae={base_dev['summary']['0.23']['mae']:.3f} "
        f"best_th_mae={min(v['mae'] for v in base_dev['summary'].values()):.3f}",
        flush=True,
    )

    history = []
    best = float('inf')
    for epoch in range(1, a.epochs + 1):
        model.set_training(True)
        optimizer.zero_grad(set_to_none=True)
        names = list(train.names)
        random.shuffle(names)
        losses, cls_losses, ctr_losses = [], [], []
        for i, name in enumerate(names):
            sample = train.get(name, augment=True)
            is_absent_step = (a.absent_prob > 0.0) and (len(train_vocab) > 1) and (random.random() < a.absent_prob)
            if is_absent_step:
                # v3 Fix #1 & #4: Absent-prompt negative step prevents prompt-reconstruction shortcut
                neg_pool = [c for c in train_vocab if c != sample['prompt']]
                abs_prompt = random.choice(neg_pool)
                classes, tidx = build_ov_dquo_class_list(abs_prompt, neg_pool, max_classes=a.max_classes)
                use_txt, ex_boxes = True, None
                step_sample = dict(
                    sample,
                    supervision_points=torch.empty((0, 2), dtype=torch.float32),
                    exemplar_indices=torch.empty((0,), dtype=torch.long),
                    exemplar_boxes=torch.empty((0, 4), dtype=torch.float32),
                )
            else:
                has_ex = a.use_exemplars and len(sample['exemplar_boxes']) > 0
                if has_ex:
                    r = random.random()
                    if r < 0.40:
                        use_txt, ex_boxes = True, None
                    elif r < 0.75:
                        use_txt, ex_boxes = True, sample['exemplar_boxes']
                    else:
                        use_txt, ex_boxes = False, sample['exemplar_boxes']
                else:
                    use_txt, ex_boxes = True, None

                if use_txt:
                    classes, tidx = build_ov_dquo_class_list(sample['prompt'], train_vocab, max_classes=a.max_classes)
                else:
                    classes, tidx = ['object'], 0
                step_sample = sample

            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                out = model.forward_stages(
                    step_sample['pil_image'], classes, target_idx=tidx, exemplar_boxes=ex_boxes, use_text=use_txt,
                )
            loss, parts = compute_ov_dquo_7layer_loss(
                criterion, out, step_sample, device, neg_class_weight=a.neg_class_weight, prior_weight=a.prior_weight,
            )
            if is_absent_step:
                loss = 0.5 * loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss on {name} (absent={is_absent_step})')
            group_start = (i // a.accumulate) * a.accumulate
            group_size = min(a.accumulate, len(names) - group_start)
            scaler.scale(loss / float(group_size)).backward()
            if (i + 1) % a.accumulate == 0 or i + 1 == len(names):
                scaler.unscale_(optimizer)
                clip_all_buckets()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            losses.append(float(loss.detach()))
            cls_losses.append(float(parts['cls']))
            ctr_losses.append(float(parts['center']))
            if (i + 1) % 200 == 0:
                sg = 0.5 * torch.tanh(model.mem_fusion.sim_gate).item()
                mg = 0.5 * torch.tanh(model.mem_fusion.fuse_gate).item()
                td = model._last_text_delta_norm
                diag = (
                    f'tb {model.clip_text_bias.item():.2f} ts {(F.elu(model.clip_text_scale)+1).item():.1f} '
                    f'td {td:.3f} sg {sg:.3f} mg {mg:.3f} ep {model.enc_prompt_prior.item():.2f}'
                )
                if a.use_exemplars:
                    alpha = 0.25 + 0.5 * torch.sigmoid(model.mm_alpha).item()
                    vg = (1.5 * torch.tanh(model.vis_mem_gate)).item()
                    diag += f' vb {model.clip_vis_bias.item():.2f} vs {(F.elu(model.clip_vis_scale)+1).item():.1f} vg {vg:.3f} mm_a {alpha:.2f}'
                print(
                    'train', epoch, i + 1, len(names),
                    'loss', round(float(np.mean(losses)), 4),
                    'cls', round(float(np.mean(cls_losses)), 4),
                    'ctr', round(float(np.mean(ctr_losses)), 4),
                    'lr', f'{optimizer.param_groups[1]["lr"]:.2e}',
                    diag, flush=True,
                )

        dev_res = evaluate_ov_dquo(
            model, val, dev, device, a.output / f'dev_epoch{epoch:02d}.json',
            use_exemplars=a.use_exemplars, use_text=True, tile_trigger=0,
        )
        ex_mae = None
        txt_mae = None
        if a.use_exemplars:
            ex_res = evaluate_ov_dquo(
                model, val, dev, device, a.output / f'dev_exonly_epoch{epoch:02d}.json',
                use_exemplars=True, use_text=False, tile_trigger=0,
            )
            txt_res = evaluate_ov_dquo(
                model, val, dev, device, a.output / f'dev_text_epoch{epoch:02d}.json',
                use_exemplars=False, use_text=True, tile_trigger=0,
            )
            ex_mae = ex_res['summary']['0.23']['mae']
            txt_mae = txt_res['summary']['0.23']['mae']

        absent_res = evaluate_ov_dquo(
            model, val, ['1919.jpg', '3436.jpg'], device, a.output / f'absent_epoch{epoch:02d}.json',
            absent=True, use_exemplars=False, tile_trigger=0,
        )
        mae = dev_res['summary']['0.23']['mae']
        best_th_mae = min(v['mae'] for v in dev_res['summary'].values())
        absent_mae = absent_res['summary']['0.23']['mae']
        row = dict(
            epoch=epoch,
            train_loss=float(np.mean(losses)),
            dev_mae=mae,
            best_th_mae=best_th_mae,
            dev_ex_only_mae=ex_mae,
            dev_text_mae=txt_mae,
            absent_mae=absent_mae,
            summary=dev_res['summary'],
        )
        history.append(row)
        save(a.output / 'history.json', history)
        ckpt_payload = dict(
            epoch=epoch,
            dev_mae=mae,
            variant=a.variant,
            resolution=a.resolution,
            architecture=model.ARCHITECTURE,
            use_exemplars=a.use_exemplars,
            trainable=model.export_trainable_state(),
        )
        torch.save(ckpt_payload, a.output / 'last.pth')
        # Select best checkpoint on `best_th_mae` (avoids missing optimal threshold shift like v1 ep6 17.42 vs ep4 19.04)
        if best_th_mae < best:
            best = best_th_mae
            torch.save(ckpt_payload, a.output / 'best.pth')
        ex_str = f' ex_only_mae {round(ex_mae, 3)} text_mae {round(txt_mae, 3)}' if ex_mae is not None else ''
        print(
            f'epoch {epoch} dev_mae {round(mae, 3)}{ex_str} best_th_mae {round(best_th_mae, 3)} '
            f'absent {round(absent_mae, 2)} best_th {round(best, 3)}',
            flush=True,
        )

    best_ckpt = torch.load(a.output / 'best.pth', map_location='cpu', weights_only=False)
    model.import_trainable_state(best_ckpt['trainable'])
    res = dict(state='complete', best_epoch=best_ckpt['epoch'], best_dev_mae=best_ckpt['dev_mae'])
    res['fullval_text_raw'] = evaluate_ov_dquo(
        model, val, val.names, device, a.output / 'adapted_fullval_text_raw.json',
        use_exemplars=False, use_text=True, tile_trigger=0)['summary']
    if a.use_exemplars:
        res['fullval_mm_raw'] = evaluate_ov_dquo(
            model, val, val.names, device, a.output / 'adapted_fullval_mm_raw.json',
            use_exemplars=True, use_text=True, tile_trigger=0)['summary']
        res['fullval_ex_only_raw'] = evaluate_ov_dquo(
            model, val, val.names, device, a.output / 'adapted_fullval_ex_only_raw.json',
            use_exemplars=True, use_text=False, tile_trigger=0)['summary']
        if a.tile_trigger > 0:
            res['fullval_mm_tiled'] = evaluate_ov_dquo(
                model, val, val.names, device, a.output / 'adapted_fullval_mm_tiled.json',
                use_exemplars=True, use_text=True, tile_trigger=a.tile_trigger)['summary']
    elif a.tile_trigger > 0:
        res['fullval_text_tiled'] = evaluate_ov_dquo(
            model, val, val.names, device, a.output / 'adapted_fullval_text_tiled.json',
            use_exemplars=False, use_text=True, tile_trigger=a.tile_trigger)['summary']
    if pilot:
        res['pilot58'] = evaluate_ov_dquo(
            model, val, pilot, device, a.output / 'adapted_pilot58.json',
            use_exemplars=a.use_exemplars, use_text=True, tile_trigger=a.tile_trigger)['summary']
    res['absent'] = evaluate_ov_dquo(
        model, val, ['1919.jpg', '3436.jpg'], device, a.output / 'adapted_absent.json',
        absent=True, use_exemplars=False, tile_trigger=0)['summary']
    save(a.output / 'result.json', res)
    print('ov-dquo counting complete', flush=True)


if __name__ == '__main__':
    main()
