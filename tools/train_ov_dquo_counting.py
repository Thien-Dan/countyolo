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
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from models.yoloworld_counting import CountingCriterion, xywh_to_xyxy, exemplar_point_indices
from models.ov_dquo_counting import OVDQUOCounting, build_ov_dquo_class_list
from datasets.yoloworld_counting import load_fsc147d_class_names
from tools.train_yoloworld_counting import THRESHOLDS, save, totals
from tools.train_llmdet_counting import make_warmup_cosine_scheduler


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


def _stage_loss(criterion: CountingCriterion, st: dict, pts, ex_idx, ex_xyxy, target_idx: int,
                device: str, neg_class_weight: float):
    boxes = st['boxes']
    has_text = st['text_logits'] is not None
    has_vis = st['vis_logits'] is not None
    if has_text and has_vis:
        k_cls = st['text_logits'].shape[1]
        t_mask = torch.zeros(k_cls, device=device, dtype=torch.bool)
        t_mask[target_idx] = True
        loss_mm, p_mm = criterion(boxes, st['mm_logits'], pts, ex_idx, ex_xyxy)
        loss_txt, p_txt = criterion(
            boxes, st['text_logits'], pts, ex_idx, ex_xyxy,
            target_tokens=t_mask, neg_class_weight=neg_class_weight,
        )
        loss_vis, p_vis = criterion(boxes, st['vis_logits'], pts, ex_idx, ex_xyxy)
        total = 0.40 * loss_mm + 0.35 * loss_txt + 0.25 * loss_vis
        parts = dict(
            cls=0.40 * p_mm['cls'] + 0.35 * p_txt['cls'] + 0.25 * p_vis['cls'],
            center=0.40 * p_mm['center'] + 0.35 * p_txt['center'] + 0.25 * p_vis['center'],
        )
        return total, parts
    elif has_text:
        k_cls = st['text_logits'].shape[1]
        t_mask = torch.zeros(k_cls, device=device, dtype=torch.bool)
        t_mask[target_idx] = True
        return criterion(
            boxes, st['text_logits'], pts, ex_idx, ex_xyxy,
            target_tokens=t_mask, neg_class_weight=neg_class_weight,
        )
    else:
        return criterion(boxes, st['vis_logits'], pts, ex_idx, ex_xyxy)


def compute_ov_dquo_7layer_loss(criterion: CountingCriterion, out: dict, sample: dict, device: str,
                                aux_weight: float = 1.0, neg_class_weight: float = 0.5):
    """Differentiable 7-layer CountGD deep supervision across final + 5 aux decoder + 1 encoder stage."""
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
    return total / count, parts_final


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
            ex_feats = model.encode_exemplars(pil_image, exemplar_boxes)

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
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--accumulate', type=int, default=4)
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--lr', type=float, default=2.5e-5)
    p.add_argument('--adapter-lr', type=float, default=1.5e-4)
    p.add_argument('--cal-lr', type=float, default=1.5e-3)
    p.add_argument('--max-classes', type=int, default=8)
    p.add_argument('--neg-class-weight', type=float, default=0.5)
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
        use_exemplars=a.use_exemplars,
        modality_dropout='40% text-only / 35% multimodal / 25% exemplar-only' if a.use_exemplars else '100% text-only',
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
        for bucket in (bb_p, tr_p, ad_p, cal_p):
            if bucket:
                torch.nn.utils.clip_grad_norm_(bucket, 1.0)

    if a.smoke:
        model.set_training(True)
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        modes = [('mm', True, True), ('ex', False, True), ('txt', True, False)] if a.use_exemplars else [('txt', True, False), ('txt', True, False)]
        last_loss = None
        for idx_s, (m_tag, use_txt, want_ex) in enumerate(modes):
            sample = train.get(train.names[idx_s], augment=True)
            classes, tidx = build_ov_dquo_class_list(sample['prompt'], train_vocab, max_classes=a.max_classes) if use_txt else (['object'], 0)
            ex_boxes = sample['exemplar_boxes'] if (want_ex and len(sample['exemplar_boxes'])) else None
            if not use_txt and ex_boxes is None:
                use_txt = True
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                out = model.forward_stages(
                    sample['pil_image'], classes, target_idx=tidx, exemplar_boxes=ex_boxes, use_text=use_txt,
                )
            loss, _ = compute_ov_dquo_7layer_loss(criterion, out, sample, device, neg_class_weight=a.neg_class_weight)
            assert torch.isfinite(loss), f'Non-finite loss in mode {m_tag}: {loss}'
            scaler.scale(loss / float(len(modes))).backward()
            last_loss = float(loss.detach())
        scaler.unscale_(optimizer)

        bb_g = sum(p.grad.abs().sum().item() for p in bb_p if p.grad is not None)
        enc_g = sum(p.grad.abs().sum().item() for p in model.transformer.encoder.parameters() if p.grad is not None)
        dec_g = sum(p.grad.abs().sum().item() for p in model.transformer.decoder.parameters() if p.grad is not None)
        eb_g = sum(p.grad.abs().sum().item() for p in model.transformer.enc_out_bbox_embed.parameters() if p.grad is not None)
        ec_g = sum(p.grad.abs().sum().item() for p in model.transformer.enc_out_class_embed.parameters() if p.grad is not None)
        tp_g = sum(p.grad.abs().sum().item() for p in model.transformer.text_proj.parameters() if p.grad is not None)
        mf_g = sum(p.grad.abs().sum().item() for p in model.mem_fusion.parameters() if p.grad is not None)
        ep_g = float(model.enc_prompt_prior.grad.abs().sum().item()) if model.enc_prompt_prior.grad is not None else 0.0
        cal_g = sum(p.grad.abs().sum().item() for p in cal_p if p.grad is not None)
        for lbl, val_g in [('bb_neck', bb_g), ('encoder', enc_g), ('decoder', dec_g), ('enc_bbox', eb_g), ('enc_cls', ec_g), ('text_proj', tp_g), ('mem_fusion', mf_g), ('enc_prior', ep_g), ('cal', cal_g)]:
            assert math.isfinite(val_g) and val_g > 0, f'{lbl} bad grad: {val_g}'
        for lvl in range(model.num_dec_layers):
            bg_l = sum(p.grad.abs().sum().item() for p in model.bbox_embed[lvl].parameters() if p.grad is not None)
            cg_l = sum(p.grad.abs().sum().item() for p in model.class_embed[lvl].parameters() if p.grad is not None)
            assert math.isfinite(bg_l) and bg_l > 0, f'bbox_embed[{lvl}] bad grad: {bg_l}'
            assert math.isfinite(cg_l) and cg_l > 0, f'class_embed[{lvl}] bad grad: {cg_l}'
        assert all(p.grad is None for p in model.backbone[0].visual.parameters()), 'EVA02 visual must stay frozen'
        ex_msg = ''
        if a.use_exemplars:
            vp_g = sum(p.grad.abs().sum().item() for p in model.vis_proj.parameters() if p.grad is not None)
            dv_g = sum(p.grad.abs().sum().item() for p in model.dec_vis_proj.parameters() if p.grad is not None)
            si_g = sum(p.grad.abs().sum().item() for p in model.sim_inject.parameters() if p.grad is not None)
            assert math.isfinite(vp_g) and vp_g > 0, f'vis_proj bad grad: {vp_g}'
            assert math.isfinite(dv_g) and dv_g > 0, f'dec_vis_proj bad grad: {dv_g}'
            assert math.isfinite(si_g) and si_g > 0, f'sim_inject bad grad: {si_g}'
            ex_msg = f' vis_proj={vp_g:.4f} dec_vis={dv_g:.4f} sim_inj={si_g:.4f}'
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        print(
            f'ov-dquo v2 smoke grad sums: bb={bb_g:.4f} enc={enc_g:.4f} dec={dec_g:.4f} '
            f'mem_fus={mf_g:.4f} enc_pr={ep_g:.4f} tproj={tp_g:.4f} cal={cal_g:.4f}{ex_msg} peak_vram_mb={peak_mb:.1f}',
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

            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                out = model.forward_stages(
                    sample['pil_image'], classes, target_idx=tidx, exemplar_boxes=ex_boxes, use_text=use_txt,
                )
            loss, parts = compute_ov_dquo_7layer_loss(criterion, out, sample, device, neg_class_weight=a.neg_class_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss on {name}')
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
                mg = 0.5 * torch.tanh(model.mem_fusion.fuse_gate).item()
                diag = f'tb {model.clip_text_bias.item():.2f} ts {(torch.nn.functional.elu(model.clip_text_scale)+1).item():.1f} mg {mg:.3f} ep {model.enc_prompt_prior.item():.2f}'
                if a.use_exemplars:
                    alpha = 0.25 + 0.5 * torch.sigmoid(model.mm_alpha).item()
                    vm = torch.softmax(model.vis_mix, dim=0).tolist()
                    diag += f' vb {model.clip_vis_bias.item():.2f} vs {(torch.nn.functional.elu(model.clip_vis_scale)+1).item():.1f} vd {vm[2]:.2f} mm_a {alpha:.2f}'
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
        if mae < best:
            best = mae
            torch.save(ckpt_payload, a.output / 'best.pth')
        ex_str = f' ex_only_mae {round(ex_mae, 3)} text_mae {round(txt_mae, 3)}' if ex_mae is not None else ''
        print(
            f'epoch {epoch} dev_mae {round(mae, 3)}{ex_str} best_th_mae {round(best_th_mae, 3)} '
            f'absent {round(absent_mae, 2)} best {round(best, 3)}',
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
