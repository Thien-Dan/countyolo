"""Train & evaluate OmDet-Turbo (`omlab/omdet-turbo-swin-tiny-hf`) with 7-layer CountGD-Box
deep supervision, balanced multi-class negative classes, optional Dual-Path ROIAlign visual
exemplar injection, FrozenBN stability, warmup+cosine LR schedule, and adaptive 2x2 tiling.
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
from models.omdet_turbo_counting import OmDetTurboCounting, build_omdet_class_list
from datasets.yoloworld_counting import load_fsc147d_class_names
from tools.train_yoloworld_counting import THRESHOLDS, save, totals
from tools.train_llmdet_counting import compute_7layer_loss, make_warmup_cosine_scheduler


def prepare_omdet_sample(image: Image.Image, record: dict, flip=False, crop_scale_range=None, max_queries=900):
    """Prepare PIL image + normalized [0,1] points/exemplars (OmDetTurboProcessor resizes to 640x640 cleanly)."""
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

    # Optional scale/crop augmentation that preserves all exemplar boxes & keeps valid GT points
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


class OmDetData:
    def __init__(self, root: Path, split: str, exclude=(), max_queries=900):
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
        crop_range = (0.75, 1.0) if (augment and random.random() < 0.5) else None
        sample = prepare_omdet_sample(
            image, record, flip=augment and random.random() < 0.5,
            crop_scale_range=crop_range, max_queries=self.max_queries)
        sample.update(name=name, prompt=self.classes[name])
        return sample


@torch.no_grad()
def predict_with_tiling(model: OmDetTurboCounting, pil_image: Image.Image, prompt: str,
                        exemplar_boxes=None, threshold: float = 0.23, tile_trigger: int = 600):
    """Run NMS-free OmDet-Turbo inference with optional 2x2 adaptive quadrant tiling."""
    use_amp = torch.cuda.is_available()
    with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
        boxes, logits = model(pil_image, prompt, exemplar_boxes=exemplar_boxes)
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
            qb, ql = model(crop, prompt, exemplar_feats=ex_feats)
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
def evaluate_omdet(model: OmDetTurboCounting, dataset: OmDetData, names, device: str, output: Path,
                   absent=False, use_exemplars=False, tile_trigger=0):
    model.set_training(False)
    rows = []
    for i, name in enumerate(names):
        sample = dataset.get(name, augment=False)
        prompt = 'elephant' if absent else sample['prompt']
        ex_boxes = sample['exemplar_boxes'] if (use_exemplars and not absent and len(sample['exemplar_boxes'])) else None
        boxes, scores = predict_with_tiling(
            model, sample['pil_image'], prompt, exemplar_boxes=ex_boxes, tile_trigger=tile_trigger)
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
    p.add_argument('--model-id', type=str, default='omlab/omdet-turbo-swin-tiny-hf')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--accumulate', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-5, help='Gentle LR for pretrained OmDet-Turbo encoder/decoder')
    p.add_argument('--exemplar-lr', type=float, default=1.5e-4, help='Higher LR for newly initialized exemplar modules')
    p.add_argument('--max-classes', type=int, default=8, help='Classes per forward (1 pos + K-1 neg)')
    p.add_argument('--neg-class-weight', type=float, default=1.0, help='Balanced total weight on negative classes')
    p.add_argument('--use-exemplars', action='store_true', help='Enable Dual-Path ROIAlign visual exemplar injection')
    p.add_argument('--exemplar-prob', type=float, default=0.5, help='Modality dropout probability for exemplars')
    p.add_argument('--tile-trigger', type=int, default=600, help='Trigger 2x2 adaptive tiling when raw count >= trigger')
    p.add_argument('--init', type=Path, help='Optional checkpoint (best.pth) to initialize weights from')
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

    random.seed(20261121)
    np.random.seed(20261121)
    torch.manual_seed(20261121)
    torch.set_num_threads(4)

    excluded = json.loads(a.exclude_train.read_text()) if a.exclude_train and a.exclude_train.exists() else []
    pilot = json.loads(a.pilot_images.read_text()) if a.pilot_images and a.pilot_images.exists() else []
    train = OmDetData(a.data_root, 'train', excluded)
    val = OmDetData(a.data_root, 'val')
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
    model = OmDetTurboCounting(a.model_id, device=device, freeze_backbones=True, use_exemplars=a.use_exemplars)
    if a.init and a.init.exists():
        ckpt = torch.load(a.init, map_location='cpu', weights_only=False)
        model.import_trainable_state(ckpt['trainable'] if 'trainable' in ckpt else ckpt)
        print('loaded init checkpoint:', a.init, flush=True)

    trainable = model.trainable_parameters()
    assert trainable, 'No trainable parameters in OmDetTurboCounting'
    train_vocab = sorted({train.classes[n] for n in train.names})

    manifest = dict(
        architecture=model.ARCHITECTURE,
        model_id=a.model_id,
        epochs=a.epochs,
        accumulate=a.accumulate,
        lr=a.lr,
        exemplar_lr=a.exemplar_lr,
        max_classes=a.max_classes,
        neg_class_weight=a.neg_class_weight,
        use_exemplars=a.use_exemplars,
        exemplar_prob=a.exemplar_prob,
        tile_trigger=a.tile_trigger,
        init=str(a.init) if a.init else None,
        eval_only=a.eval_only,
        trainable_parameters=sum(p.numel() for _, p in trainable),
        supervision='Differentiable 7-layer CountGD-Box deep supervision on RT-DETR HybridEncoder + 6-layer Task-Conditioned Decoder',
        fsc147d_prompts=True,
        nms_free=True,
    )
    save(a.output / 'manifest.json', manifest)

    if a.eval_only:
        res = dict(state='complete', mode='eval_only', init=str(a.init) if a.init else None)
        res['fullval_text_raw'] = evaluate_omdet(
            model, val, val.names, device, a.output / 'eval_fullval_text_raw.json',
            use_exemplars=False, tile_trigger=0)['summary']
        if a.tile_trigger > 0:
            res['fullval_text_tiled'] = evaluate_omdet(
                model, val, val.names, device, a.output / 'eval_fullval_text_tiled.json',
                use_exemplars=False, tile_trigger=a.tile_trigger)['summary']
        if a.use_exemplars:
            res['fullval_ex_raw'] = evaluate_omdet(
                model, val, val.names, device, a.output / 'eval_fullval_ex_raw.json',
                use_exemplars=True, tile_trigger=0)['summary']
            if a.tile_trigger > 0:
                res['fullval_ex_tiled'] = evaluate_omdet(
                    model, val, val.names, device, a.output / 'eval_fullval_ex_tiled.json',
                    use_exemplars=True, tile_trigger=a.tile_trigger)['summary']
        if pilot:
            res['pilot58'] = evaluate_omdet(
                model, val, pilot, device, a.output / 'eval_pilot58.json',
                use_exemplars=a.use_exemplars, tile_trigger=a.tile_trigger)['summary']
        res['absent'] = evaluate_omdet(
            model, val, ['1919.jpg', '3436.jpg'], device, a.output / 'eval_absent.json',
            absent=True, use_exemplars=False, tile_trigger=0)['summary']
        save(a.output / 'result.json', res)
        print('eval_only complete', flush=True)
        return

    criterion = CountingCriterion()
    optimizer = torch.optim.AdamW(model.parameter_groups(base_lr=a.lr, exemplar_lr=a.exemplar_lr, weight_decay=1e-4))
    steps_per_epoch = math.ceil(len(train.names) / a.accumulate)
    scheduler = make_warmup_cosine_scheduler(optimizer, total_steps=a.epochs * steps_per_epoch, warmup_steps=min(450, steps_per_epoch // 2))
    use_amp = (device == 'cuda') and not a.no_amp
    scaler = torch.amp.GradScaler('cuda', init_scale=256.0, enabled=use_amp)

    if a.smoke:
        model.set_training(True)
        optimizer.zero_grad(set_to_none=True)
        for name in train.names[:2]:
            sample = train.get(name, augment=True)
            classes, task, target_idx = build_omdet_class_list(sample['prompt'], train_vocab, max_classes=a.max_classes)
            inputs = model.processor(images=sample['pil_image'], text=classes, task=task, return_tensors='pt').to(device)
            ex_boxes = sample['exemplar_boxes'] if (a.use_exemplars and len(sample['exemplar_boxes'])) else None
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                st = model.forward_stages(
                    inputs['pixel_values'],
                    inputs['classes_input_ids'],
                    inputs['classes_attention_mask'],
                    inputs['tasks_input_ids'],
                    inputs['tasks_attention_mask'],
                    inputs['classes_structure'],
                    target_idx=target_idx,
                    exemplar_boxes=ex_boxes,
                )
            loss, _ = compute_7layer_loss(criterion, st, sample, device, neg_class_weight=a.neg_class_weight)
            assert torch.isfinite(loss), f'Non-finite loss: {loss}'
            scaler.scale(loss / 2.0).backward()
        scaler.unscale_(optimizer)
        enc_g = sum(p.grad.abs().sum().item() for p in model.detector.encoder.parameters() if p.grad is not None)
        dec_b_g = sum(p.grad.abs().sum().item() for p in model.detector.decoder.decoder_bbox_head.parameters() if p.grad is not None)
        dec_c_g = sum(p.grad.abs().sum().item() for p in model.detector.decoder.decoder_class_head.parameters() if p.grad is not None)
        enc_b_g = sum(p.grad.abs().sum().item() for p in model.detector.decoder.encoder_bbox_head.parameters() if p.grad is not None)
        assert math.isfinite(enc_g) and enc_g > 0, f'encoder bad grad: {enc_g}'
        assert math.isfinite(dec_b_g) and dec_b_g > 0, f'decoder_bbox_head bad grad: {dec_b_g}'
        assert math.isfinite(dec_c_g) and dec_c_g > 0, f'decoder_class_head bad grad: {dec_c_g}'
        assert math.isfinite(enc_b_g) and enc_b_g > 0, f'encoder_bbox_head bad grad: {enc_b_g}'
        ex_msg = ''
        if a.use_exemplars:
            ex_p_g = sum(p.grad.abs().sum().item() for p in model.exemplar_proj.parameters() if p.grad is not None)
            ex_pr_g = sum(p.grad.abs().sum().item() for p in model.exemplar_proto_head.parameters() if p.grad is not None)
            ex_gt_g = model.exemplar_cls_gate.grad.abs().sum().item() if model.exemplar_cls_gate.grad is not None else 0.0
            assert math.isfinite(ex_p_g) and ex_p_g > 0, f'exemplar_proj bad grad: {ex_p_g}'
            assert math.isfinite(ex_pr_g) and ex_pr_g > 0, f'exemplar_proto_head bad grad: {ex_pr_g}'
            assert math.isfinite(ex_gt_g) and ex_gt_g > 0, f'exemplar_cls_gate bad grad: {ex_gt_g}'
            ex_msg = f' ex_proj={ex_p_g:.4f} ex_proto={ex_pr_g:.4f} ex_gate={ex_gt_g:.4f}'
        print(f'omdet smoke grad sums: enc={enc_g:.4f} enc_bbox={enc_b_g:.4f} dec_bbox={dec_b_g:.4f} dec_cls={dec_c_g:.4f}{ex_msg}', flush=True)
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 0.1)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        model.set_training(False)
        with torch.no_grad():
            b, s = predict_with_tiling(
                model, sample['pil_image'], sample['prompt'],
                exemplar_boxes=sample['exemplar_boxes'] if a.use_exemplars else None, tile_trigger=0)
        assert b.shape == (model.num_queries, 4) and s.shape == (model.num_queries,)
        save(a.output / 'smoke.json', dict(passed=True, loss=float(loss.detach()), queries=len(b), stages=7,
                                           max_classes=a.max_classes, use_exemplars=a.use_exemplars))
        print('omdet-turbo smoke passed', float(loss.detach()), flush=True)
        return

    evaluate_omdet(model, val, dev, device, a.output / 'baseline_dev.json', use_exemplars=a.use_exemplars)
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
            classes, task, target_idx = build_omdet_class_list(sample['prompt'], train_vocab, max_classes=a.max_classes)
            inputs = model.processor(images=sample['pil_image'], text=classes, task=task, return_tensors='pt').to(device)
            inject_ex = a.use_exemplars and len(sample['exemplar_boxes']) > 0 and (random.random() < a.exemplar_prob)
            ex_boxes = sample['exemplar_boxes'] if inject_ex else None
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                st = model.forward_stages(
                    inputs['pixel_values'],
                    inputs['classes_input_ids'],
                    inputs['classes_attention_mask'],
                    inputs['tasks_input_ids'],
                    inputs['tasks_attention_mask'],
                    inputs['classes_structure'],
                    target_idx=target_idx,
                    exemplar_boxes=ex_boxes,
                )
            loss, parts = compute_7layer_loss(criterion, st, sample, device, neg_class_weight=a.neg_class_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss on {name}')
            group_start = (i // a.accumulate) * a.accumulate
            group_size = min(a.accumulate, len(names) - group_start)
            scaler.scale(loss / group_size).backward()
            if (i + 1) % a.accumulate == 0 or i + 1 == len(names):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 0.1)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            cls_losses.append(float(parts['cls']))
            ctr_losses.append(float(parts['center']))
            del st, loss, inputs
            if (i + 1) % 200 == 0:
                torch.cuda.empty_cache()
                gate_val = float(torch.tanh(model.exemplar_cls_gate).item()) if a.use_exemplars else 0.0
                save(a.output / 'progress.json', dict(epoch=epoch, images=i + 1, total=len(names),
                                                      mean_loss=float(np.mean(losses)),
                                                      cls_loss=float(np.mean(cls_losses)),
                                                      center_loss=float(np.mean(ctr_losses)),
                                                      lr=optimizer.param_groups[0]['lr'],
                                                      ex_gate=gate_val))
                g_str = f' gate {gate_val:.3f}' if a.use_exemplars else ''
                print(f"train {epoch} {i+1} {len(names)} loss {round(float(np.mean(losses)), 4)} "
                      f"cls {round(float(np.mean(cls_losses)), 4)} ctr {round(float(np.mean(ctr_losses)), 4)} "
                      f"lr {optimizer.param_groups[0]['lr']:.2e}{g_str}", flush=True)

        res = evaluate_omdet(model, val, dev, device, a.output / f'dev_epoch{epoch:02d}.json', use_exemplars=a.use_exemplars)
        res_abs = evaluate_omdet(model, val, ['1919.jpg', '3436.jpg'], device, a.output / f'absent_epoch{epoch:02d}.json',
                                 absent=True, use_exemplars=False)
        mae = res['summary']['0.23']['mae']
        best_th_mae = min(res['summary'][str(t)]['mae'] for t in THRESHOLDS)
        absent_mae = res_abs['summary']['0.23']['mae']
        entry = dict(epoch=epoch, loss=float(np.mean(losses)), cls_loss=float(np.mean(cls_losses)),
                     center_loss=float(np.mean(ctr_losses)), dev_mae=mae, dev_best_th_mae=best_th_mae,
                     absent_mae=absent_mae)
        if a.use_exemplars:
            res_txt = evaluate_omdet(model, val, dev, device, a.output / f'dev_text_epoch{epoch:02d}.json', use_exemplars=False)
            entry['dev_text_mae'] = res_txt['summary']['0.23']['mae']
            entry['ex_gate'] = float(torch.tanh(model.exemplar_cls_gate).item())
        history.append(entry)
        save(a.output / 'history.json', history)
        torch.save(dict(trainable=model.export_trainable_state(), epoch=epoch, dev_mae=mae, manifest=manifest), a.output / 'last.pth')
        if mae < best:
            best = mae
            torch.save(dict(trainable=model.export_trainable_state(), epoch=epoch, dev_mae=best, manifest=manifest), a.output / 'best.pth')
        extra_str = f" text_mae {round(entry['dev_text_mae'], 3)}" if 'dev_text_mae' in entry else ''
        print(f"epoch {epoch} dev_mae {round(mae, 3)}{extra_str} best_th_mae {round(best_th_mae, 3)} "
              f"absent {round(absent_mae, 1)} best {round(best, 3)}", flush=True)

    best_state = torch.load(a.output / 'best.pth', map_location='cpu', weights_only=False)
    model.import_trainable_state(best_state['trainable'])
    final_summary = dict(state='complete', best_epoch=best_state['epoch'], best_dev_mae=best)
    final_summary['fullval_raw'] = evaluate_omdet(
        model, val, val.names, device, a.output / 'adapted_fullval_raw.json',
        use_exemplars=a.use_exemplars, tile_trigger=0)['summary']
    if a.tile_trigger > 0:
        final_summary['fullval_tiled'] = evaluate_omdet(
            model, val, val.names, device, a.output / 'adapted_fullval_tiled.json',
            use_exemplars=a.use_exemplars, tile_trigger=a.tile_trigger)['summary']
    if a.use_exemplars:
        final_summary['fullval_text_only'] = evaluate_omdet(
            model, val, val.names, device, a.output / 'adapted_fullval_text_only.json',
            use_exemplars=False, tile_trigger=a.tile_trigger)['summary']
    if pilot:
        final_summary['pilot58'] = evaluate_omdet(
            model, val, pilot, device, a.output / 'adapted_pilot58.json',
            use_exemplars=a.use_exemplars, tile_trigger=a.tile_trigger)['summary']
    final_summary['absent'] = evaluate_omdet(
        model, val, ['1919.jpg', '3436.jpg'], device, a.output / 'adapted_absent.json',
        absent=True, use_exemplars=False, tile_trigger=0)['summary']
    save(a.output / 'result.json', final_summary)


if __name__ == '__main__':
    main()
