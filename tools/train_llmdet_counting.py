"""Train & evaluate LLMDet (`iSEE-Laboratory/llmdet_tiny` / `llmdet_base`) with 7-layer CountGD-Box
deep supervision, single-forward multi-class negative captioning, optional ROIAlign visual exemplars,
and adaptive 2x2 dense-image tiling (100% NMS-free).
"""
import argparse
import json
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
from models.llmdet_counting import LLMDetCounting, build_multiclass_caption
from datasets.yoloworld_counting import load_fsc147d_class_names
from tools.train_yoloworld_counting import THRESHOLDS, save, totals


def prepare_llmdet_sample(image: Image.Image, record: dict, short=640, max_size=1100, flip=False, max_queries=900):
    """Prepare PIL image and normalized [0,1] supervision targets without YOLO stride-32 grey padding."""
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

    ratio = min(short / min(w, h), max_size / max(w, h))
    nw, nh = max(32, round(w * ratio)), max(32, round(h * ratio))
    image = image.resize((nw, nh), Image.Resampling.BILINEAR)

    if len(points):
        points = points / np.array([w, h], dtype=np.float32)
        supervision_points = supervision_points / np.array([w, h], dtype=np.float32)
    if len(boxes):
        boxes = boxes / np.array([w, h, w, h], dtype=np.float32)

    # Deformable DETR has 900 queries; subsample >900 GT points in train while keeping exemplar targets.
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


class LLMDetData:
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
        short = random.choice([448, 512, 576, 640, 704]) if augment else 640
        sample = prepare_llmdet_sample(
            image, record, short=short, max_size=1100,
            flip=augment and random.random() < 0.5, max_queries=self.max_queries)
        sample.update(name=name, prompt=self.classes[name])
        return sample


def compute_7layer_loss(criterion: CountingCriterion, stage_out: dict, sample: dict, device: str, aux_weight=1.0):
    """Compute final decoder loss + 5 intermediate decoder losses + 1 encoder proposal loss."""
    sup_pts = sample['supervision_points'].to(device)
    ex_idx = sample['exemplar_indices'].to(device)
    ex_box = sample['exemplar_boxes'].to(device)
    t_mask = stage_out['target_tokens'].to(device)

    total_loss, parts = criterion(stage_out['boxes'], stage_out['logits'], sup_pts, ex_idx, ex_box, target_tokens=t_mask)
    aux_count = 0
    for abox, alog in zip(stage_out['aux_boxes'], stage_out['aux_logits']):
        l_aux, _ = criterion(abox, alog, sup_pts, ex_idx, ex_box, target_tokens=t_mask)
        total_loss = total_loss + aux_weight * l_aux
        aux_count += 1
    if stage_out['enc_boxes'] is not None and stage_out['enc_logits'] is not None:
        l_enc, _ = criterion(stage_out['enc_boxes'], stage_out['enc_logits'], sup_pts, ex_idx, ex_box, target_tokens=t_mask)
        total_loss = total_loss + aux_weight * l_enc
        aux_count += 1
    return total_loss / (1.0 + aux_weight * aux_count), parts


@torch.no_grad()
def predict_with_tiling(model: LLMDetCounting, pil_image: Image.Image, prompt: str,
                        exemplar_boxes=None, threshold: float = 0.23, tile_trigger: int = 600):
    """Run NMS-free inference with optional 2x2 adaptive quadrant tiling for dense images (>= tile_trigger)."""
    use_amp = torch.cuda.is_available()
    with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
        boxes, logits = model(pil_image, prompt, exemplar_boxes=exemplar_boxes)
    boxes = boxes.float().cpu()
    scores = logits.float().sigmoid().cpu()
    if tile_trigger <= 0 or int((scores > threshold).sum().item()) < tile_trigger:
        return boxes, scores

    # Dense image detected: split into 4 quadrants and reuse global exemplar embeddings if enabled.
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
        cw, ch = crop.size
        ratio = min(640 / min(cw, ch), 1100 / max(cw, ch))
        crop = crop.resize((max(32, round(cw * ratio)), max(32, round(ch * ratio))), Image.Resampling.BILINEAR)
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
def evaluate_llmdet(model: LLMDetCounting, dataset: LLMDetData, names, device: str, output: Path,
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
    p.add_argument('--model-id', type=str, default='iSEE-Laboratory/llmdet_tiny')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=14)
    p.add_argument('--step-size', type=int, default=9)
    p.add_argument('--accumulate', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max-classes', type=int, default=40, help='Total classes per single-forward caption (1 pos + K-1 neg)')
    p.add_argument('--use-exemplars', action='store_true', help='Enable ROIAlign visual exemplar injection')
    p.add_argument('--exemplar-prob', type=float, default=0.7, help='Probability of injecting exemplars during training')
    p.add_argument('--tile-trigger', type=int, default=600, help='Trigger 2x2 adaptive tiling when raw count >= trigger (0 disables)')
    p.add_argument('--init', type=Path, help='Optional checkpoint (best.pth) to initialize weights from')
    p.add_argument('--eval-only', action='store_true', help='Skip training and run full evaluation on --init checkpoint')
    p.add_argument('--no-amp', action='store_true', help='Disable fp16 autocast')
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
    train = LLMDetData(a.data_root, 'train', excluded)
    val = LLMDetData(a.data_root, 'val')
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
    model = LLMDetCounting(a.model_id, device=device, freeze_backbones=True, use_exemplars=a.use_exemplars)
    if a.init and a.init.exists():
        ckpt = torch.load(a.init, map_location='cpu', weights_only=False)
        model.import_trainable_state(ckpt['trainable'] if 'trainable' in ckpt else ckpt)
        print('loaded init checkpoint:', a.init, flush=True)

    trainable = model.trainable_parameters()
    assert trainable, 'No trainable parameters in LLMDetCounting'
    train_vocab = sorted({train.classes[n] for n in train.names})

    manifest = dict(
        architecture=model.ARCHITECTURE,
        model_id=a.model_id,
        epochs=a.epochs,
        step_size=a.step_size,
        accumulate=a.accumulate,
        lr=a.lr,
        max_classes=a.max_classes,
        use_exemplars=a.use_exemplars,
        exemplar_prob=a.exemplar_prob,
        tile_trigger=a.tile_trigger,
        init=str(a.init) if a.init else None,
        eval_only=a.eval_only,
        trainable_parameters=sum(p.numel() for _, p in trainable),
        supervision='7-layer CountGD-Box deep supervision (1 Encoder + 5 Aux Decoder + 1 Final Decoder)',
        fsc147d_prompts=True,
        nms_free=True,
    )
    save(a.output / 'manifest.json', manifest)

    if a.eval_only:
        res = dict(state='complete', mode='eval_only', init=str(a.init) if a.init else None)
        res['fullval_text_raw'] = evaluate_llmdet(
            model, val, val.names, device, a.output / 'eval_fullval_text_raw.json',
            use_exemplars=False, tile_trigger=0)['summary']
        if a.tile_trigger > 0:
            res['fullval_text_tiled'] = evaluate_llmdet(
                model, val, val.names, device, a.output / 'eval_fullval_text_tiled.json',
                use_exemplars=False, tile_trigger=a.tile_trigger)['summary']
        if a.use_exemplars:
            res['fullval_ex_raw'] = evaluate_llmdet(
                model, val, val.names, device, a.output / 'eval_fullval_ex_raw.json',
                use_exemplars=True, tile_trigger=0)['summary']
            if a.tile_trigger > 0:
                res['fullval_ex_tiled'] = evaluate_llmdet(
                    model, val, val.names, device, a.output / 'eval_fullval_ex_tiled.json',
                    use_exemplars=True, tile_trigger=a.tile_trigger)['summary']
        if pilot:
            res['pilot58'] = evaluate_llmdet(
                model, val, pilot, device, a.output / 'eval_pilot58.json',
                use_exemplars=a.use_exemplars, tile_trigger=a.tile_trigger)['summary']
        res['absent'] = evaluate_llmdet(
            model, val, ['1919.jpg', '3436.jpg'], device, a.output / 'eval_absent.json',
            absent=True, use_exemplars=False, tile_trigger=0)['summary']
        save(a.output / 'result.json', res)
        print('eval_only complete', json.dumps({k: v.get('0.23') if isinstance(v, dict) and '0.23' in v else v for k, v in res.items()}), flush=True)
        return

    criterion = CountingCriterion()
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=a.step_size, gamma=0.1)
    use_amp = (device == 'cuda') and not a.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if a.smoke:
        model.set_training(True)
        optimizer.zero_grad(set_to_none=True)
        for name in train.names[:2]:
            sample = train.get(name, augment=True)
            caption, target_idx = build_multiclass_caption(sample['prompt'], train_vocab, max_classes=a.max_classes)
            inputs = model.processor(images=sample['pil_image'], text=caption, do_resize=False, return_tensors='pt').to(device)
            ex_boxes = sample['exemplar_boxes'] if (a.use_exemplars and len(sample['exemplar_boxes'])) else None
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                st = model.forward_stages(
                    inputs['pixel_values'], inputs.get('pixel_mask'),
                    inputs['input_ids'], inputs['attention_mask'], inputs.get('token_type_ids'),
                    target_phrase_idx=target_idx, exemplar_boxes=ex_boxes)
            loss, _ = compute_7layer_loss(criterion, st, sample, device)
            assert torch.isfinite(loss)
            scaler.scale(loss / 2.0).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 0.1)
        scaler.step(optimizer)
        scaler.update()
        model.set_training(False)
        with torch.no_grad():
            b, s = predict_with_tiling(model, sample['pil_image'], sample['prompt'],
                                       exemplar_boxes=sample['exemplar_boxes'] if a.use_exemplars else None,
                                       tile_trigger=0)
        assert b.shape == (900, 4) and s.shape == (900,)
        save(a.output / 'smoke.json', dict(passed=True, loss=float(loss.detach()), queries=len(b), stages=7,
                                           max_classes=a.max_classes, use_exemplars=a.use_exemplars))
        print('llmdet smoke passed', float(loss.detach()), flush=True)
        return

    evaluate_llmdet(model, val, dev, device, a.output / 'baseline_dev.json', use_exemplars=a.use_exemplars)
    history = []
    best = float('inf')
    for epoch in range(1, a.epochs + 1):
        model.set_training(True)
        optimizer.zero_grad(set_to_none=True)
        names = list(train.names)
        random.shuffle(names)
        losses = []
        for i, name in enumerate(names):
            sample = train.get(name, augment=True)
            caption, target_idx = build_multiclass_caption(sample['prompt'], train_vocab, max_classes=a.max_classes)
            inputs = model.processor(images=sample['pil_image'], text=caption, do_resize=False, return_tensors='pt').to(device)
            inject_ex = a.use_exemplars and len(sample['exemplar_boxes']) > 0 and (random.random() < a.exemplar_prob)
            ex_boxes = sample['exemplar_boxes'] if inject_ex else None
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                st = model.forward_stages(
                    inputs['pixel_values'], inputs.get('pixel_mask'),
                    inputs['input_ids'], inputs['attention_mask'], inputs.get('token_type_ids'),
                    target_phrase_idx=target_idx, exemplar_boxes=ex_boxes)
            loss, _ = compute_7layer_loss(criterion, st, sample, device)
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
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            del st, loss, inputs
            if (i + 1) % 100 == 0:
                torch.cuda.empty_cache()
                save(a.output / 'progress.json', dict(epoch=epoch, images=i + 1, total=len(names), mean_loss=float(np.mean(losses))))
                print('train', epoch, i + 1, len(names), 'loss', round(float(np.mean(losses)), 4), flush=True)
        scheduler.step()
        res = evaluate_llmdet(model, val, dev, device, a.output / f'dev_epoch{epoch:02d}.json', use_exemplars=a.use_exemplars)
        mae = res['summary']['0.23']['mae']
        best_th_mae = min(res['summary'][str(t)]['mae'] for t in THRESHOLDS)
        entry = dict(epoch=epoch, loss=float(np.mean(losses)), dev_mae=mae, dev_best_th_mae=best_th_mae)
        if a.use_exemplars:
            res_txt = evaluate_llmdet(model, val, dev, device, a.output / f'dev_text_epoch{epoch:02d}.json', use_exemplars=False)
            entry['dev_text_mae'] = res_txt['summary']['0.23']['mae']
        history.append(entry)
        save(a.output / 'history.json', history)
        torch.save(dict(trainable=model.export_trainable_state(), epoch=epoch, dev_mae=mae, manifest=manifest), a.output / 'last.pth')
        if mae < best:
            best = mae
            torch.save(dict(trainable=model.export_trainable_state(), epoch=epoch, dev_mae=best, manifest=manifest), a.output / 'best.pth')
        extra_str = f" text_mae {round(entry['dev_text_mae'], 3)}" if 'dev_text_mae' in entry else ''
        print(f'epoch {epoch} dev_mae {round(mae, 3)}{extra_str} best_th_mae {round(best_th_mae, 3)} best {round(best, 3)}', flush=True)

    best_state = torch.load(a.output / 'best.pth', map_location='cpu', weights_only=False)
    model.import_trainable_state(best_state['trainable'])
    final_summary = dict(state='complete', best_epoch=best_state['epoch'], best_dev_mae=best)
    final_summary['fullval_raw'] = evaluate_llmdet(
        model, val, val.names, device, a.output / 'adapted_fullval_raw.json',
        use_exemplars=a.use_exemplars, tile_trigger=0)['summary']
    if a.tile_trigger > 0:
        final_summary['fullval_tiled'] = evaluate_llmdet(
            model, val, val.names, device, a.output / 'adapted_fullval_tiled.json',
            use_exemplars=a.use_exemplars, tile_trigger=a.tile_trigger)['summary']
    if a.use_exemplars:
        final_summary['fullval_text_only'] = evaluate_llmdet(
            model, val, val.names, device, a.output / 'adapted_fullval_text_only.json',
            use_exemplars=False, tile_trigger=a.tile_trigger)['summary']
    if pilot:
        final_summary['pilot58'] = evaluate_llmdet(
            model, val, pilot, device, a.output / 'adapted_pilot58.json',
            use_exemplars=a.use_exemplars, tile_trigger=a.tile_trigger)['summary']
    final_summary['absent'] = evaluate_llmdet(
        model, val, ['1919.jpg', '3436.jpg'], device, a.output / 'adapted_absent.json',
        absent=True, use_exemplars=False, tile_trigger=0)['summary']
    save(a.output / 'result.json', final_summary)


if __name__ == '__main__':
    main()
