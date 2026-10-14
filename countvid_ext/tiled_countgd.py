"""
Canonical-Scale Tiling cho CountGD-Box trong CountVid (Stage 1).

CountGD resize mọi frame về cạnh ngắn 800px. Với frame lớn chứa vật nhỏ (Crystals 1600px, vật ~22px -> ~11px sau
resize; MOT20 1080p đông người) vật bị bỏ sót, và mỗi lần chạy chỉ có 900 query. Ý tưởng giống
Canonical-Scale Zoom của CountYOLO: ước lượng kích thước vật từ box của lần chạy toàn frame, nếu nhỏ hơn
kích thước chuẩn thì chia frame thành lưới g x g ô chồng lấn, mỗi ô được CountGD phóng về 800px, rồi gộp box:
  - bỏ box của ô chạm cạnh ô nằm bên trong frame (vật bị cắt; ô bên cạnh nhìn thấy trọn vật nhờ phần chồng lấn)
  - NMS trên [box toàn frame + box các ô]
"""
import math

import torch
from torchvision.ops import nms


def countgd_scale(W, H, short=800, max_size=1333):
    """Hệ số resize của T.RandomResize([800], max_size=1333) mà CountGD dùng."""
    return min(short / min(W, H), max_size / max(W, H))


def choose_grid(boxes, W, H, target, max_grid):
    """Số ô mỗi chiều để vật (trung vị sqrt(w*h) của box toàn frame, đo sau resize của CountGD) đạt ~target px.
    Không có box nào: coi như vật quá nhỏ để thấy -> dùng max_grid."""
    if len(boxes) == 0:
        return max_grid
    side = ((boxes[:, 2] - boxes[:, 0]).clamp(min=1) * (boxes[:, 3] - boxes[:, 1]).clamp(min=1)).sqrt().median()
    side = side.item() * countgd_scale(W, H)
    return int(min(max(math.ceil(target / max(side, 1e-6)), 1), max_grid))


def tile_windows(W, H, grid, overlap):
    """Lưới grid x grid cửa sổ [x0, y0, x1, y1] phủ frame, mỗi ô rộng hơn W/grid một tỉ lệ overlap."""
    tw, th = min(W, W / grid * (1 + overlap)), min(H, H / grid * (1 + overlap))
    xs = [0.0] if grid == 1 else [i * (W - tw) / (grid - 1) for i in range(grid)]
    ys = [0.0] if grid == 1 else [i * (H - th) / (grid - 1) for i in range(grid)]
    return [(round(x), round(y), round(x + tw), round(y + th)) for y in ys for x in xs]


def _drop_cut_boxes(boxes, win, W, H, margin):
    """Bỏ box chạm cạnh ô mà cạnh đó nằm trong frame (không phải biên ảnh)."""
    x0, y0, x1, y1 = win
    keep = torch.ones(len(boxes), dtype=torch.bool)
    if x0 > 0:
        keep &= boxes[:, 0] > x0 + margin
    if y0 > 0:
        keep &= boxes[:, 1] > y0 + margin
    if x1 < W:
        keep &= boxes[:, 2] < x1 - margin
    if y1 < H:
        keep &= boxes[:, 3] < y1 - margin
    return keep


def tiled_boxes(count_fn, transform, frame, full_boxes, full_scores, text, exemplar_image, exemplars, label,
                conf_thresh, target=32, max_grid=3, overlap=0.25, nms_iou=0.5, batch_size=4, fixed_grid=0):
    """
    Args:
        count_fn: hàm count(...) của count_in_videos.py (batch ảnh -> pred_logits, pred_boxes cxcywh chuẩn hoá)
        frame: PIL.Image frame gốc (W, H)
        full_boxes, full_scores: box xyxy (pixel frame) + score của lần chạy toàn frame
        exemplar_image, exemplars, label: đầu vào exemplar của chính frame này (như trong batch gốc)
        fixed_grid: > 0 thì dùng lưới cố định thay vì chọn theo kích thước vật
    Returns:
        (boxes xyxy, scores, grid) sau khi gộp
    """
    W, H = frame.size
    grid = fixed_grid if fixed_grid > 0 else choose_grid(full_boxes, W, H, target, max_grid)
    all_boxes, all_scores = [full_boxes.float()], [full_scores.float()]
    if grid > 1:
        wins = tile_windows(W, H, grid, overlap)
        margin = 0.01 * min(W, H) / grid
        text_only = len(exemplars) == 0
        for s in range(0, len(wins), batch_size):
            chunk = wins[s:s + batch_size]
            tiles = [transform(frame.crop(w), {"exemplars": torch.tensor([])})[0] for w in chunk]
            tiles = torch.stack(tiles) if len({t.shape for t in tiles}) == 1 else None
            groups = [tiles] if tiles is not None else \
                [transform(frame.crop(w), {"exemplars": torch.tensor([])})[0].unsqueeze(0) for w in chunk]
            out_boxes, out_logits = [], []
            for g in groups:
                ex_img = g if text_only else exemplar_image.unsqueeze(0).expand(len(g), -1, -1, -1)
                ex = torch.stack([exemplars] * len(g))
                out = count_fn(g, ex_img, ex, torch.stack([label] * len(g)), text)
                out_logits.append(out["pred_logits"].sigmoid().max(dim=-1).values.cpu())
                out_boxes.append(out["pred_boxes"].float().cpu())
            for w, logits, boxes in zip(chunk, torch.cat(out_logits), torch.cat(out_boxes)):
                m = logits > conf_thresh
                b, sc = boxes[m], logits[m]
                tw, th = w[2] - w[0], w[3] - w[1]
                xyxy = torch.stack([w[0] + tw * (b[:, 0] - b[:, 2] / 2), w[1] + th * (b[:, 1] - b[:, 3] / 2),
                                    w[0] + tw * (b[:, 0] + b[:, 2] / 2), w[1] + th * (b[:, 1] + b[:, 3] / 2)], 1)
                keep = _drop_cut_boxes(xyxy, w, W, H, margin)
                all_boxes.append(xyxy[keep])
                all_scores.append(sc[keep].float())
    boxes, scores = torch.cat(all_boxes), torch.cat(all_scores)
    if nms_iou < 1.0 and len(boxes) > 0:
        keep = nms(boxes, scores, nms_iou)
        boxes, scores = boxes[keep], scores[keep]
    return boxes, scores, grid
