"""Geometry helpers for CountYOLO -> SAM 2 proposals (CPU tensors)."""
import torch
from torchvision.ops import box_iou, nms


def fuse_proposals(peaks, masses, detections, scores, iou=0.6):
    """Use detector geometry where a density peak agrees; retain unmatched peaks.

    Detector boxes have priority, each may absorb at most one density peak.
    Scores of different heads are never compared as probabilities.
    """
    peaks, masses = peaks.float().cpu(), masses.float().cpu()
    detections, scores = detections.float().cpu(), scores.float().cpu()
    if len(detections) == 0:
        return peaks, masses
    keep = nms(detections, scores, iou)
    detections, scores = detections[keep], scores[keep]
    remaining = torch.ones(len(peaks), dtype=torch.bool)
    if len(peaks):
        centers = (peaks[:, :2] + peaks[:, 2:]) / 2
        dc = (detections[:, :2] + detections[:, 2:]) / 2
        wh = (detections[:, 2:] - detections[:, :2]).clamp(min=1)
        offset = (centers[:, None] - dc[None]).abs() / wh[None]
        inside = (offset <= 0.5).all(-1)
        overlaps = box_iou(peaks, detections)
        costs = ((centers[:, None] - dc[None]) / wh[None]).square().sum(-1)
        for j in range(len(detections)):
            eligible = remaining & (inside[:, j] | (overlaps[:, j] >= 0.3))
            if eligible.any():
                candidates = torch.where(eligible)[0]
                idx = candidates[costs[candidates, j].argmin()]
                remaining[idx] = False
    boxes = torch.cat([detections, peaks[remaining]])
    # Detector priority is explicit; density mass is only used to rank other peaks.
    peak_rank = masses[remaining] / masses[remaining].clamp(min=1e-6).max() if remaining.any() else masses[remaining]
    rank = torch.cat([2 + scores, peak_rank])
    keep = nms(boxes, rank, iou)
    return boxes[keep], rank[keep]


def decode_yolo(dets, width, height, size=640, conf=0.15):
    d = dets[0].float().cpu()
    xy, wh, scores = d[:2].T, d[2:4].T, d[4:].max(0).values
    boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)
    boxes *= torch.tensor([width / size, height / size] * 2)
    valid = (scores >= conf) & torch.isfinite(boxes).all(1) & ((boxes[:, 2:] - boxes[:, :2]) > 1).all(1)
    boxes, scores = boxes[valid], scores[valid]
    keep = nms(boxes, scores, 0.5)
    return boxes[keep], scores[keep]
