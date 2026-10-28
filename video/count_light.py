"""
Đếm vật duy nhất trong video không dùng SAM 2: CountYOLO đếm từng frame -> điểm vật = đỉnh density (lọc min_mass)
kèm feature DINOv2 -> tracker nhẹ (utils/tracking.py) -> số đếm = số track xuất hiện ít nhất min_hits frame.
Lấy mẫu frame và text prompt giống CountVid (test_*.py) để so sánh trực tiếp trên VideoCount.

    python video/count_light.py --data-root VideoCount --dataset penguins --checkpoint <ckpt> --min-mass 0.2
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from models.density_yolo import DensityYOLOWorld  # noqa: E402
from utils.scale_zoom import density_tiled_count  # noqa: E402
from utils.tracking import Tracker  # noqa: E402

# dataset -> (thư mục, file GT, hệ số giảm frame như CountVid)
DATASETS = {
    "crystals": ("Crystals", "crystals-count-gt.json", 1.0),
    "penguins": ("Penguins", "penguins-count-gt.json", 1.0),
    "mot20": ("MOT20-Count", "MOT20-count-gt.json", 25 / 3),
}
# Cấu hình tracker cố định trước (không chỉnh theo kết quả test)
TRACKERS = {
    "position": dict(gate=1.0, max_miss=3, min_hits=3, app_weight=0.0),
    "appearance": dict(gate=1.5, max_miss=10, min_hits=3, app_weight=1.0, min_cos=0.5),
}


def frame_list(video_dir, downsample=1.0):
    names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in (".jpg", ".jpeg", ".png")]
    names.sort(key=lambda p: int(os.path.splitext(p)[0][re.search(r"\d", p).start():]))
    if downsample > 1:
        idx = np.round(np.linspace(0, len(names) - 1, int(np.ceil(len(names) / downsample)))).astype(int)
        names = [names[i] for i in idx]
    return [os.path.join(video_dir, n) for n in names]


@torch.no_grad()
def count_video(model, frames, text, device, min_mass=0.2, ttn_thr=400, ttn_grid=2, tracker="appearance",
                on_frame=None):
    """
    frames: list đường dẫn ảnh hoặc PIL.Image. on_frame(i, image, boxes, ids, frame_count) gọi sau mỗi frame
    (dùng cho UI vẽ kết quả). Trả về {"count", "frame_counts", "frames", "sec"}.
    """
    tr = Tracker(**TRACKERS[tracker])
    use_feats = TRACKERS[tracker].get("app_weight", 0) > 0
    frame_counts = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        img = (Image.open(f) if isinstance(f, str) else f).convert("RGB")
        count, info, boxes, _ = density_tiled_count(model, img, text, device, count_thr=ttn_thr, grid=ttn_grid,
                                                    return_boxes=True, min_mass=min_mass)
        b = boxes.numpy()
        centers = (b[:, :2] + b[:, 2:]) / 2
        sides = np.sqrt(np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1, None))
        feats = info["feats"].numpy() if use_feats and info["feats"] is not None else None
        ids = tr.update(centers, sides, feats)
        frame_counts.append(count)
        if on_frame is not None:
            on_frame(i, img, b, ids, count)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"count": tr.count(), "frame_counts": frame_counts, "frames": len(frames),
            "sec": time.perf_counter() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Thư mục VideoCount")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--min-mass", type=float, default=0.2, help="Ngưỡng khối lượng đỉnh (tools/calibrate_peaks.py)")
    p.add_argument("--ttn-thr", type=float, default=400, help="Test-time tiling theo số đếm (chọn trên FSC val)")
    p.add_argument("--ttn-grid", type=int, default=2)
    p.add_argument("--tracker", choices=TRACKERS, default="appearance")
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--results", default="video_light_results.json")
    p.add_argument("--name", default=None)
    a = p.parse_args()
    name = a.name or a.tracker

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DensityYOLOWorld("yolov8s-world.pt")
    model.load_head(a.checkpoint, map_location="cpu")
    model.to(device).eval_mode()
    sub, gt_name, down = DATASETS[a.dataset]
    gt = json.load(open(os.path.join(a.data_root, sub, "anno", gt_name)))

    per_video, frames_total, t_total = {}, 0, 0.0
    for video, texts in gt.items():
        if a.videos and video not in a.videos:
            continue
        frames = frame_list(os.path.join(a.data_root, sub, "frames", video), down)
        for text, gt_count in texts.items():
            gt_count = len(gt_count) if isinstance(gt_count, list) else gt_count
            r = count_video(model, frames, text, device, a.min_mass, a.ttn_thr, a.ttn_grid, a.tracker)
            per_video[video] = {"gt": gt_count, "pred": r["count"], "frames": r["frames"], "sec": round(r["sec"], 1),
                                "fps": round(r["frames"] / r["sec"], 2),
                                "mean_frame_count": round(float(np.mean(r["frame_counts"])), 1)}
            frames_total += r["frames"]
            t_total += r["sec"]
            print(f"  {a.dataset}/{name} {video}: pred {r['count']} gt {gt_count} | {r['frames']} frame, "
                  f"{r['frames'] / r['sec']:.1f} FPS | đếm TB/frame {np.mean(r['frame_counts']):.1f}", flush=True)

    errs = [v["pred"] - v["gt"] for v in per_video.values()]
    res = {"mae": float(np.mean(np.abs(errs))), "rmse": float(np.sqrt(np.mean(np.square(errs)))),
           "bias": float(np.mean(errs)), "fps": round(frames_total / t_total, 2), "videos": per_video,
           "params": {"min_mass": a.min_mass, "ttn_thr": a.ttn_thr, "ttn_grid": a.ttn_grid,
                      "tracker": TRACKERS[a.tracker]}}
    print(f"== {a.dataset}/{name}: MAE {res['mae']:.2f} RMSE {res['rmse']:.2f} bias {res['bias']:+.2f} "
          f"| {res['fps']:.1f} FPS", flush=True)
    allres = json.load(open(a.results)) if os.path.exists(a.results) else {}
    allres[f"{a.dataset}/{name}"] = res
    json.dump(allres, open(a.results, "w"), indent=1)


if __name__ == "__main__":
    main()
