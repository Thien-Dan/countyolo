"""
Đo độ trễ từng phần của CountYOLO + CLIP kiểm tra lớp (utils/clip_verify.VerifiedCounter), batch 1, ảnh FSC-147 thật:
  model      : CountYOLO một lượt (chế độ nhanh fp16 + torch.compile)
  verified   : VerifiedCounter đầy đủ (chế độ auto)
  cands      : tạo vùng ứng viên (đỉnh density thô + box YOLO-World + roi_align)
  clip_<n>   : CLIP encode_image n vùng (fp16)
  clip_<n>_compiled : như trên, torch.compile(mode="reduce-overhead") với batch cố định

    python tools/bench_verify.py --checkpoint <ckpt> --data-dir data/FSC147 --out bench_verify.json
"""
# Chạy được từ thư mục gốc dự án: python tools/<script>.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.fsc147_dataset import FSC147Dataset  # noqa: E402
from models.density_yolo import DensityYOLOWorld  # noqa: E402
from utils.clip_verify import ClipVerifier, VerifiedCounter, coarse_peak_boxes, default_vocab, yolo_boxes  # noqa: E402


def timeit(fn, inputs, warmup=10):
    for i in range(warmup):
        fn(*inputs[i % len(inputs)])
    torch.cuda.synchronize()
    ms = []
    for inp in inputs:
        t0 = time.perf_counter()
        fn(*inp)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
    return {"median": float(np.median(ms)), "p90": float(np.percentile(ms, 90))}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data/FSC147")
    p.add_argument("--n", type=int, default=100, help="Số ảnh test đem đo")
    p.add_argument("--out", default="bench_verify.json")
    a = p.parse_args()
    dev = torch.device("cuda")
    model = DensityYOLOWorld("yolov8s-world.pt")
    model.load_head(a.checkpoint, map_location="cpu")
    model.to(dev).eval_mode()
    model.optimize_inference(half_vit=True, compile_vit=True)
    v = ClipVerifier(model, default_vocab())
    vc = VerifiedCounter(model, v, thr=0.1279)
    ds = FSC147Dataset(a.data_dir, split="test")
    items = [(ds[i]["image"][None].to(dev), ds[i]["class_name"]) for i in range(a.n)]
    for _, c in items:  # cache text CLIP của mọi prompt trước khi đo
        model.encode_text([c])
        v.probs(torch.zeros(1, 512, device=dev), c)
    res = {"gpu": torch.cuda.get_device_name(0)}
    res["model"] = timeit(lambda x, c: model(x, [c]), items)
    res["verified"] = timeit(lambda x, c: vc(x, c), items)

    # vùng ứng viên trên density thô / dets thật của từng ảnh
    cached = []
    for x, c in items:
        grab = {}
        model(x, [c], exemplar_fn=lambda coarse, dets: grab.update(coarse=coarse, dets=dets))
        cached.append((x, grab["coarse"], grab["dets"]))
    n_crops = []

    def cands(x, coarse, dets):
        b = torch.cat([coarse_peak_boxes(coarse, 12), yolo_boxes(dets, 12)])
        n_crops.append(len(b))
        return v.crops(x[0], b)
    res["cands"] = timeit(cands, cached)
    res["n_crops_median"] = float(np.median(n_crops))
    for n in (13, 25):
        crops = [(torch.rand(n, 3, 224, 224, device=dev),) for _ in range(30)]
        res[f"clip_{n}"] = timeit(v.image_feats, crops)
    enc = torch.compile(v.net.encode_image, mode="reduce-overhead")
    for n in (13, 25):
        crops = [((torch.rand(n, 3, 224, 224, device=dev) - v.mean) / v.std,) for _ in range(30)]
        res[f"clip_{n}_compiled"] = timeit(enc, crops, warmup=15)
    for k, r in res.items():
        print(f"{k:22s}: {r}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
