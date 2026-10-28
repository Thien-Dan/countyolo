"""
Đo chi phí suy luận các biến thể DensityYOLOWorld (và YOLO-World đơn thuần) trên ảnh 640x640:
số tham số, độ trễ batch 1, thông lượng batch 8, bộ nhớ GPU đỉnh. Tốc độ không phụ thuộc trọng số đã
train nên head được khởi tạo ngẫu nhiên.

    python tools/benchmark_speed.py --out bench_countyolo.json
"""
# Chạy được từ thư mục gốc dự án: python tools/<script>.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import time

import torch

from models.density_yolo import DensityYOLOWorld

VARIANTS = {
    "simcentric": dict(arch="simcentric"),
    "vit-small-448": dict(arch="vit", vit_name="facebook/dinov2-small", vit_input=448),
    "vit-small-560": dict(arch="vit", vit_name="facebook/dinov2-small", vit_input=560),
    "vit-small-840": dict(arch="vit", vit_name="facebook/dinov2-small", vit_input=840),
    "vit-base-560": dict(arch="vit", vit_name="facebook/dinov2-base", vit_input=560),
    # Tối ưu suy luận (model.optimize_inference): DINOv2 fp16 thật, tuỳ chọn torch.compile. "base" = biến thể gốc
    # để kiểm tra số đếm không đổi (cùng seed khởi tạo head, cùng đầu vào).
    "vit-small-560-fp16": dict(arch="vit", vit_name="facebook/dinov2-small", vit_input=560,
                               opt=dict(half_vit=True), base="vit-small-560"),
    "vit-small-448-fp16": dict(arch="vit", vit_name="facebook/dinov2-small", vit_input=448,
                               opt=dict(half_vit=True), base="vit-small-448"),
    "vit-small-560-fp16-compile": dict(arch="vit", vit_name="facebook/dinov2-small", vit_input=560,
                                       opt=dict(half_vit=True, compile_vit=True), base="vit-small-560"),
}


def n_params(module):
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


@torch.no_grad()
def timeit(fn, x, amp, warmup=10, iters=50):
    for _ in range(warmup):
        with torch.autocast("cuda", enabled=amp):
            fn(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        with torch.autocast("cuda", enabled=amp):
            fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def bench(name, fn, params, amp_modes=(False, True)):
    res = {"params_M": round(params / 1e6, 2)}
    for amp in amp_modes:
        tag = "amp" if amp else "fp32"
        torch.cuda.reset_peak_memory_stats()
        lat = timeit(fn, torch.rand(1, 3, 640, 640, device="cuda"), amp)
        thr = timeit(fn, torch.rand(8, 3, 640, 640, device="cuda"), amp, warmup=3, iters=20)
        res[tag] = {"latency_ms": round(lat * 1000, 1), "fps_bs1": round(1 / lat, 1),
                    "fps_bs8": round(8 / thr, 1), "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    print(f"{name:16s} {res['params_M']:7.1f}M | fp32 {res['fp32']['latency_ms']:6.1f}ms "
          f"({res['fp32']['fps_bs1']:5.1f} FPS, bs8 {res['fp32']['fps_bs8']:5.1f}) | "
          f"amp {res['amp']['latency_ms']:6.1f}ms ({res['amp']['fps_bs1']:5.1f} FPS, bs8 {res['amp']['fps_bs8']:5.1f})",
          flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8s-world.pt")
    p.add_argument("--out", default="bench_countyolo.json")
    p.add_argument("--only", nargs="*", default=None, help="Chỉ đo các biến thể này")
    a = p.parse_args()
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    results, ref = {}, {}
    torch.manual_seed(123)
    x0 = torch.rand(1, 3, 640, 640, device="cuda")
    for name, cfg in VARIANTS.items():
        if a.only and name not in a.only:
            continue
        cfg = dict(cfg)
        opt, base = cfg.pop("opt", None), cfg.pop("base", None)
        torch.manual_seed(0)  # cùng khởi tạo head giữa biến thể gốc và biến thể tối ưu
        model = DensityYOLOWorld(a.weights, **cfg).cuda()
        model.eval_mode()
        if opt:
            model.optimize_inference(**opt)
        classes = ["apple"]
        model.encode_text(classes)  # cache CLIP text, không tính vào thời gian

        def run(x, model=model):
            return model(x, classes * len(x))

        if name == "simcentric":
            # YOLO-World đơn thuần (cùng đầu vào, cùng text) làm mốc
            yolo = model.yolo.model
            txt = model.encode_text(classes)

            def run_yolo(x):
                return yolo.predict(x, txt_feats=txt.expand(len(x), -1, -1).to(x.device))

            results["yolo-world-v8s"] = bench("yolo-world-v8s", run_yolo, n_params(model.yolo.model))
        total = n_params(model.yolo.model) + n_params(model.vit_extractor) + n_params(model.trainable_modules())
        results[name] = bench(name, run, total)
        results[name]["trainable_M"] = round(n_params(model.trainable_modules()) / 1e6, 2)
        with torch.no_grad(), torch.autocast("cuda"):
            c = float(model(x0, ["apple"])[1].float().sum())
        ref[name] = c
        if base in ref:
            rel = abs(c - ref[base]) / max(abs(ref[base]), 1e-6)
            results[name]["count_rel_diff_vs_base"] = round(rel, 5)
            print(f"   {name}: số đếm {c:.3f} vs {base} {ref[base]:.3f} (lệch {rel * 100:.3f}%)", flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
