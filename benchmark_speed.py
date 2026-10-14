"""
Đo chi phí suy luận các biến thể DensityYOLOWorld (và YOLO-World đơn thuần) trên ảnh 640x640:
số tham số, độ trễ batch 1, thông lượng batch 8, bộ nhớ GPU đỉnh. Tốc độ không phụ thuộc trọng số đã
train nên head được khởi tạo ngẫu nhiên.

    python benchmark_speed.py --out bench_countyolo.json
"""
import argparse
import json
import time

import torch

from models.density_yolo import DensityYOLOWorld

VARIANTS = {
    "simcentric": dict(arch="simcentric"),
    "vit-small-560": dict(arch="vit", vit_name="facebook/dinov2-small", vit_input=560),
    "vit-small-840": dict(arch="vit", vit_name="facebook/dinov2-small", vit_input=840),
    "vit-base-560": dict(arch="vit", vit_name="facebook/dinov2-base", vit_input=560),
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
    a = p.parse_args()
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    results = {}
    for name, cfg in VARIANTS.items():
        model = DensityYOLOWorld(a.weights, **cfg).cuda()
        model.eval_mode()
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
        del model
        torch.cuda.empty_cache()
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
