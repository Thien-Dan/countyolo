"""
Đo chi phí suy luận CountGD-Box (bộ đếm từng frame của CountVid), chỉ text, đúng tiền xử lý của CountVid
(resize cạnh ngắn 800, tối đa 1333). Chạy trong thư mục CountVid đã cài đặt:

    python bench_countgd.py --checkpoint checkpoints/countgd_box.pth --out bench_countgd.json

Dựng model bằng chính mã của count_in_videos.py (thực thi phần định nghĩa hàm, dừng trước khi parse args).
"""
import argparse
import json
import sys
import time

import torch

MARK = "# Parse commandline arguments."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="bench_countgd.json")
    a = p.parse_args()

    src = open("count_in_videos.py", encoding="utf-8").read()
    g = {"__name__": "countvid_defs"}
    exec(compile(src[:src.index(MARK)], "count_in_videos.py", "exec"), g)
    args = g["get_args_parser"]().parse_args(["--pretrain_model_path", a.checkpoint])
    model, _ = g["build_model_and_transforms"](args)
    model = model.cuda().eval()
    nested = g["nested_tensor_from_tensor_list"]
    params = sum(q.numel() for q in model.parameters())

    def run(x):
        # Như count() của CountVid ở chế độ chỉ text: ảnh exemplar = chính ảnh, không có box exemplar
        return model(nested(list(x)), nested(list(x)), [torch.tensor([], device="cuda")] * len(x),
                     [torch.tensor([0], device="cuda")] * len(x), captions=["apple ."] * len(x))

    @torch.no_grad()
    def timeit(x, amp, warmup=5, iters=20):
        for _ in range(warmup):
            with torch.autocast("cuda", enabled=amp):
                run(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            with torch.autocast("cuda", enabled=amp):
                run(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / iters

    # Ảnh FSC-147 384x576 sau resize của CountGD -> 800x1200; frame 1080p -> 800x1333 (max_size)
    res = {"params_M": round(params / 1e6, 2)}
    for size_name, (h, w) in {"fsc147_800x1200": (800, 1200), "video_800x1333": (800, 1333)}.items():
        for amp in (False, True):
            tag = f"{size_name}_{'amp' if amp else 'fp32'}"
            torch.cuda.reset_peak_memory_stats()
            try:
                lat = timeit(torch.rand(1, 3, h, w, device="cuda"), amp)
                thr = timeit(torch.rand(4, 3, h, w, device="cuda"), amp, warmup=2, iters=10)
                res[tag] = {"latency_ms": round(lat * 1000, 1), "fps_bs1": round(1 / lat, 2),
                            "fps_bs4": round(4 / thr, 2),
                            "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            except Exception as e:  # AMP có thể không chạy với CUDA op của GroundingDINO
                res[tag] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"countgd-box {tag:24s} {res[tag]}", flush=True)
    print(f"countgd-box params {res['params_M']}M", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
