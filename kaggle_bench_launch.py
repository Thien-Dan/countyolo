"""
Kaggle kernel đo chi phí suy luận trên cùng một GPU: các biến thể CountYOLO (benchmark_speed.py) và CountGD-Box
của CountVid (countvid_ext/bench_countgd.py). Tốc độ không phụ thuộc trọng số nên không cần checkpoint đã train.

    python kaggle_bench_launch.py --username <kaggle_user> [--push]

Kết quả: bench_countyolo.json, bench_countgd.json trong output kernel <user>/countyolo-bench.
"""
import argparse
import base64
import json
import os
import subprocess

SOURCES = [
    "models/density_yolo.py",
    "benchmark_speed.py",
    "countvid_ext/bench_countgd.py",
]
SLUG = "countyolo-bench"
OUT_DIR = "kaggle_bench_run"

RUNNER = r'''
import base64, os, subprocess, sys
FILES = __FILES__
W = "/kaggle/working"
cy = os.path.join(W, "countyolo")
for path, b64 in FILES.items():
    dst = os.path.join(cy, path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(base64.b64decode(b64))
open(os.path.join(cy, "models", "__init__.py"), "a").close()

def run(cmd, cwd=None, check=True, env=None):
    print("\n$ " + cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if check and r.returncode != 0:
        sys.exit(f"Lệnh thất bại ({r.returncode}): {cmd}")
    return r.returncode

run("nvidia-smi --query-gpu=name,memory.total --format=csv")
# transformers v5 bỏ BertModel.get_head_mask mà GroundingDINO cần; DINOv2 chạy được với 4.x
run("pip install -q ultralytics addict yapf==0.40.1 termcolor pycocotools 'transformers>=4.40,<5'")

# 1. CountYOLO
run(f"python benchmark_speed.py --out {W}/bench_countyolo.json", cwd=cy, check=False)

# 2. CountGD-Box (CountVid): cài như kaggle_countvid_launch.py
cv = os.path.join(W, "CountVid")
run(f"git clone -q --depth 1 https://github.com/niki-amini-naieni/CountVid.git {cv}")
run("pip install -q 'git+https://github.com/facebookresearch/sam2.git'", env=dict(os.environ, SAM2_BUILD_CUDA="0"))
ops = os.path.join(cv, "models/GroundingDINO/ops")
run(f"rm -rf {ops}/build && sed -i '/^import spaces/d' {ops}/setup.py")
cu = os.path.join(ops, "src/cuda/ms_deform_attn_cuda.cu")
s = open(cu).read().replace(".type().is_cuda()", ".is_cuda()").replace("value.type()", "value.scalar_type()")
s = s.replace(".data<scalar_t>()", ".data_ptr<scalar_t>()").replace(".data<int64_t>()", ".data_ptr<int64_t>()")
open(cu, "w").write(s)
if run("python setup.py build install > build.log 2>&1", cwd=ops, check=False) != 0 or \
        run("python -c 'import torch, MultiScaleDeformableAttention'", cwd=W, check=False) != 0:
    sys.exit("!! Không biên dịch được CUDA op của GroundingDINO -> số đo CountGD sẽ không công bằng, dừng")
os.makedirs(os.path.join(cv, "checkpoints"), exist_ok=True)
run("python download_bert.py", cwd=cv)
# Checkpoint rỗng: build_model_and_transforms nạp strict=False, tốc độ không phụ thuộc trọng số
run("python -c \"import torch; torch.save({'model': {}}, 'checkpoints/countgd_dummy.pth')\"", cwd=cv)
run(f"cp {cy}/countvid_ext/bench_countgd.py {cv}/")
run(f"python bench_countgd.py --checkpoint checkpoints/countgd_dummy.pth --out {W}/bench_countgd.json", cwd=cv,
    check=False)
run(f"rm -rf {cv}", check=False)
print("\nDONE", flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    files = {p: base64.b64encode(open(p, "rb").read()).decode() for p in SOURCES}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "run.py"), "w", encoding="utf-8") as f:
        f.write(RUNNER.replace("__FILES__", repr(files)))
    meta = {"id": f"{args.username}/{SLUG}", "title": SLUG, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    with open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"Đã tạo {OUT_DIR}/ cho kernel {meta['id']}")
    if args.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", OUT_DIR], check=True)


if __name__ == "__main__":
    main()
