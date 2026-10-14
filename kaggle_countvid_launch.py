"""
Đóng gói countvid_ext/ vào một Kaggle script kernel: cài CountVid (CountGD-Box + SAM 2.1), tải checkpoint và
VideoCount, vá Canonical-Scale Tiling + NMS vào Stage 1, rồi so sánh các cấu hình trên Crystals / Penguins
(chỉ text, giống Bảng 2 của paper CountVid: Crystals MAE 69.1, Penguins MAE 4.0).

    python kaggle_countvid_launch.py --username <kaggle_user> [--datasets crystals penguins] [--push]

Theo dõi:  kaggle kernels status <user>/countvid-tiling
Kết quả:   kaggle kernels output <user>/countvid-tiling -p <dir>   (results.json)
"""
import argparse
import base64
import json
import os
import subprocess

SOURCES = ["countvid_ext/tiled_countgd.py", "countvid_ext/patch_countvid.py", "countvid_ext/run_videocount.py"]
SLUG = "countvid-tiling"
OUT_DIR = "kaggle_countvid_run"
# Cấu hình so sánh: tên -> cờ thêm cho count_in_videos.py
CONFIGS = {
    "baseline": "",
    "grid2": "--tile_grid 2 --nms_iou 0.5",
    "grid3": "--tile_grid 3 --nms_iou 0.5",
}

RUNNER = r'''
import base64, glob, json, os, subprocess, sys
FILES = __FILES__
DATASETS = __DATASETS__
CONFIGS = __CONFIGS__
COUNTGD_BOX_GDRIVE = "1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD"
VIDEOCOUNT_GDRIVE = "1elk1-83dstQ0hEPLctrLKUNz4qUkrdDs"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

W = "/kaggle/working"
ext = os.path.join(W, "countvid_ext")
for path, b64 in FILES.items():
    dst = os.path.join(W, path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(base64.b64decode(b64))

def run(cmd, cwd=None, check=True, env=None):
    print("\n$ " + cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if check and r.returncode != 0:
        sys.exit(f"Lệnh thất bại ({r.returncode}): {cmd}")
    return r.returncode

run("nvidia-smi --query-gpu=name,memory.total --format=csv")
cv = os.path.join(W, "CountVid")
run(f"git clone -q --depth 1 https://github.com/niki-amini-naieni/CountVid.git {cv}")
# transformers v5 bỏ BertModel.get_head_mask mà bertwarper.py của GroundingDINO cần
run("pip install -q addict yapf==0.40.1 termcolor pycocotools gdown 'transformers>=4.40,<5'")
env = dict(os.environ, SAM2_BUILD_CUDA="0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
run("pip install -q 'git+https://github.com/facebookresearch/sam2.git'", env=env)

# CUDA op của GroundingDINO; không biên dịch được -> dùng bản PyTorch thuần có sẵn (chậm hơn, kết quả như nhau)
ops = os.path.join(cv, "models/GroundingDINO/ops")
run(f"rm -rf {ops}/build && sed -i '/^import spaces/d' {ops}/setup.py")
# torch mới bỏ Tensor.type() và Tensor.data<T>(): x.type().is_cuda() -> x.is_cuda(),
# AT_DISPATCH(value.type()) -> value.scalar_type(), data<T>() -> data_ptr<T>()
cu = os.path.join(ops, "src/cuda/ms_deform_attn_cuda.cu")
s = open(cu).read().replace(".type().is_cuda()", ".is_cuda()").replace("value.type()", "value.scalar_type()")
s = s.replace(".data<scalar_t>()", ".data_ptr<scalar_t>()").replace(".data<int64_t>()", ".data_ptr<int64_t>()")
open(cu, "w").write(s)
# torch >= 2.6 mặc định weights_only=True; checkpoint CountGD-Box chính thức có argparse.Namespace
civ = os.path.join(cv, "count_in_videos.py")
s = open(civ).read()
s = s.replace('torch.load(args.pretrain_model_path, map_location="cpu")',
              'torch.load(args.pretrain_model_path, map_location="cpu", weights_only=False)')
open(civ, "w").write(s)
if run("python setup.py build install > build.log 2>&1", cwd=ops, check=False) != 0 or \
        run("python -c 'import torch, MultiScaleDeformableAttention'", cwd=W, check=False) != 0:
    print("!! Không biên dịch được MultiScaleDeformableAttention, dùng bản PyTorch thuần", flush=True)
    run(f"grep -E 'error|Error' {ops}/build.log | head -20", check=False)
    f = os.path.join(cv, "models/GroundingDINO/ms_deform_attn.py")
    s = open(f).read()
    # Dòng import thật (không phải bản đã comment "#    import ..." ở phía trên)
    s = s.replace("\nimport MultiScaleDeformableAttention as _C\n", "\n_C = None\n", 1)
    s = s.replace("if torch.cuda.is_available() and value.is_cuda:", "if _C is not None and value.is_cuda:")
    open(f, "w").write(s)

ck = os.path.join(cv, "checkpoints")
os.makedirs(ck, exist_ok=True)
run("python download_bert.py", cwd=cv)
run(f"gdown {COUNTGD_BOX_GDRIVE} -O {ck}/countgd_box.pth")
run(f"wget -q -P {ck} {SAM_URL}")

data_root = os.path.join(W, "data")
os.makedirs(data_root, exist_ok=True)
run(f"gdown {VIDEOCOUNT_GDRIVE} -O {W}/VideoCount.zip")
run(f"unzip -q {W}/VideoCount.zip -d {data_root} && rm {W}/VideoCount.zip")
anno = glob.glob(f"{data_root}/**/Crystals/anno", recursive=True)
if not anno:
    sys.exit("Không thấy VideoCount/Crystals sau khi giải nén")
vc_root = os.path.dirname(os.path.dirname(anno[0]))
print("VideoCount:", vc_root, os.listdir(vc_root), flush=True)

results = os.path.join(W, "results.json")
try:
    run(f"python {ext}/patch_countvid.py {cv}")
    for ds in DATASETS:
        for name, extra in CONFIGS.items():
            run(f'python {ext}/run_videocount.py --countvid_dir {cv} --data_root {vc_root} --checkpoint_dir {ck} '
                f'--dataset {ds} --name {name} --extra="{extra}" --results {results}', check=False)
    print("\n===== TỔNG KẾT =====", flush=True)
    for k, v in (json.load(open(results)) if os.path.exists(results) else {}).items():
        print(f"{k:24s} MAE {v['mae']:7.2f}  RMSE {v['rmse']:7.2f}  bias {v['bias']:+7.2f}  {v['sec']:6.0f}s",
              flush=True)
finally:
    # Output kernel chỉ giữ log + results.json, không giữ dữ liệu / checkpoint / repo
    run(f"rm -rf {data_root} {cv} {W}/temp_*", check=False)
print("DONE", flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--datasets", nargs="+", default=["crystals", "penguins"],
                        choices=["crystals", "penguins", "mot20"])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--push", action="store_true", help="Đẩy kernel lên Kaggle và chạy ngay")
    args = parser.parse_args()

    files = {}
    for path in SOURCES:
        with open(path, "rb") as f:
            files[path] = base64.b64encode(f.read()).decode()
    runner = RUNNER.replace("__FILES__", repr(files)).replace("__DATASETS__", repr(args.datasets)) \
        .replace("__CONFIGS__", repr({k: CONFIGS[k] for k in args.configs}))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "run.py"), "w", encoding="utf-8") as f:
        f.write(runner)
    meta = {
        "id": f"{args.username}/{SLUG}",
        "title": SLUG,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"Đã tạo {OUT_DIR}/ cho kernel {meta['id']}")

    if args.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", OUT_DIR], check=True)


if __name__ == "__main__":
    main()
