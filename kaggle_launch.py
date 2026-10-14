"""
Đóng gói source code vào một Kaggle script kernel (không cần push GitHub) rồi đẩy lên Kaggle.

    python kaggle_launch.py --username <kaggle_user> [--epochs 50] [--push]
    python kaggle_launch.py --username <kaggle_user> --finetune [--epochs 15 --lr 1e-5] [--push]

--finetune: kernel riêng (countyolo-density-finetune) gắn output của countyolo-density-train, train tiếp
từ density_head_best.pth của lần đó với lr nhỏ.
--unfreeze-from 12: như --finetune nhưng train cả neck YOLO-World (kernel countyolo-density-unfreeze),
lr neck = lr x --yolo-lr-mult, L2-SP kéo neck về trọng số pretrained.

Kernel sẽ: ghi lại source code -> train Density Head + Self-Exemplar + Scale Head -> eval density-count trên
test (exemplar tự chọn / GT) và Canonical-Scale Zoom với nhiều kích thước chuẩn trên val + test.
Theo dõi:  kaggle kernels status <user>/countyolo-density-train
Kết quả:   kaggle kernels output <user>/countyolo-density-train -p <dir>
"""
import argparse
import base64
import json
import os
import subprocess

SOURCES = [
    "models/density_yolo.py",
    "datasets/__init__.py",
    "datasets/density_generator.py",
    "datasets/fsc147_dataset.py",
    "utils/custom_nms.py",
    "utils/scale_zoom.py",
    "train_density_head.py",
    "eval_fsc147.py",
]
DATASET = "xuncngng/fsc147-0"
SLUG = "countyolo-density-train"
FINETUNE_SLUG = "countyolo-density-finetune"
UNFREEZE_SLUG = "countyolo-density-unfreeze"
VIT_SLUG = "countyolo-density-vit"
ANALYZE_SLUG = "countyolo-analyze"
OUT_DIR = "kaggle_run"

RUNNER = r'''
import base64, glob, os, subprocess, sys

FILES = __FILES__
EPOCHS = __EPOCHS__
LR = __LR__
FINETUNE = __FINETUNE__
TRAIN_EXTRA = __TRAIN_EXTRA__
ANALYZE = __ANALYZE__
os.environ["TQDM_DISABLE"] = "1"  # log gọn: không in progress bar

work = "/kaggle/working/countyolo"
for path, b64 in FILES.items():
    dst = os.path.join(work, path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(base64.b64decode(b64))
os.chdir(work)

def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"Lệnh thất bại ({r.returncode}): {cmd}")

run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
anno = glob.glob("/kaggle/input/**/annotation_FSC147_384.json", recursive=True)
if not anno:
    sys.exit("Không tìm thấy FSC-147 trong /kaggle/input")
data_dir = os.path.dirname(anno[0])
print("FSC-147:", data_dir, flush=True)

if ANALYZE:
    # Không train: lấy best checkpoint của từng kernel nguồn, eval val/test và ghi CSV kết quả từng ảnh
    os.makedirs("/kaggle/working/dumps", exist_ok=True)
    for src in ANALYZE:
        ck = glob.glob(f"/kaggle/input/**/{src}/checkpoints/density_head_best.pth", recursive=True)
        if not ck:
            print("!! Không thấy checkpoint của", src, flush=True)
            continue
        for split in ["test", "val"]:
            run([sys.executable, "eval_fsc147.py", "--mode", "density-count", "--split", split, "--data-dir",
                 data_dir, "--checkpoint", ck[0], "--dump", f"/kaggle/working/dumps/{src}_{split}.csv"])
    print("\nDONE", flush=True)
    sys.exit(0)

ckpt_dir = "/kaggle/working/checkpoints"
extra = ["--lr", str(LR), *TRAIN_EXTRA]
if FINETUNE:
    init = glob.glob("/kaggle/input/**/checkpoints/density_head_best.pth", recursive=True)
    if not init:
        sys.exit("Không tìm thấy density_head_best.pth của kernel train trong /kaggle/input")
    print("Checkpoint khởi tạo:", init[0], flush=True)
    extra += ["--init", init[0]]
run([sys.executable, "train_density_head.py", "--data-dir", data_dir, "--out-dir", ckpt_dir,
     "--epochs", str(EPOCHS), "--batch-size", "8", "--workers", "2", *extra])
def evaluate(split, *extra):
    run([sys.executable, "eval_fsc147.py", "--mode", "density-count", "--split", split, "--data-dir", data_dir,
         "--checkpoint", os.path.join(ckpt_dir, "density_head_best.pth"), *extra])

evaluate("test", "--dump", "/kaggle/working/test_dump.csv")
evaluate("test", "--exemplars", "gt")
# Canonical-Scale Zoom: chọn kích thước chuẩn trên val, báo cáo test
for target in ["32", "44", "56"]:
    evaluate("val", "--zoom", target)
    evaluate("test", "--zoom", target)
print("\nDONE", flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--epochs", type=int, default=None, help="Mặc định 50 (train) / 15 (finetune)")
    parser.add_argument("--lr", type=float, default=None, help="Mặc định 1e-4 (train) / 1e-5 (finetune)")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune từ output của kernel train")
    parser.add_argument("--unfreeze-from", type=int, default=None,
                        help="Train cả YOLO-World từ layer này (12 = neck); ngụ ý --finetune")
    parser.add_argument("--yolo-lr-mult", type=float, default=0.1)
    parser.add_argument("--l2sp-weight", type=float, default=0.01)
    parser.add_argument("--vit", action="store_true",
                        help="Train từ đầu kiến trúc YOLO-World + DINOv2 (arch vit, kernel countyolo-density-vit)")
    parser.add_argument("--vit-name", default=None, help="vd facebook/dinov2-base (ngụ ý --vit)")
    parser.add_argument("--vit-input", type=int, default=None, help="vd 840 (ngụ ý --vit)")
    parser.add_argument("--slug-suffix", default="", help="Hậu tố tên kernel để chạy song song nhiều cấu hình")
    parser.add_argument("--mosaic-grids", type=int, nargs="+", default=None, help="vd 2 4 (truyền cho train)")
    parser.add_argument("--mosaic-prob", type=float, default=None)
    parser.add_argument("--analyze", nargs="+", default=None, metavar="KERNEL",
                        help="Chỉ eval + ghi CSV từng ảnh cho best checkpoint của các kernel này (cùng tài khoản)")
    parser.add_argument("--push", action="store_true", help="Đẩy kernel lên Kaggle và chạy ngay")
    args = parser.parse_args()

    train_extra = []
    if args.unfreeze_from is not None:
        args.finetune = True
        train_extra = ["--unfreeze-from", str(args.unfreeze_from), "--yolo-lr-mult", str(args.yolo_lr_mult),
                       "--l2sp-weight", str(args.l2sp_weight)]
    if args.mosaic_grids:
        train_extra += ["--mosaic-grids", *map(str, args.mosaic_grids)]
    if args.mosaic_prob is not None:
        train_extra += ["--mosaic-prob", str(args.mosaic_prob)]
    if args.vit_name or args.vit_input:
        args.vit = True
    if args.vit:
        train_extra += ["--arch", "vit"]
        if args.vit_name:
            train_extra += ["--vit-name", args.vit_name]
        if args.vit_input:
            train_extra += ["--vit-input", str(args.vit_input)]
    files = {}
    for path in SOURCES:
        with open(path, "rb") as f:
            files[path] = base64.b64encode(f.read()).decode()
    epochs = args.epochs or (15 if args.finetune else 50)
    lr = args.lr or (1e-5 if args.finetune else 1e-4)
    slug = VIT_SLUG if args.vit else UNFREEZE_SLUG if train_extra else FINETUNE_SLUG if args.finetune else SLUG
    if args.analyze:
        slug = ANALYZE_SLUG
    slug += args.slug_suffix
    runner = RUNNER.replace("__FILES__", repr(files)).replace("__EPOCHS__", str(epochs))         .replace("__LR__", repr(lr)).replace("__FINETUNE__", str(args.finetune))         .replace("__TRAIN_EXTRA__", repr(train_extra))         .replace("__ANALYZE__", repr(args.analyze or []))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "run.py"), "w", encoding="utf-8") as f:
        f.write(runner)
    meta = {
        "id": f"{args.username}/{slug}",
        "title": slug,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [f"{args.username}/{k}" for k in args.analyze] if args.analyze
        else [f"{args.username}/{SLUG}"] if args.finetune else [],
    }
    with open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"Đã tạo {OUT_DIR}/ cho kernel {meta['id']}")

    if args.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", OUT_DIR], check=True)


if __name__ == "__main__":
    main()
