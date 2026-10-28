"""
Đóng gói source code vào một Kaggle script kernel (không cần push GitHub) rồi đẩy lên Kaggle.

    python launchers/kaggle_launch.py --username <kaggle_user> [--epochs 50] [--push]
    python launchers/kaggle_launch.py --username <kaggle_user> --finetune [--epochs 15 --lr 1e-5] [--push]

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
    "utils/tracking.py",
    "tools/visualize_density.py",
    "tools/visualize_detections.py",
    "tools/ttn_sweep.py",
    "tools/calibrate_peaks.py",
    "video/count_light.py",
    "tools/prompt_eval.py",
    "tools/presence_eval.py",
    "tools/clip_verify_eval.py",
    "utils/clip_verify.py",
    "utils/fsc147_train_classes.txt",
    "tools/bench_verify.py",
    "tools/sam3_pseudo.py",
    "engines/__init__.py",
    "engines/sam3_counter.py",
]
SAM3_DATASET = "nadthien/sam3-weights"  # dataset riêng tư chứa sam3.pt (trọng số SAM 3 cần quyền của Meta)
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
VIS_IMAGES = __VIS_IMAGES__
TTN_SWEEP = __TTN_SWEEP__
VIDEO_LIGHT = __VIDEO_LIGHT__
PROMPT_EVAL = __PROMPT_EVAL__
INIT_URL = __INIT_URL__
SAM3_DISTILL = __SAM3_DISTILL__
FAST_EVAL = __FAST_EVAL__
FAST_EVAL_RUNS = __FAST_EVAL_RUNS__
PRESENCE_EVAL = __PRESENCE_EVAL__
CLIP_VERIFY = __CLIP_VERIFY__
BENCH_VERIFY = __BENCH_VERIFY__
CLIP_K = __CLIP_K__
PROMPT_ONLY = __PROMPT_ONLY__
PROMPT_EVAL_ARGS = __PROMPT_EVAL_ARGS__
OLD_CKPT_URL = ("https://raw.githubusercontent.com/Thien-Dan/countyolo/master/checkpoints/"
                "density_head_best_vit_seed3_ep19_valmae21.44.pth")
VIDEOCOUNT_GDRIVE = "1elk1-83dstQ0hEPLctrLKUNz4qUkrdDs"
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
        ck = glob.glob(f"/kaggle/input/**/{src}/checkpoints/density_head_best.pth", recursive=True) or             glob.glob(f"/kaggle/input/**/{src}/**/density_head_best.pth", recursive=True)
        if not ck:
            print("!! Không thấy checkpoint của", src, "| .pth trong /kaggle/input:",
                  glob.glob("/kaggle/input/**/*.pth", recursive=True)[:20], flush=True)
            continue
        if PROMPT_ONLY is not None:
            # Chỉ chạy tools/prompt_eval.py (thêm cờ, vd --oracle-exemplars) cho checkpoint nguồn; có bản cuối
            # (density_head_last.pth) thì đo cả nó + test MAE với exemplar GT (best theo val có thể là bản khởi tạo)
            for tag, c in [("best", ck[0]), ("last", ck[0].replace("density_head_best", "density_head_last"))]:
                if not os.path.exists(c):
                    continue
                run([sys.executable, "tools/prompt_eval.py", "--checkpoint", c, "--data-dir", data_dir,
                     "--out", f"/kaggle/working/prompt_eval_{src}_{tag}.json", *PROMPT_ONLY])
                if tag == "last":
                    run([sys.executable, "eval_fsc147.py", "--mode", "density-count", "--split", "test",
                         "--data-dir", data_dir, "--checkpoint", c, "--exemplars", "gt"])
            continue
        if BENCH_VERIFY:
            run([sys.executable, "tools/bench_verify.py", "--checkpoint", ck[0], "--data-dir", data_dir,
                 "--out", f"/kaggle/working/bench_verify_{src}.json"])
            continue
        if CLIP_VERIFY:
            run([sys.executable, "tools/clip_verify_eval.py", "--checkpoint", ck[0], "--data-dir", data_dir,
                 "--out", f"/kaggle/working/clip_verify_{src}.json", "--fast", "--k", str(CLIP_K),
                 "--dump", f"/kaggle/working/clip_verify_{src}.csv"])
            continue
        if PRESENCE_EVAL:
            run([sys.executable, "tools/presence_eval.py", "--checkpoint", ck[0], "--data-dir", data_dir,
                 "--out", f"/kaggle/working/presence_{src}.json"])
            continue
        if FAST_EVAL:
            # Kiểm tra suy luận nhanh (fp16 + compile) không làm giảm độ chính xác: test thường vs --fast
            for tag, more in FAST_EVAL_RUNS:
                run([sys.executable, "eval_fsc147.py", "--mode", "density-count", "--split", "test", "--data-dir",
                     data_dir, "--checkpoint", ck[0], "--dump", f"/kaggle/working/dumps/{src}_{tag}.csv", *more])
            continue
        if VIDEO_LIGHT:
            # Chọn ngưỡng lọc đỉnh trên FSC val, rồi đếm video không cần SAM 2 trên VideoCount
            import json as _json
            calib = "/kaggle/working/calib.json"
            run([sys.executable, "tools/calibrate_peaks.py", "--checkpoint", ck[0], "--data-dir", data_dir, "--out", calib])
            mm = _json.load(open(calib))["best_min_mass"]
            run([sys.executable, "-m", "pip", "install", "-q", "gdown"])
            run(["gdown", VIDEOCOUNT_GDRIVE, "-O", "/kaggle/working/VideoCount.zip"])
            run(["unzip", "-q", "/kaggle/working/VideoCount.zip", "-d", "/kaggle/working/vc"])
            vc = os.path.dirname(os.path.dirname(glob.glob("/kaggle/working/vc/**/Crystals/anno", recursive=True)[0]))
            res = "/kaggle/working/video_light_results.json"
            for ds in ["crystals", "penguins", "mot20"]:
                for trk in ["position", "appearance"]:
                    run([sys.executable, "video/count_light.py", "--data-root", vc, "--dataset", ds,
                         "--checkpoint", ck[0], "--min-mass", str(mm), "--tracker", trk, "--results", res])
            run(["rm", "-rf", "/kaggle/working/vc", "/kaggle/working/VideoCount.zip"])
            continue
        if TTN_SWEEP:
            # Chọn test-time tiling trên val, báo cáo test
            run([sys.executable, "tools/ttn_sweep.py", "--checkpoint", ck[0], "--data-dir", data_dir,
                 "--out", f"/kaggle/working/ttn_{src}.json", "--dump-dir", "/kaggle/working/dumps"])
            continue
        if VIS_IMAGES:
            # Chỉ trực quan hoá vài ảnh test, không eval toàn bộ
            run([sys.executable, "tools/visualize_density.py", "--data-dir", data_dir, "--checkpoint", ck[0],
                 "--images", *VIS_IMAGES, "--out", f"/kaggle/working/vis_{src}.png"])
            run([sys.executable, "tools/visualize_detections.py", "--data-dir", data_dir, "--checkpoint", ck[0],
                 "--images", *VIS_IMAGES, "--out", f"/kaggle/working/det_{src}.png"])
            continue
        for split in ["test", "val"]:
            run([sys.executable, "eval_fsc147.py", "--mode", "density-count", "--split", split, "--data-dir",
                 data_dir, "--checkpoint", ck[0], "--dump", f"/kaggle/working/dumps/{src}_{split}.csv"])
    print("\nDONE", flush=True)
    sys.exit(0)

ckpt_dir = "/kaggle/working/checkpoints"
extra = ["--lr", str(LR), *TRAIN_EXTRA]
if SAM3_DISTILL:
    # Chưng cất SAM 3: sinh mask "khái niệm được hỏi" (gồm negative khó) cho ảnh train, rồi train cổng text theo đó
    w = glob.glob("/kaggle/input/**/sam3.pt", recursive=True)
    if not w:
        sys.exit("Không thấy sam3.pt trong /kaggle/input (dataset " + "sam3-weights" + ")")
    os.makedirs("weights", exist_ok=True)
    if not os.path.exists("weights/sam3.pt"):
        os.symlink(w[0], "weights/sam3.pt")
    run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics", "timm"])
    pseudo = "/kaggle/working/sam3_pseudo_train.npz"
    run([sys.executable, "tools/sam3_pseudo.py", "--data-dir", data_dir, "--split", "train", "--out", pseudo])
    extra += ["--pseudo", pseudo]
if INIT_URL:
    # Khởi tạo từ checkpoint công khai trên GitHub (vd vitgate từ vit tốt nhất)
    run(["wget", "-q", "-O", "/kaggle/working/init.pth", INIT_URL])
    extra += ["--init", "/kaggle/working/init.pth"]
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
# Test-time tiling theo số đếm (ngưỡng 400, lưới 2 đã chọn trên val ở lần trước)
evaluate("test", "--ttn-thr", "400", "--ttn-grid", "2")
if CLIP_VERIFY:
    # Chống nhầm lớp bằng CLIP trên checkpoint vừa train (cùng cấu hình UI: --fast, K)
    run([sys.executable, "tools/clip_verify_eval.py", "--checkpoint", os.path.join(ckpt_dir, "density_head_best.pth"),
         "--data-dir", data_dir, "--out", "/kaggle/working/clip_verify_new.json", "--fast", "--k", str(CLIP_K),
         "--dump", "/kaggle/working/clip_verify_new.csv"])
if PROMPT_EVAL:
    # Mức "nghe theo text": mô hình vừa train và bản tốt nhất cũ (arch vit) trên cùng GPU
    run([sys.executable, "tools/prompt_eval.py", "--checkpoint", os.path.join(ckpt_dir, "density_head_best.pth"),
         "--data-dir", data_dir, "--out", "/kaggle/working/prompt_eval_new.json", *PROMPT_EVAL_ARGS])
    old = "/kaggle/working/old_vit.pth"
    run(["wget", "-q", "-O", old, OLD_CKPT_URL])
    run([sys.executable, "tools/prompt_eval.py", "--checkpoint", old, "--data-dir", data_dir,
         "--out", "/kaggle/working/prompt_eval_old.json", *PROMPT_EVAL_ARGS])
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
    parser.add_argument("--init-url", default=None, help="URL checkpoint khởi tạo (vd vit tốt nhất trên GitHub)")
    parser.add_argument("--gate-only", action="store_true", help="Truyền cho train (arch vitgate)")
    parser.add_argument("--loss-norm-ref", type=float, default=0.0, help="Truyền cho train: chuẩn hoá loss theo mẫu")
    parser.add_argument("--ema", type=float, default=0.0, help="Truyền cho train: EMA trọng số (vd 0.999)")
    parser.add_argument("--sam3-distill", action="store_true",
                        help="Sinh nhãn SAM 3 trên Kaggle (dataset sam3-weights) rồi train với --pseudo")
    parser.add_argument("--hard-prob", type=float, default=0.0, help="Truyền cho train (cần --sam3-distill)")
    parser.add_argument("--gate-bce", type=float, default=0.0, help="Truyền cho train (cần --sam3-distill)")
    parser.add_argument("--arch", choices=["vit", "vitpa", "vitgate"], default=None,
                        help="vitpa: thêm kênh logit text (dùng cùng --neg-prob/--mixed-prob); ngụ ý --vit")
    parser.add_argument("--neg-prob", type=float, default=0.0, help="Truyền cho train: prompt sai, đáp án 0")
    parser.add_argument("--mixed-prob", type=float, default=0.0, help="Truyền cho train: mosaic trộn lớp")
    parser.add_argument("--prompt-eval", action="store_true",
                        help="Sau train: đo mức nghe theo text (tools/prompt_eval.py) cho bản mới và bản cũ")
    parser.add_argument("--vit", action="store_true",
                        help="Train từ đầu kiến trúc YOLO-World + DINOv2 (arch vit, kernel countyolo-density-vit)")
    parser.add_argument("--vit-name", default=None, help="vd facebook/dinov2-base (ngụ ý --vit)")
    parser.add_argument("--vit-input", type=int, default=None, help="vd 840 (ngụ ý --vit)")
    parser.add_argument("--slug-suffix", default="", help="Hậu tố tên kernel để chạy song song nhiều cấu hình")
    parser.add_argument("--mosaic-grids", type=int, nargs="+", default=None, help="vd 2 4 (truyền cho train)")
    parser.add_argument("--mosaic-prob", type=float, default=None)
    parser.add_argument("--fast-eval", action="store_true",
                        help="Cùng --analyze: so test MAE bản thường vs --fast (fp16 + torch.compile)")
    parser.add_argument("--prompt-eval-args", default="", help="Cờ thêm cho prompt_eval sau train (vd '--oracle-exemplars')")
    parser.add_argument("--gt-exemplar-prob", type=float, default=None, help="Truyền cho train")
    parser.add_argument("--ex-noise", type=float, default=0.0, help="Truyền cho train")
    parser.add_argument("--ex-noise-other", action="store_true", help="Truyền cho train")
    parser.add_argument("--val-mosaics", type=int, default=0, help="Truyền cho train (chọn best theo cả mosaic 2 lớp)")
    parser.add_argument("--prompt-only", default=None,
                        help="Cùng --analyze: chỉ chạy prompt_eval với các cờ này (chuỗi, vd '--oracle-exemplars')")
    parser.add_argument("--ckpt-dataset", default=None,
                        help="Cùng --analyze: dataset Kaggle chứa <nguồn>/checkpoints/density_head_best.pth thay cho "
                             "output kernel nguồn (chạy được trên tài khoản khác)")
    parser.add_argument("--bench-verify", action="store_true",
                        help="Cùng --analyze: đo độ trễ từng phần CountYOLO + CLIP kiểm tra lớp (tools/bench_verify.py)")
    parser.add_argument("--clip-k", type=int, default=12, help="Cùng --clip-verify: số đỉnh density và số box YOLO-World")
    parser.add_argument("--clip-verify", action="store_true",
                        help="Cùng --analyze: chống nhầm lớp bằng CLIP kiểm tra vùng đỉnh density (tools/clip_verify_eval.py)")
    parser.add_argument("--presence-eval", action="store_true",
                        help="Cùng --analyze: cổng vitgate dùng ở mức có/không (tools/presence_eval.py)")
    parser.add_argument("--tta-eval", action="store_true", help="Cùng --fast-eval: so fast vs fast+TTA lật ngang")
    parser.add_argument("--video-light", action="store_true",
                        help="Cùng --analyze: chọn ngưỡng đỉnh trên FSC val rồi đếm VideoCount không dùng SAM 2")
    parser.add_argument("--ttn-sweep", action="store_true",
                        help="Cùng --analyze: chọn test-time tiling trên val rồi eval test (ttn_sweep.py)")
    parser.add_argument("--vis-images", nargs="+", default=None,
                        help="Cùng --analyze: chỉ vẽ pipeline cho các ảnh test này (vd 6248.jpg 7426.jpg)")
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
    if args.arch:
        args.vit = True
    if args.vit:
        train_extra += ["--arch", args.arch or "vit"]
        if args.vit_name:
            train_extra += ["--vit-name", args.vit_name]
        if args.vit_input:
            train_extra += ["--vit-input", str(args.vit_input)]
    if args.neg_prob:
        train_extra += ["--neg-prob", str(args.neg_prob)]
    if args.mixed_prob:
        train_extra += ["--mixed-prob", str(args.mixed_prob)]
    if args.gate_only:
        train_extra += ["--gate-only"]
    if args.loss_norm_ref:
        train_extra += ["--loss-norm-ref", str(args.loss_norm_ref)]
    if args.gt_exemplar_prob is not None:
        train_extra += ["--gt-exemplar-prob", str(args.gt_exemplar_prob)]
    if args.val_mosaics:
        train_extra += ["--val-mosaics", str(args.val_mosaics)]
    if args.ex_noise:
        train_extra += ["--ex-noise", str(args.ex_noise)]
    if args.ex_noise_other:
        train_extra += ["--ex-noise-other"]
    if args.ema:
        train_extra += ["--ema", str(args.ema)]
    if args.hard_prob:
        train_extra += ["--hard-prob", str(args.hard_prob)]
    if args.gate_bce:
        train_extra += ["--gate-bce", str(args.gate_bce)]
    files = {}
    for path in SOURCES:
        with open(path, "rb") as f:
            files[path] = base64.b64encode(f.read()).decode()
    epochs =args.epochs or (15 if args.finetune else 50)
    lr = args.lr or (1e-5 if args.finetune else 1e-4)
    slug = VIT_SLUG if args.vit else UNFREEZE_SLUG if train_extra else FINETUNE_SLUG if args.finetune else SLUG
    if args.analyze:
        slug = ANALYZE_SLUG
    slug += args.slug_suffix
    runner = RUNNER.replace("__FILES__", repr(files)).replace("__EPOCHS__", str(epochs))         .replace("__LR__", repr(lr)).replace("__FINETUNE__", str(args.finetune))         .replace("__TRAIN_EXTRA__", repr(train_extra))         .replace("__ANALYZE__", repr(args.analyze or [])).replace("__VIS_IMAGES__", repr(args.vis_images or []))         .replace("__TTN_SWEEP__", str(bool(args.ttn_sweep)))         .replace("__VIDEO_LIGHT__", str(bool(args.video_light)))         .replace("__PROMPT_EVAL__", str(bool(args.prompt_eval)))         .replace("__INIT_URL__", repr(args.init_url))         .replace("__SAM3_DISTILL__", str(bool(args.sam3_distill)))         .replace("__FAST_EVAL__", str(bool(args.fast_eval)))         .replace("__PRESENCE_EVAL__", str(bool(args.presence_eval)))         .replace("__CLIP_VERIFY__", str(bool(args.clip_verify)))         .replace("__BENCH_VERIFY__", str(bool(args.bench_verify)))         .replace("__CLIP_K__", str(args.clip_k))         .replace("__PROMPT_ONLY__", repr(args.prompt_only.split() if args.prompt_only is not None else None))         .replace("__PROMPT_EVAL_ARGS__", repr(args.prompt_eval_args.split()))         .replace("__FAST_EVAL_RUNS__", repr([(t, ["--fast", *e.split()] if e else ["--fast"]) for t, e in [("fast", ""), ("fast_tta", "--tta")]] if args.tta_eval else [("base", []), ("fast", ["--fast"]), ("fast_ttn", ["--fast", "--ttn-thr", "400"])]))

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
        "dataset_sources": [DATASET] + ([SAM3_DATASET] if args.sam3_distill else []) + ([args.ckpt_dataset] if args.ckpt_dataset else []),
        "competition_sources": [],
        "kernel_sources": [f"{args.username}/{k}" for k in args.analyze] if args.analyze and not args.ckpt_dataset
        else [f"{args.username}/{SLUG}"] if args.finetune else [],
    }
    with open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"Đã tạo {OUT_DIR}/ cho kernel {meta['id']}")

    if args.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", OUT_DIR], check=True)


if __name__ == "__main__":
    main()
