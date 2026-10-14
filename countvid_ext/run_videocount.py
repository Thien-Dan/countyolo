"""
Chạy CountVid (đã vá bằng patch_countvid.py) trên một tập VideoCount với một cấu hình, rồi tính MAE/RMSE
giống evaluate_counting_accuracy.py. Lệnh count_in_videos.py lấy đúng như test_crystals.py / test_penguins.py /
test_mot20_count.py của CountVid (chỉ text), cộng thêm cờ của cấu hình.

    python run_videocount.py --countvid_dir CountVid --data_root VideoCount --dataset crystals \
        --name tile32 --extra "--tile_target 32 --nms_iou 0.5" --results results.json
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

# dataset -> (thư mục, file GT, text, cờ gốc của CountVid)
DATASETS = {
    "crystals": ("Crystals", "crystals-count-gt.json",
                 "--obj_batch_size 30 --img_batch_size 4 --downsample_factor 1 --temporal_filter --w 3 "
                 "--convert_to_rgb"),
    "penguins": ("Penguins", "penguins-count-gt.json",
                 "--obj_batch_size 30 --img_batch_size 4 --downsample_factor 1 --temporal_filter --w 3"),
    "mot20": ("MOT20-Count", "MOT20-count-gt.json",
              f"--obj_batch_size 10 --img_batch_size 4 --downsample_factor {25 / 3}"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--countvid_dir", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--name", required=True, help="tên cấu hình")
    p.add_argument("--extra", default="", help="cờ thêm cho count_in_videos.py")
    p.add_argument("--videos", nargs="*", default=None, help="chỉ chạy các video này")
    p.add_argument("--results", required=True, help="file JSON tích luỹ kết quả mọi cấu hình")
    a = p.parse_args()

    sub, gt_name, base_flags = DATASETS[a.dataset]
    data_dir = os.path.abspath(os.path.join(a.data_root, sub))
    gt = json.load(open(os.path.join(data_dir, "anno", gt_name)))
    ckpt = os.path.abspath(a.checkpoint_dir)
    out_file = os.path.abspath(f"pred_{a.dataset}_{a.name}.json")
    json.dump({}, open(out_file, "w"))
    temp_dir = os.path.abspath(f"temp_{a.dataset}_{a.name}")

    per_video, t0 = {}, time.time()
    for video, texts in gt.items():
        if a.videos and video not in a.videos:
            continue
        for text, gt_count in texts.items():
            shutil.rmtree(temp_dir, ignore_errors=True)
            cmd = (f'{sys.executable} count_in_videos.py --video_dir "{os.path.join(data_dir, "frames", video)}" '
                   f'--input_text "{text}" --sam_checkpoint "{os.path.join(ckpt, "sam2.1_hiera_large.pt")}" '
                   f'--sam_model_cfg "configs/sam2.1/sam2.1_hiera_l.yaml" '
                   f'--pretrain_model_path "{os.path.join(ckpt, "countgd_box.pth")}" '
                   f'--output_file "{out_file}" --temp_dir "{temp_dir}" {base_flags} {a.extra}')
            tv = time.time()
            r = subprocess.run(shlex.split(cmd), cwd=a.countvid_dir, capture_output=True, text=True)
            log = r.stdout + r.stderr
            for line in log.splitlines():
                if line.startswith(("time stage", "tiling:", "Traceback")) or "Error" in line:
                    print(f"    [{video}] {line}", flush=True)
            if r.returncode != 0:
                print(log[-3000:], flush=True)
                sys.exit(f"count_in_videos.py lỗi ở {video}")
            preds = json.load(open(out_file))
            key = next(k for k in preds if k.endswith("/" + video) or k == video)
            pred = preds[key][text]
            gt_count = len(gt_count) if isinstance(gt_count, list) else gt_count
            per_video[video] = {"gt": gt_count, "pred": pred, "sec": round(time.time() - tv, 1)}
            print(f"  {a.dataset}/{a.name} {video}: pred {pred} gt {gt_count} ({time.time() - tv:.0f}s)", flush=True)

    errs = [v["pred"] - v["gt"] for v in per_video.values()]
    res = {"mae": sum(abs(e) for e in errs) / len(errs), "rmse": (sum(e * e for e in errs) / len(errs)) ** 0.5,
           "bias": sum(errs) / len(errs), "sec": round(time.time() - t0, 1), "videos": per_video, "extra": a.extra}
    print(f"== {a.dataset}/{a.name}: MAE {res['mae']:.2f} RMSE {res['rmse']:.2f} bias {res['bias']:+.2f} "
          f"({res['sec']:.0f}s)", flush=True)
    allres = json.load(open(a.results)) if os.path.exists(a.results) else {}
    allres[f"{a.dataset}/{a.name}"] = res
    json.dump(allres, open(a.results, "w"), indent=1)


if __name__ == "__main__":
    main()
