# CountYOLO

Đếm vật thể theo text (zero-shot) trên ảnh và video: **YOLO-World + DINOv2 đóng băng**, chỉ train các head
nhỏ (~0.6M tham số) phía sau. Chi tiết kiến trúc, thí nghiệm và lịch sử: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

```
Ảnh 640 ─► YOLO-World v8s (đóng băng) ─► P3/P4/P5 + Text Map
       ├─► Sobel ─► Edge Map
       └─► DINOv2-small (đóng băng) ─► token 40×40
Density Head ─► density thô ─► 5 đỉnh = exemplar ─► similarity P3/P4/DINOv2 ─► Refine Head ─► density cuối
SỐ ĐẾM = tổng density   (ảnh đếm ≥ 400 được chia 2×2 và đếm lại)
```

## Kết quả

| FSC-147 test (chỉ text) | MAE | Thời gian/ảnh (T4) |
|---|---|---|
| **CountYOLO + DINOv2 + chia ô** | **17.98** | ~44 ms |
| CountYOLO + DINOv2 (3 seed) | 19.39 ± 0.57 | 43.5 ms |
| CountYOLO simcentric (không ViT) | 21.4 – 25.0 | 17.1 ms |
| CountGD-Box (CountVid) | 15.01 | 422 ms |
| YOLO-Count (ICCV 2025) / RT-Counter (preprint 2026) | 14.80 / 13.30 | — |

| Video (MAE số vật duy nhất) | CountVid | CountYOLO + SAM 2 | Tracker nhẹ: vị trí | Tracker nhẹ: vị trí + DINOv2 |
|---|---|---|---|---|
| Crystals | 69.14 | **49.86** | 58.7 | 60.9 |
| Penguins | **3.33** | 6.67 | 33.7 | 17.3 |
| MOT20 | **50.0** | — (MOT20-01: 102 vs 97) | 287.0 (đếm thừa) | 258.0 (đếm thiếu) |

Tracker nhẹ chạy ~8–10 FPS trên T4 (CountVid ~0.02 FPS) nhưng còn kém xa SAM 2 về giữ danh tính vật: theo
vị trí thì tách một vật thành nhiều track, thêm feature DINOv2 thì gộp nhầm người trong đám đông.

## Cài đặt

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -r requirements.txt
```
Checkpoint khuyến nghị cho ảnh: `checkpoints/density_head_best_vit_exnoise.pth`; seed3 được giữ làm đối chứng và cấu hình được chọn trên video pilot.
`yolov8s-world.pt`, CLIP và DINOv2 tự tải lần đầu. Dữ liệu: FSC-147 ở `data/FSC147`, VideoCount ở `data/VideoCount`.

## Sử dụng

```bash
# Giao diện (ảnh + video)
.venv/Scripts/python app/app.py                      # http://127.0.0.1:7860

# Train / đánh giá FSC-147
python train_density_head.py --arch vit
python eval_fsc147.py --mode density-count --checkpoint density_head_best_vit_seed3_ep19_valmae21.44.pth \
    --ttn-thr 400 --ttn-grid 2

# Đếm video nhẹ trên VideoCount
python video/count_light.py --data-root data/VideoCount --dataset penguins \
    --checkpoint checkpoints/density_head_best_vit_seed3_ep19_valmae21.44.pth --tracker appearance

# Chạy trên Kaggle (train, phân tích, CountVid, đo tốc độ)
python launchers/kaggle_launch.py --username <user> --vit --push
```

## Cấu trúc thư mục

```
countyolo/
├── models/density_yolo.py     # DensityYOLOWorld: YOLO-World + DINOv2 + các head (arch simcentric / vit)
├── datasets/                  # FSC-147 loader (mosaic, size map), density GT
├── utils/
│   ├── scale_zoom.py          # chia ô khi suy luận, điểm/box từ đỉnh density
│   ├── tracking.py            # tracker nhẹ (vị trí + feature DINOv2) cho video
│   └── custom_nms.py          # Density-Guided Soft-NMS
├── train_density_head.py      # train
├── eval_fsc147.py             # đánh giá FSC-147 (--ttn-thr, --dump, --exemplars gt)
├── tools/                     # phân tích: ttn_sweep, calibrate_peaks, visualize_*, benchmark_speed
├── video/count_light.py       # đếm vật duy nhất trong video không cần SAM 2
├── countvid_ext/              # plugin cho CountVid: tiling + NMS Stage 1, CountYOLO làm Stage 1, đo CountGD
├── launchers/                 # đóng gói + đẩy kernel Kaggle
├── app/app.py                 # giao diện Gradio
├── checkpoints/               # checkpoint local và manifest Hugging Face
├── legacy/                    # code / cấu hình cũ (tầm nhìn v4, pipeline video YOLO-World + SAM 2 đầu tiên)
├── PROJECT_OVERVIEW.md        # tài liệu kỹ thuật + toàn bộ thí nghiệm
└── train_kaggle.ipynb, notebooke7cc063adf.ipynb   # notebook Kaggle (giữ ở gốc để Kaggle đồng bộ GitHub)
```

## Đánh giá video trên Kaggle (2026-11)

`launchers/kaggle_video_launch.py` tạo từng job độc lập cho CountVid gốc (CountGD-Box + SAM 2), SAM 3 video,
và các biến thể CountYOLO + SAM 2. Không cần chạy suy luận trên máy local.

- `countyolo_fusion`: checkpoint exnoise, exemplar SAM 3 mỗi 5 frame, lọc đỉnh yếu và ghép box YOLO-World/SAM 3
  với đỉnh density trước khi đưa vào SAM 2. Đây là biến thể thử nghiệm, chưa mặc định tốt hơn bản cũ.
- Đối chứng: `countyolo_old`, `countyolo_exnoise`, `countyolo_kf5`. `sam3_native` giữ ngưỡng phát hiện nội bộ mặc định 0.5;
  `sam3` dùng 0.3 để chẩn đoán. Cả hai báo riêng số ID thô và số ID xuất hiện ít nhất 3 frame.
- VideoCount: Crystals (7 video), Penguins (3), MOT20-Count (3). Frame/prompt theo CountVid; mỗi video/prompt có
  manifest frame, GT, số đếm, thời gian và log riêng. MAE/RMSE/bias/WAPE tính theo số vật duy nhất toàn video.
- Pilot: Penguins-1 và MOT20-01. Chọn cấu hình CountYOLO bằng MAE trên pilot; các video còn lại là heldout.
  Không coi pilot là kết quả độc lập để khẳng định khả năng tổng quát hóa.

Trạng thái bền vững: `experiments/video_tasks.json`. Công cụ theo dõi giới hạn 2 job GPU/tài khoản, chờ khi hết slot,
tải và kiểm tra kết quả, rồi gửi job tiếp. Nếu job lỗi, dừng gửi thêm trên tài khoản đó để kiểm tra log.
`--full-after-pilot` mở rộng đánh giá sau khi toàn bộ pilot hoàn tất; cấu hình được chọn và điểm pilot được ghi lại.

```powershell
# Gửi một job riêng (credential nằm trong kaggle/*.json, không commit)
.kaggle-tools/Scripts/python.exe launchers/kaggle_video_launch.py --username nadthien --engine countyolo_fusion --dataset penguins --videos penguins_1 --push

# Theo dõi/gửi tiếp; chỉ chạy một watcher cho mỗi ledger
.kaggle-tools/Scripts/python.exe -u tools/video_tasks.py --watch --full-after-pilot --interval 60 --max-active 2

# Đọc kết quả đã tải, không chạy model và không gọi mạng
python tools/compare_video_runs.py
python tools/test_video_pipeline.py
```

Kết quả: `outputs/video_benchmark/video_cases.csv`, `video_summary.json`.
Log watcher: `outputs/video_benchmark/watcher.log`; log và trace từng job ở `outputs/kaggle/`.
Các nhóm chưa đủ video luôn phải được đọc cùng trạng thái các task còn chờ; chưa phải benchmark toàn bộ dataset.
Checkpoint lớn được gắn qua dataset Kaggle, không nhúng vào script (giới hạn script 1 MB).
SDK được buộc dùng credential của tài khoản đã chọn để token toàn cục cũ không ghi đè legacy key.

Cache VideoCount riêng tư trên mỗi tài khoản: `<username>/videocount-benchmark`.
Tạo bằng `tools/cache_videocount.py`, kiểm tra bằng `tools/verify_video_cache.py`.
Cache chỉ chứa đúng frame được chọn theo CountVid, giữ nguyên byte JPEG; evaluator đọc manifest
`countyolo_sampling.json` và không lấy mẫu MOT20 lần thứ hai. Job mới không tải VideoCount từ Google Drive.

Baseline CountVid dùng thêm dataset checkpoint riêng tư `<username>/countvid-weights`.
Tạo bằng `tools/cache_countvid_weights.py --username <username>` sau khi được phép upload checkpoint.
Launcher xác minh SHA-256 trước suy luận; không tải CountGD-Box lặp lại từ Google Drive.


SAM3 setup audit (2026-11-08): `sam3_aligned` uses the exposed temporal settings
from Meta SAM3 commit `f1647e8d0c1d280991f4d1b7d974c2704537fe66`, with
Ultralytics pinned to `8.4.204`. This is still the Ultralytics backend, not an
official Meta reference run. It delays output by 15 frames to discard unstable
or duplicate tracks before counting them, removes masks that become empty after
non-overlap processing, and checks that checkpoint mapping loads every parameter.
Objects observed before a later track expiration remain counted.

`sam3_aligned_fp32` repeats the same video policy in FP32. That job also writes
`sam3_image_probe.json`: identical first frames from Penguins-1 and two Crystals
clips, in FP16 and FP32, with confidence fixed at 0.3 and numerical diagnostics.
The video profiles use detection/output thresholds of 0.5; `sam3_native` keeps
its old settings as a control. No thresholds are fitted to ground-truth counts.

New SAM3 jobs are in `experiments/video_diagnostics.json`; they are development
diagnostics and do not replace the original benchmark. See
`outputs/video_diagnostics/watcher.log` for dispatch and completion. The watcher
polls every 60 seconds while the local machine remains awake and connected.
Queued jobs are not running until their ledger state becomes `submitted`.

The application keeps `tracker_profile="legacy"` until GPU validation completes.
Select `tracker_profile="aligned"` explicitly for the repaired experimental
policy; the `sam3_aligned*` Kaggle configurations already do this.


`sam3_aligned_batch8` bounds each tracker state to8 newly detected objects
without limiting the total object count or changing resolution/thresholds.
Kaggle validation: Penguins1 count27 unchanged, peak allocation7.34 ->4.06GiB;
Penguins2 previously OOM, now completed94 frames with count54/59 and peak7.10GiB.
These are development diagnostics, not a claim of stability on all videos.
Artifacts: experiments/sam3_memory_tasks.json and outputs/sam3_memory_tasks/.


## Lưu checkpoint trên Hugging Face

Ba checkpoint CountYOLO (`seed3`, `exmix2`, `exnoise`) được lưu trong model repo
riêng tư [ThienDan2705/countyolo-checkpoints](https://huggingface.co/ThienDan2705/countyolo-checkpoints).
Git lưu manifest `checkpoints/huggingface.json` chứa commit cố định, kích thước và
SHA-256. Công cụ tải kiểm tra toàn bộ file trước khi thay checkpoint local.
SAM3, CountGD và pretrained backbone không thuộc lần upload này.

```powershell
# Dùng môi trường local hiện có, không cần cài lại model
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python.exe tools/download_checkpoints.py
```

Tài khoản cần quyền đọc repo riêng tư; trên Kaggle dùng secret `HF_TOKEN` nếu tải
trực tiếp từ Hugging Face. Không commit token. Script cũng chạy trong môi trường
công cụ có `huggingface_hub`, không cần import PyTorch.

Các bản trọng số local được giữ nguyên. File `.pth` được bỏ khỏi phiên bản Git mới;
lịch sử Git vẫn giữ phiên bản cũ. Hugging Face quản lý phiên bản trọng số trực tiếp,
không cần DVC hoặc Google Drive cho ba checkpoint này.

Launcher Kaggle hiện vẫn dùng URL commit Git lịch sử cố định cho seed3 (có kiểm tra
SHA-256) và dataset Kaggle riêng tư cho exnoise, nên không cần đổi job đang chạy.

Khi cập nhật trọng số: upload lên repo riêng tư, kiểm tra bản tải lại, rồi cập nhật
revision, size và SHA-256 trong manifest trước khi commit code.

Ledger `experiments/*tasks.json` và `video_diagnostics.json` là trạng thái máy local,
không commit. Lần đầu tạo queue bằng `tools/video_tasks.py --init` sau khi có dữ liệu;
kết quả và giới hạn đánh giá được ghi trong `experiments/LOG.md`.
