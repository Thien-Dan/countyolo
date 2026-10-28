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
Checkpoint tốt nhất có sẵn: `checkpoints/density_head_best_vit_seed3_ep19_valmae21.44.pth`.
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
├── checkpoints/               # checkpoint head (chỉ bản tốt nhất được commit)
├── legacy/                    # code / cấu hình cũ (tầm nhìn v4, pipeline video YOLO-World + SAM 2 đầu tiên)
├── PROJECT_OVERVIEW.md        # tài liệu kỹ thuật + toàn bộ thí nghiệm
└── train_kaggle.ipynb, notebooke7cc063adf.ipynb   # notebook Kaggle (giữ ở gốc để Kaggle đồng bộ GitHub)
```
