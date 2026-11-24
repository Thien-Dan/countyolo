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

## Kết quả tổng hợp (xem chi tiết đầy đủ tại [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md))

### 1. Ảnh tĩnh — FSC-147 Validation (1.286 ảnh, 29 lớp unseen) & Test lịch sử

| Mô hình | Chế độ | **Full-Val MAE ↓** | **F1 ↑** | **Absent MAE ↓** |
|---|---|---:|---:|---:|
| **OV-DQUO v2 (`ViT-L/14`, Chủ đạo)** | Multimodal 2×2 tiled (`@0.35`) / Text (`@0.26`) | **10.85** / **13.52** | **0.830** / **0.806** | **0.50** (`@0.35`) |
| **OV-DQUO v3.1 (`ViT-L/14`)** | Multimodal 2×2 tiled (`@0.35`) / Text (`@0.26`) | **11.08** / 13.79 | **0.824** / 0.798 | **0.50** (`@0.35`), **3.50** (`@0.26`) |
| **OV-DQUO v1 (`ViT-B/16`)** | Multimodal 2×2 tiled (`@0.35`) / Text (`@0.26`) | **13.32** / 14.56 | **0.802** / 0.786 | **0.00** (`@0.26`) |
| CountYOLO + DINOv2 (`exnoise`) | Text-Only Density (Val / Test chia ô) | 23.36 *(Test: 16.53)* | 0.665 | — |
| YOLO-World + CountGD Adapt | Text-Only (`@0.23`) | 27.21 *(zero-shot: 69.80)* | 0.641 | — |
| *LLMDet v3 (Đối chiếu họ DINO — lưu checkpoint)* | Text-Only 2×2 tiled (`@0.26`) | 10.42 | 0.818 | 0.00 |

### 2. Video — VideoCount Benchmark (11 video đối chiếu trực tiếp: 3 Penguins + 7 Crystals + MOT20-01)

| Hệ thống Video | Penguins (3) ↓ | Crystals (7) ↓ | MOT20-01 (1) ↓ | **Tổng 11 video MAE ↓** |
|---|---:|---:|---:|---:|
| **`OV-DQUO v2 + SAM 2.1` *(Chủ đạo)*** | 6.33 | **37.71** | 13.00 | **26.91** *(-42.2% vs CountVid)* |
| **`OV-DQUO v3.1 + SAM 2.1`** | 9.00 | 43.29 | **8.00** | **30.73** |
| **`OV-DQUO v1 + SAM 2.1`** | 7.33 | 53.00 | 12.00 | **36.82** |
| SAM 3 native (`conf=0.5`) | 20.67 | 48.86 | 12.00 | 37.82 |
| CountVid gốc (`CountGD-Box + SAM 2.1`) | 3.33 | 69.14 | 18.00 | 46.55 |
| CountYOLO cũ (`countyolo_old + SAM 2.1`) | **2.33** | 74.71 | **4.00** | 48.55 |

Chi tiết từng video, từng chế độ và đường dẫn checkpoint: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) và [experiments/BEST_MODELS.md](experiments/BEST_MODELS.md).

## Cài đặt

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -r requirements.txt
```
Checkpoint khuyến nghị cho ảnh: `checkpoints/density_head_best_vit_exnoise.pth`; seed3 được giữ làm đối chứng và cấu hình được chọn trên video pilot.
Sau đợt dọn local ngày 2026-11-17, `checkpoints/` giữ ba head `exnoise`, `exmix2`, `seed3` cùng weight CountVid. Các head ViT thử nghiệm độc nhất chuyển vào `checkpoints/archive/` (truyền đường dẫn này khi cần chạy lại); head đời trước DINOv2 và bản weight trùng đã được xóa. SAM2 tiny cũ nằm ở `legacy/weights/`, log gốc cũ ở `legacy/logs/`. Chi tiết file đã xóa/chuyển: `experiments/cleanup_20261117.json`.
`yolov8s-world.pt`, CLIP và DINOv2 tự tải lần đầu. Dữ liệu: FSC-147 ở `data/FSC147`, VideoCount ở `data/VideoCount`.

## Đếm video trên UI: pipeline CountYOLO + SAM2

Tab video dùng ba stage theo CountVid: proposal CountYOLO → mask độc lập SAM2 →
lọc ngắn hạn theo chuỗi frame có IoU >0.5 → theo dõi SAM2 dài hạn, chỉ thêm vật
khi mask không chồng lấn với các mask đã theo dõi. Cửa sổ mặc định w=3 frame.
UI hiển thị mask thật và ID; không còn dùng tracker nhẹ `appearance/position`.
Engine video được chọn riêng với engine ảnh để tránh nhầm luồng xử lý.

Bản tích hợp loại thêm mask rỗng và mask độc lập trùng nhau (IoU ≥0.85), xử lý
frame đầu rỗng và chia batch vật để giới hạn VRAM. Đây là adapter theo ba stage,
không phải bản sao nguyên trạng cấu hình benchmark trước đây. Các số đo video
trong bảng lịch sử ở trên không phải số đo của UI mới. Phân đoạn không bảo đảm
sửa được mọi proposal sai của density head.

Cài bổ sung vào môi trường đang dùng (không tạo venv hay tải lại PyTorch):

```powershell
$env:SAM2_BUILD_CUDA = "0"
uv pip install --python .venv/Scripts/python.exe wheel
uv pip install --python .venv/Scripts/python.exe --no-build-isolation -r requirements-video.txt
.\.venv\Scripts\python.exe tools/download_checkpoints.py --group countvid
```

SAM2 được ghim commit upstream trong `requirements-video.txt`. Trên Windows,
CUDA extension hậu xử lý là tùy chọn; model vẫn chạy khi không build extension.
UI dùng checkpoint SAM 2.1 large; tính toán chậm hơn tracker nhẹ. FPS lấy mẫu
và số frame tối đa áp dụng trước ba stage; giới hạn frame cắt từ đầu clip.

Chạy trực tiếp một thư mục frame bằng cùng pipeline trên UI:

```powershell
.\.venv\Scripts\python.exe tools/count_frames.py "data/VideoCount/Penguins/frames/penguins_1"
```

Với VideoCount, tên vật được đọc từ annotation gần thư mục frame (chỉ đọc tên lớp,
không dùng số đếm GT). Thư mục riêng cần thêm `--text "cube"`. Mặc định xử lý
toàn bộ frame, không lấy mẫu lại; dùng `--max-frames 10` để thử nhanh hoặc
`--dry-run` để kiểm tra input mà không chạy model. JSON lưu ở
`outputs/video_sam2/<tên thư mục>/results.json`; đổi bằng `--output`.

UI và terminal dùng chung bộ đọc/sắp xếp frame. Trên UI, đặt số frame tối đa
bằng **0** để lấy toàn bộ như terminal; với thư mục VideoCount, có thể để trống
text để lấy đúng tên lớp từ annotation. FPS không lấy mẫu lại thư mục frame.
UI có file JSON kết quả tải xuống và hiện checkpoint, text, min_mass, window,
batch thực tế. Hai JSON phải khớp `frame_content_sha256`, `checkpoint_sha256`,
`sam_checkpoint_sha256`, text và tham số trước khi so sánh số đếm. Kết quả mới
còn ghi hash mã pipeline để phát hiện trường hợp UI chưa được khởi động lại.

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
public [ThienDan2705/countyolo-checkpoints](https://huggingface.co/ThienDan2705/countyolo-checkpoints).
Git lưu manifest `checkpoints/huggingface.json` chứa commit cố định, kích thước và
SHA-256. Công cụ tải kiểm tra toàn bộ file trước khi thay checkpoint local.
Repo cũng lưu bản sao không chỉnh sửa của SAM3 và hai checkpoint chính của CountVid:
`countgd_box.pth` + `sam2.1_hiera_large.pt`. Nguồn, giấy phép riêng từng file và
checksum được ghi trong manifest và thư mục `licenses/` trên Hugging Face.
BERT, YOLO-World và DINOv2 vẫn được tải qua quy trình upstream hiện có.

```powershell
# Dùng môi trường local hiện có, không cần cài lại model
.\.venv\Scripts\python.exe tools/download_checkpoints.py

# Tải riêng SAM3 (~3,45 GB), CountVid (~1,85 GB), hoặc tất cả
.\.venv\Scripts\python.exe tools/download_checkpoints.py --group sam3
.\.venv\Scripts\python.exe tools/download_checkpoints.py --group countvid
.\.venv\Scripts\python.exe tools/download_checkpoints.py --group all
```

Repo trọng số public: người clone có thể tải trực tiếp, không cần đăng nhập hoặc
`HF_TOKEN`. Script cũng chạy trong môi trường công cụ có `huggingface_hub`,
không cần import PyTorch.

Các bản trọng số local được giữ nguyên. File `.pth` được bỏ khỏi phiên bản Git mới;
lịch sử Git vẫn giữ phiên bản cũ. Hugging Face quản lý phiên bản trọng số trực tiếp,
không cần DVC hoặc Google Drive để tải các checkpoint trong manifest.
SAM3 được đặt ở `sam3.pt`, CountVid ở `checkpoints/`; dùng `--output-dir`
để chọn một thư mục khác (ví dụ thư mục checkpoints của bản clone CountVid).
Hash các file lớn được tính theo luồng, không cần nạp cả checkpoint vào RAM.

Launcher Kaggle hiện vẫn dùng URL commit Git lịch sử cố định cho seed3 (có kiểm tra
SHA-256) và dataset Kaggle riêng tư cho exnoise, nên không cần đổi job đang chạy.

Khi cập nhật trọng số: upload lên repo Hugging Face, kiểm tra bản tải lại, rồi cập nhật
revision, size và SHA-256 trong manifest trước khi commit code.

Ledger `experiments/*tasks.json` và `video_diagnostics.json` là trạng thái máy local,
không commit. Lần đầu tạo queue bằng `tools/video_tasks.py --init` sau khi có dữ liệu;
kết quả và giới hạn đánh giá được ghi trong `experiments/LOG.md`.


## Th? nghi?m exemplar c? ki?m ch?ng (2026-11-17)

Nh?nh `reliable` l? th? nghi?m suy lu?n v?i checkpoint hi?n c?, ch?a thay m?c ??nh UI.
N? ch?n exemplar theo margin CLIP so v?i t? v?ng, density mass v? h? tr? detector
ho?c b? nh? ng?n h?n ?? x?c nh?n. ?i?m n?y kh?ng ph?i x?c su?t ???c hi?u ch?nh.
C?c exemplar tr?ng v? tr? kh?ng ???c l?p ?? ?? K; ?i?m b? lo?i c? weight 0.
Kh?ng c?n exemplar th? d?ng density th? ???c gi?m s?t khi train, kh?ng t? k?t lu?n
l?p v?ng m?t. Density th? v?n c? th? c? false positive, c?n ?o tr??c khi tri?n khai.

```powershell
.\.venv\Scripts\python.exe tools/count_frames.py data/VideoCount/Penguins/frames/penguins_1 --exemplar-policy reliable
.\.kaggle-tools\Scripts\python.exe launchers/kaggle_exemplar_launch.py --username ttdan2006 --push
```

Job pilot ri?ng t? `ttdan2006/countyolo-exemplar-reliability-20261117` so `control`
v? `reliable` tr?n to?n b? Penguins-1 v? Crystals ma2035_039, c?ng checkpoint,
frame v? SAM2. ??y l? development diagnostic, kh?ng ph?i ??nh gi? O2-Unique.
M?i ch? ?? c? `results.json`, `recognition_progress.csv` v? ?nh mask/ID m?u;
trace ch?a candidate CLIP, l? do lo?i/ch?n, fallback, proposal/mask v? ID m?i.
Tr?ng th?i local: `experiments/exemplar_reliability_20261117.json`.
Watcher ?? b?t ki?m tra m?i 60 gi?y, t?i ?a 4 gi? khi m?y c?n ch?y v? c? m?ng;
k?t qu? nh? ???c t?i v? `outputs/exemplar_reliability_kaggle/` v? ki?m tra
fingerprint gi?a hai ch? ?? tr??c khi t?o `comparison.json`.

### Open-world proposal experiment v2 (opt-in)

Run `tools/count_frames.py <frames> --text "penguins" --exemplar-policy openworld`.
Optional `--aliases` accepts explicit equivalent names. Defaults remain unchanged.
Paired evaluator also accepts `--modes control reliable openworld`.

- Density integral no longer caps the number of proposals. Local peaks use a
  mass threshold; an explicit 256-region compute budget reports truncation.
- Detector geometry replaces matching square peak boxes. Unmatched density
  peaks and detector-only boxes remain candidates, including coarse-only peaks.
- Exemplar initialization does not require a detector hit. Absolute CLIP margin
  strength blends refinement with coarse density instead of being normalized away.
- Every final region gets CLIP evidence. Two consecutive positive observations
  with one-to-one IoU AND appearance association are required before SAM2 receives
  it. These local region IDs are not unique-object IDs or global ReID.
- Explicit aliases share positive evidence and are excluded from competitors.
  Parent/child ambiguity remains unresolved; thresholds are not calibrated.

Important limits: the first frame has no confirmed proposals; combined with the
SAM2 persistence window this penalizes short appearances. Fast motion, missing
frames, small objects and inaccurate square fallback geometry remain risks.
This is not a trained bbox detector; no AP/recall improvement is established.
Per-region uncertain/confirmed states and proposal-budget truncation are saved
in `proposal_trace`. The already submitted Kaggle v1 experiment is unchanged.

### FSC147 proposal ablations (validation only)

`tools/eval_fsc147_proposals.py --per-class 2 --output outputs/fsc147_openworld/pilot_val`
selects two images per validation class with seed 1701. Use `--per-class 0` for
all 1,286 validation images. Test images and GT exemplar boxes are not used.

The evaluator separates density-count MAE/RMSE from proposal-count MAE and
one-to-one point precision/recall at 8 and 16 pixels in the 640x640 input space.
These localization diagnostics are not bbox AP or official FSC147 detection
metrics. FSC147 target annotations do not establish absence of other classes.

Ablations: original capped peaks, uncapped peaks (mass .2, budget 256), detector
geometry fusion, v2 density/raw proposals, and static CLIP-supported proposals.
Static evaluation disables temporal confirmation and clears region memory.
Artifacts include per-image scores, diagnostics, selected images, source copies
and checkpoint/image hashes. No validation thresholds are fitted by this script.

`tools/count_frames.py <frames> --exemplar-policy recall` isolates uncapped peaks
while preserving original density and self-exemplars. This experimental mode
does not enable CLIP gating or change SAM2 tracking. It keeps all valid density
peaks, not the CLIP branch's 256-crop budget. The default remains `self`.
The evaluator's `uncapped_all` ablation measures this exact decoder; use
`--baseline-only --per-class 0` to skip v2 CLIP inference while keeping the same
initialized text-encoder precision for the control.

### V3 fixes and isolated ablations

- `plateau`: original self-exemplars and density, all valid peaks, with connected
  flat maxima collapsed to one representative. No CLIP veto or temporal gate.
- `geometry`: plateau proposals with one-to-one detector geometry replacement.
  Original density anchors are retained in logs. A detector covering multiple
  anchors cannot delete them. No global NMS over density square boxes.
- `rank`, `weighted`, `blend`, `coarse_fallback`: experimental exemplar policies.
  Candidates are peak-centered crops bounded by nearest-neighbor spacing, with
  exact ROIAlign crops used for CLIP. The normal fallback is original self-exemplar.
  These policies are ablations, not recommended defaults.
- Offline `--v3 region` additionally tests ROI-average prototypes in the exact
  semantic crop, without adding model parameters or retraining the checkpoint.

Run `tools/eval_fsc147_proposals.py --per-class 0 --v3 rank weighted blend coarse_fallback --output <folder>`.
Use `--split test` only after freezing configuration; the default split is val.
`tools/export_exemplar_crops.py` reconstructs float32 ROI tensors and PNG previews
from recorded coordinates. Fallback cases are explicitly marked as unselected.
`tools/eval_prompt_sanity_v3.py --output <json>` runs two visually reviewed easy
negative images and a synthetic two-class scene; this is not a general benchmark.

Video CLI accepts `--exemplar-policy plateau` or `geometry`. The paired video
evaluator supports `--max-frames 6` for integration diagnostics; prefix runs
disable ground-truth count metrics and do not claim full-video accuracy.
Existing `self`, `recall` and submitted Kaggle payloads remain reproducible.

### Reviewed semantic supervision and diagnostic pilots

Negative or mixed-class supervision now requires `--verified-absent-json` in
`train_density_head.py`. Format: an image-name to reviewed absent-class list,
for example `{"a.jpg": ["elephants"]}`. A different FSC147 class name alone
does not establish absence. Unreviewed samples retain positive labels, skipped
negative requests are counted, and mosaic absence must hold in every tile.
Mixed validation likewise requires reviewed cross-class absence in both pools.

Legacy `tools/prompt_eval.py` and `tools/presence_eval.py` inferred absences and
are blocked by default. `--allow-unverified-legacy-diagnostic` permits historical
reproduction only and marks output labels unverified; those scores are not
validated open-world accuracy. No old embedded Kaggle payload is rewritten.

`tools/proposal_ceiling.py` measures nondeployable GT-oracle proposal recall with
one-to-one matching and duplicate-center removal.

### Supported paths after experiment cleanup (2026-11-18)

- Main CountYOLO: `models/density_yolo.py`, `train_density_head.py`, image/video
  evaluators and the current application. DINOv2 and adapted DINOv3 backbone
  support remain available through checkpoint metadata.
- Best retained DINOv3 + YOLO-World detection checkpoint:
  `outputs/semantic_path_audit/dinov3_e10_last.pth`, SHA256
  `05a90c22c05e51590e4e7b0d4b121753f8c306c5379fa09efcc57c093d6a39ef`.
  This local candidate does not replace the original default checkpoint.
- CountVid comparison, SAM2/SAM3 paths, backbone adaptation and semantic-logic
  safeguards remain maintained.

Standalone DINOv3-only, region/text heads, proposal-point/residual heads and the
small semantic fine-tuning pilot were retired after failed validation guards.
Their trainers, launchers, tests specific to removed heads, weights and feature
caches were removed. Small JSON evaluation records remain for historical audit;
paths to deleted code/weights in old records are historical, not runnable
instructions. The offset-only proposal pilot had a small localization gain but
no count MAE gain and is also retired. See
`experiments/cleanup_20261118.json` for the exact deletion inventory.

### SAM3 temporal identity association (experimental, opt-in)

`video/identity.py` stitches nonoverlapping SAM3 tracklets into counting identities
after video processing. Native track IDs, detections and masks remain unchanged.
Short gaps require consistent appearance and predicted motion; longer gaps require
boundary exit/entry and stricter appearance. At least three consecutive endpoint
observations are needed. Competing plausible matches are left separate. Gap limits
are in processed frames, not seconds. This cannot repair identity switches already
inside a native track or recover missed detections.

`video/identity_features.py` uses frozen, pinned DINOv3 box ROI embeddings. These
are generic appearance features, not a trained instance-ReID model. Small/crowded
objects, similar individuals, pose changes and moving cameras remain limitations.
CPU is the default feature device to avoid adding GPU memory pressure to SAM3.

```powershell
# Enable during a new run; default remains --identity-mode none.
.\.venv\Scripts\python.exe tools/eval_video_sam3.py --data-root data/VideoCount --dataset penguins --videos penguins_1 --weights sam3.pt --tracker-profile aligned --object-batch-size 8 --identity-mode dinov3 --identity-device cpu --results outputs/sam3_identity/live/results.json

# Or replay a previous result with its traces and original frames.
.\.venv\Scripts\python.exe tools/eval_sam3_identity.py --results <previous-results.json> --output outputs/sam3_identity/replay --device cuda
```

Returned `identity` includes native-to-counting-ID mapping, accepted links and
ambiguous candidates. Final stitched counts are offline; frame callbacks continue
to display native SAM3 IDs. `baseline_count` is always reported separately.
Verification and current limitations: `experiments/sam3_identity_20261118.json`.

### YOLO development diagnostics (2026-11-19)

`tools/eval_yolo_recovery.py`, `tools/compare_yolo_detectors.py`,
`tools/audit_exemplar_oracle.py` and `tools/eval_detector_exemplar.py` are isolated
development tools, not application policies. Multiscale proposal recovery and
detector-supported exemplar selection failed pilot retention and are not enabled.
The official-exemplar-center diagnostic uses annotation input and must not be
reported as text-only accuracy. Existing best checkpoint and defaults remain.
Findings and limitations: `experiments/yolo_improvement_pilot_20261119.json`.

### Region/text exemplar ranker training (2026-11-19)

`models/exemplar_ranker.py` learns a bounded residual around frozen CLIP region
features using DINOv3 ROI features and explicit text similarity. A small quality
branch uses four local cues. It starts with zero adjustment and exact self
fallback, ranks actual coarse-peak candidates, and only changes exemplar
locations. It is not a presence classifier or final proposal filter.

`tools/train_exemplar_ranker.py` uses3,647 official FSC147 train images from89
classes, excludes12 reviewed images for selector-heldout checks, and adds30
reviewed cross-class mosaics with two target prompts each. Point-center ranking,
positive-region class supervision and residual regularization train only the
ranker for2,000 fixed updates. YOLO/DINO/density/refine/matchers and CLIP remain
frozen. Other FSC class names are not used as image-level absence labels.

The last checkpoint is evaluated on pilot58, reviewed mixed scenes and absence
cases. Full validation runs only after retention and improvement guards pass.
No application defaults change automatically. Submission and source hashes:
`experiments/exemplar_ranker_kaggle_20261119.json`.

### CountVid detector replacement (2026-11-19)

The legacy `YOLOWorldCounting` experiment is a text-only loss adaptation, not a
full reproduction of CountVid. The new `YOLOWorldCountVid` branch adds explicit
exemplar ROI/text conditioning and counting-to-box training. Its video entry
point executes the pinned original CountVid algorithm with only the detector
functions replaced. Read
`experiments/COUNTVID_YOLOWORLD_CONTRACT_20261119.md` for retained operations,
unavoidable detector differences, verification and training status. This model
has not completed training or full video evaluation and is not promoted.
