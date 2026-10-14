# CountYOLO — Tổng Quan Dự Án

> **Cập nhật lần cuối:** 2026-10-14
> Mô tả kiến trúc **đang chạy trong code** (`models/density_yolo.py`). Bản tốt nhất: arch `vit` (mục 0);
> các mục 2–9 mô tả bản `simcentric` mà `vit` mở rộng.
> `README.md` và `configs/countyolo_fsc147.yaml` là tầm nhìn v4 đầy đủ (ResNet-50, RepVL-PAN, NMSFreeHead,
> LLM attributes…) — **chưa được triển khai**, không script nào đọc file YAML.

---

## 0. Bản Tốt Nhất: YOLO-World + DINOv2-small (arch `vit`)

`simcentric` + DINOv2-small đóng băng (ảnh resize 560 → lưới patch 40×40): feature ViT chiếu 64 kênh ghép vào
Density Head, thêm similarity Self-Exemplar trên feature ViT cho Refine Head. Chỉ train các head (~0.6M tham số).

| FSC-147 test | MAE | Ghi chú |
|---|---|---|
| `vit` (3 seed) | **19.39 ± 0.57** | checkpoint `checkpoints/density_head_best_vit_seed3_ep19_valmae21.44.pth` (test 18.90) |
| `vit`, exemplar GT | ~18.0 | |
| `simcentric` | 21.4 – 25.0 | 2 lần train |
| CountGD-Box (paper CountVid, text) | 15.01 | tham chiếu |

Tốc độ (Kaggle T4, batch 1, ảnh 640): `vit` 43.5 ms (AMP, 23 FPS); `simcentric` 17.1 ms; CountGD-Box 422 ms (AMP).
Sai số dồn vào ~9 ảnh test > 500 vật (bỏ 5 ảnh tệ nhất: MAE 14.15). Đã thử và **không** cải thiện: mở neck
YOLO-World, DINOv2-base, ViT 840px, mosaic 4×4, Canonical-Scale Zoom.

```bash
# Train (Kaggle): python kaggle_launch.py --username <user> --vit --push
python train_density_head.py --arch vit
python eval_fsc147.py --mode density-count --checkpoint density_head_best_vit_seed3_ep19_valmae21.44.pth
python benchmark_speed.py   # đo tốc độ các biến thể
```

`countvid_ext/`: Canonical-Scale Tiling + NMS cắm vào Stage 1 của CountVid (Crystals MAE 69.1 → 50.0,
Penguins 3.33 → 1.67), chạy bằng `kaggle_countvid_launch.py`.

---

## 1. Mục Tiêu

Đếm vật thể theo text prompt (zero-shot) trên FSC-147, không fine-tune backbone:
YOLO-World đóng băng cung cấp feature, chỉ học các head nhỏ phía sau. Hướng mở rộng: exemplar few-shot,
video (SAM 2.1, `count_in_videos_yoloworld.py` / `eval_mot20.py`).

---

## 2. Kiến Trúc (`DensityYOLOWorld`)

```
Ảnh 640x640 ──► YOLO-World (frozen, eval, no_grad, nc=1 = text prompt của ảnh)
                  │ hook trước lớp Detect: P3 (128, 80x80), P4 (256, 40x40), P5 (512, 20x20)
                  │ score map region-text P3/P4/P5 ─► Text Map (1, 80x80)
Ảnh ──► Sobel ──► Edge Map (1, 80x80)

Bước 1  [P3 | Text | Edge] (130) ──► Density Head ──► density thô (80x80)
                                                        │ top-5 đỉnh cục bộ (hoặc điểm cho trước)
Bước 2  Self-Exemplar Matcher P3 & P4: cosine giữa feature tại 5 exemplar và toàn map
          ──► similarity P3 (mean, max) + similarity P4 (mean, max, upsample về 80x80)
        [P3 nén 16 kênh + Dropout2d 0.5 | Text | Edge | density thô | sim P3 | sim P4] (23)
          ──► Refine Head ──► density cuối (count = sum)
Bước 3  [P3, P4, P5 nén 32 kênh mỗi mức | sim P3 | density thô] (99)
          ──► Scale Head ──► log2 kích thước vật (px trong ảnh 640) mỗi ô
```

Ý tưởng chính:

- **Text Map** cho head biết *lớp nào* cần đếm (dùng chính xác suất region-text của YOLO-World).
- **Self-Exemplar**: text của YOLO-World yếu với nhiều lớp FSC-147, nhưng vật cần đếm luôn lặp lại trong
  ảnh → độ giống thị giác với chính các vật đó là tín hiệu mạnh hơn và không phụ thuộc lớp.
- **Simcentric**: lớp val/test không xuất hiện trong train, nên P3 bị nén + dropout mạnh ở Refine Head để
  head không học thuộc ngoại hình lớp train mà dựa vào similarity, density thô, text, edge.
- **Scale Head** phục vụ **Canonical-Scale Zoom** lúc suy luận (mục 6).

### Các module (`models/density_yolo.py`)

| Module | Vai trò | Ghi chú |
|---|---|---|
| `SobelEdgeExtractor` | Biên ảnh, chuẩn hoá [0,1] theo từng ảnh | Không có tham số học |
| `DensityHead` | Conv3x3(128)-BN-ReLU → Conv3x3(64)-BN-ReLU → Conv1x1 → ReLU | Dùng cho cả density thô và Refine Head; lớp cuối init nhỏ |
| `SelfExemplarMatcher` | Conv1x1 chiếu về 64 chiều, cosine với K=5 exemplar → [mean, max] | Đỉnh yếu hơn 10% đỉnh mạnh nhất bị coi là nền, thay bằng lặp lại đỉnh thật |
| `ScaleHead` | Dự đoán log2 kích thước vật | Bias init 5.5 (~45px) |
| `DensityYOLOWorld` | Ghép tất cả; `forward(x, class_names, exemplar_points, return_aux)` | `save_head`/`load_head` chỉ lưu các head; arch suy ra từ checkpoint |

Quy ước đơn vị: head dự đoán density × `DENSITY_SCALE` (100) để tránh ReLU chết; `forward()` chia lại nên
density trả về có `sum ≈ số vật`. Gradient không đi vào YOLO; density thô đưa vào bước 2/3 qua `detach()`.

Tương thích checkpoint: `coarse` (chỉ Density Head), `selfex` (Refine Head trên 130 + sim P3),
`simcentric` (mặc định). Checkpoint không có `scale_head` → tắt Scale Head/zoom.

---

## 3. Dữ Liệu (`datasets/fsc147_dataset.py`)

- Ảnh `images_384_VarV2` kéo giãn về 640×640. Points `[x, y]` **đã ở không gian ảnh 384** (đã kiểm chứng với
  `gt_density_map_adaptive_384_VarV2`), chỉ scale theo 640/W, 640/H.
- Exemplar box: 4 điểm → `[x1, y1, x2, y2]` trong không gian 640.
- Density GT: map adaptive chính thức, resize INTER_AREA về 80×80 rồi chuẩn hoá tổng = count
  (thiếu thư mục GT → Gaussian từ points, `datasets/density_generator.py`).
- `size_map`: log2 trung vị `sqrt(w·h)` của exemplar box, hằng trên toàn ảnh — nhãn cho Scale Head.
- Augment khi train: hflip; **mosaic 2×2 cùng lớp** (p=0.3) — mỗi ô thu nhỏ 2 lần, density sum-pool giữ tổng,
  size_map − 1.

---

## 4. Huấn Luyện (`train_density_head.py`)

| Thành phần | Chi tiết |
|---|---|
| Tham số học | Density Head, 2 Matcher, P3 compress, Refine Head, Scale Head |
| Loss | MSE(density cuối) + 0.5·MSE(density thô) + 25·region-count L1 (vùng 4/8/16/80 ô) + 2·L1 log2 kích thước (trọng số theo density GT) |
| Exemplar khi train | 50% ảnh dùng 5 GT point ngẫu nhiên, 50% dùng đỉnh density thô (như suy luận) |
| Optimizer | AdamW lr 1e-4, wd 1e-4, CosineAnnealing, AMP |
| Chọn checkpoint | `density_head_best.pth` theo val MAE (zero-shot); `density_head_last.pth` mỗi epoch |
| Fine-tune | `--init <ckpt>`: nạp head có sẵn, mốc best = val MAE của checkpoint đó |

---

## 5. Đánh Giá (`eval_fsc147.py`)

| `--mode` | Đếm bằng |
|---|---|
| `density-count` | `sum(density)`; `--exemplars self|gt`; `--zoom <px>` bật Canonical-Scale Zoom |
| `density-nms` | Density-Guided Soft-NMS (`utils/custom_nms.py`) trên box thô của cùng lần forward |
| `soft-nms` | Soft-NMS thường, baseline cho `density-nms` |
| `zero-shot` | YOLO-World + NMS mặc định |

In MAE, RMSE, bias, MAE theo nhóm kích thước vật (0–24 / 24–40 / ≥40 px), MAE density thô, chất lượng Scale Head.

**Kết quả (checkpoint epoch 25, trước khi sửa chọn exemplar):**

| | MAE |
|---|---|
| Val (self) | 23.84 |
| Test (self) | 21.39 |
| Test (exemplar GT) | 19.35 |
| Test zoom 32 / 44 / 56 | 20.97 / 22.02 / 25.89 |

Sai số dồn vào vật < 24px (test MAE ~176, đếm thiếu ~160). Fine-tune 15 epoch lr 1e-5 không cải thiện;
val MAE dao động mạnh giữa các epoch (23.8–37.8).

---

## 6. Canonical-Scale Zoom (`utils/scale_zoom.py`)

Từ Scale Head ước lượng kích thước vật; nếu vật nhỏ hơn `target / min_zoom`, chia density thành k×k ô,
ô nào có vật nhỏ thì cắt từ ảnh gốc, phóng về 640, đếm lại và thay phần density của ô đó (density có tính cộng).
Giới hạn: FSC-147 chỉ có ảnh 384px, nên phóng to không thêm chi tiết thật.

---

## 7. Chạy Trên Kaggle (`kaggle_launch.py`)

Đóng gói source vào một script kernel (không phụ thuộc GitHub), train rồi eval test/val và các mức zoom.

```bash
# Windows: cần PYTHONUTF8=1 để in tiếng Việt
python kaggle_launch.py --username nadthien --push              # train từ đầu (50 epoch, lr 1e-4)
python kaggle_launch.py --username nadthien --finetune --push   # fine-tune từ output kernel train (15 epoch, lr 1e-5)
kaggle kernels status nadthien/countyolo-density-train
kaggle kernels output nadthien/countyolo-density-train -p <dir>
```

`train_kaggle.ipynb` clone repo từ GitHub nên chỉ chạy được code **đã push**.

---

## 8. Cấu Trúc Thư Mục

```
countyolo/
├── models/density_yolo.py         # DensityYOLOWorld + các head
├── datasets/fsc147_dataset.py     # FSC-147 loader, mosaic, size_map
├── datasets/density_generator.py  # Gaussian density (dự phòng)
├── utils/custom_nms.py            # Density-Guided Soft-NMS
├── utils/scale_zoom.py            # Canonical-Scale Zoom
├── train_density_head.py          # Train / fine-tune
├── eval_fsc147.py                 # Đánh giá FSC-147
├── visualize_density.py           # So sánh density dự đoán vs GT
├── kaggle_launch.py               # Đóng gói + đẩy kernel Kaggle
├── train_kaggle.ipynb             # Notebook Kaggle (clone GitHub)
├── count_in_videos_yoloworld.py   # Video: YOLO-World + SAM 2.1
├── eval_mot20.py                  # Đánh giá MOT20
└── configs/                       # countyolo_fsc147.yaml (chưa dùng), sam2.1/
```

---

## 9. Lịch Sử

| Phiên bản | Thay đổi |
|---|---|
| P4 → P3 | Density Head chuyển sang P3 (80×80) để phân giải vật nhỏ |
| Edge Map | Thêm Sobel làm kênh đầu vào |
| Text Map | Thêm xác suất region-text → head biết lớp cần đếm (130 kênh) |
| Self-Exemplar | Density 2 bước: đỉnh density thô làm exemplar → similarity → Refine Head |
| Simcentric | Similarity P3+P4, nén + dropout P3 để tổng quát sang lớp chưa thấy |
| Scale Head + Zoom | Ước lượng kích thước vật, đếm lại vùng vật nhỏ ở kích thước chuẩn |
| Mosaic, region loss | Ảnh đông giả cùng lớp; loss số đếm theo vùng đa tỷ lệ |
| 2026-09-29 | Sửa chọn exemplar khi < 5 đỉnh thật; thêm `--init` fine-tune |
