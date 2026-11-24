# Bảng Tổng Hợp Kết Quả Đánh Giá (MAE / RMSE / F1 / Absent)

Tài liệu tổng hợp toàn bộ kết quả đo thực tế của các mô hình đếm vật thể trong dự án trên **FSC-147 Validation (1.286 ảnh, 29 lớp unseen — không chạm tập test)** và **VideoCount Benchmark (11 video đối chiếu trực tiếp: 3 Penguins + 7 Crystals + MOT20-01)**.

- **Bản chủ đạo hiện tại (Flagship):** **`OV-DQUO v2` (`EVA02-CLIP-L/14 @672`)** — dẫn đầu nhánh non-DINO trên ảnh tĩnh (**10.85 MAE / 0.830 F1**) và dẫn đầu tuyệt đối trên video (**26.91 MAE**, vượt **CountVid gốc `-19.64 MAE` (-42.2%)** và vượt **SAM 3 `-10.91 MAE` (-28.8%)**).
- **4 nhánh mã nguồn giữ lại:** `OV-DQUO` (`v2` chủ đạo), `YOLO-World / CountYOLO`, `CountVid`, `SAM 3`.

---

## 1. Kết quả trên ảnh tĩnh — FSC-147 Full-Val (1.286 ảnh, 29 lớp unseen)

| Mô hình / Nhánh | Backbone & Độ phân giải | Chế độ đánh giá (Ngưỡng) | **MAE ↓** | **RMSE ↓** | **F1 ↑** | **Absent MAE ↓** | Nguồn JSON |
|---|---|---|---:|---:|---:|---:|---|
| **OV-DQUO v2 *(Chủ đạo)*** | `EVA02-CLIP-L/14` @672 (`ep12`) | **Multimodal 2×2 tiled (`@0.35`)** | **10.85** | **56.36** | **0.830** | **0.50** | `experiments/ov_dquo_v2_vitl14_mm_result_20261122.json` |
| OV-DQUO v2 | `EVA02-CLIP-L/14` @672 (`ep12`) | Multimodal raw (`@0.35`) | 11.72 | 65.19 | 0.821 | 0.50 | `experiments/ov_dquo_v2_vitl14_mm_result_20261122.json` |
| OV-DQUO v2 | `EVA02-CLIP-L/14` @672 (`ep12`) | Exemplar-Only raw (`@0.35`) | 12.05 | 66.39 | 0.818 | — | `experiments/ov_dquo_v2_vitl14_mm_result_20261122.json` |
| OV-DQUO v2 | `EVA02-CLIP-L/14` @672 (`ep12`) | Text-Only raw (`@0.26`) | **13.52** | 65.04 | **0.806** | 29.00 (`0.50 @0.35`) | `experiments/ov_dquo_v2_vitl14_mm_result_20261122.json` |
| **OV-DQUO v3.1** | `EVA02-CLIP-L/14` @672 (`ep8`) | **Multimodal 2×2 tiled (`@0.35`)** | **11.08** | 56.67 | **0.824** | **0.50** | `experiments/ov_dquo_v31_vitl14_mm_result_20261122.json` |
| OV-DQUO v3.1 | `EVA02-CLIP-L/14` @672 (`ep8`) | Multimodal raw (`@0.26`) | 12.26 | 60.79 | 0.809 | **3.50** | `experiments/ov_dquo_v31_vitl14_mm_result_20261122.json` |
| OV-DQUO v3.1 | `EVA02-CLIP-L/14` @672 (`ep8`) | Exemplar-Only raw (`@0.35`) | 12.93 | 67.46 | 0.806 | — | `experiments/ov_dquo_v31_vitl14_mm_result_20261122.json` |
| OV-DQUO v3.1 | `EVA02-CLIP-L/14` @672 (`ep8`) | Text-Only raw (`@0.26`) | 13.79 | 68.39 | 0.798 | **3.50** | `experiments/ov_dquo_v31_vitl14_mm_result_20261122.json` |
| **OV-DQUO v1** | `EVA02-CLIP-B/16` @768 (`ep4`) | **Multimodal 2×2 tiled (`@0.35`)** | **13.32** | 59.38 | **0.802** | **0.00** (`@0.26`) | `experiments/ov_dquo_v1_vitb16_mm_result_20261122.json` |
| OV-DQUO v1 | `EVA02-CLIP-B/16` @768 (`ep4`) | Multimodal raw (`@0.26`) | 13.77 | 58.76 | 0.793 | 0.00 | `experiments/ov_dquo_v1_vitb16_mm_result_20261122.json` |
| OV-DQUO v1 | `EVA02-CLIP-B/16` @768 (`ep4`) | Exemplar-Only raw (`@0.35`) | 14.29 | 69.32 | 0.790 | — | `experiments/ov_dquo_v1_vitb16_mm_result_20261122.json` |
| OV-DQUO v1 | `EVA02-CLIP-B/16` @768 (`ep4`) | Text-Only raw (`@0.26`) | 14.56 | 67.52 | 0.786 | 0.00 | `experiments/ov_dquo_v1_vitb16_mm_result_20261122.json` |
| **CountYOLO + DINOv2** (`exnoise`) | `YOLOv8s-World` + `DINOv2-S` @640 | Density map (Text-Only, Val) | 23.36 | 77.85 | 0.665 *(peaks r=16)* | — | `experiments/fsc147_openworld_validation_20261117.json` |
| **YOLO-World CountVid Point** | `YOLOv8s-World` @640 (`ep7`) | Multimodal Text+Ex (`@0.23`) | 26.53 | 109.13 | 0.226 | 0.00 | `experiments/countvid_yoloworld_point_results_20261120.json` |
| **YOLO-World CountGD Adapt** | `YOLOv8s-World` @640 (`ep10`) | Text-Only (`@0.23`) | 27.21 | 88.29 | 0.641 | 496.5 *(high FP)* | `experiments/yoloworld_counting_final_results_20261119.json` |
| YOLO-World Zero-Shot | `YOLOv8s-World` @640 | Baseline Text-Only (`@0.23`) | 69.80 | 143.38 | 0.062 | — | `experiments/yoloworld_counting_final_results_20261119.json` |
| *LLMDet v3 (Đối chiếu họ DINO — chỉ giữ checkpoint)* | `llmdet_tiny` (`ep9`) | Text-Only 2×2 tiled (`@0.26`) | **10.42** | **51.18** | 0.818 | 0.00 | `experiments/llmdet_v3_text_result_20261121.json` |
| *LLMDet v3 (Đối chiếu họ DINO)* | `llmdet_tiny` (`ep9`) | Text-Only raw (`@0.26`) | 10.94 | 58.04 | 0.810 | 0.00 | `experiments/llmdet_v3_text_result_20261121.json` |

> **Ghi chú FSC-147 Test lịch sử (từ `README.md` / `LOG.md`):** CountYOLO + DINOv2 `exnoise` đạt **17.89 MAE** (raw) / **16.53 MAE** (chia ô 2×2); CountGD-Box gốc (CountVid) đạt **15.01 MAE**. Các nhánh OV-DQUO chưa đánh giá trên tập Test để giữ kỷ luật chọn mô hình hoàn toàn trên Validation.

---

## 2. Kết quả trên Video — VideoCount Benchmark (11 video đối chiếu trực tiếp)

Tất cả các hệ thống dưới đây chạy trên **cùng 11 video lấy mẫu giống hệt nhau** (`Penguins`: 3 video, `Crystals`: 7 video, `MOT20-01`: 1 video; hai video siêu dài `MOT20-02` và `MOT20-05` đều vượt trần thời gian 12h/4h ở cả CountVid gốc lẫn các pipeline SAM 2.1).

### 2.1. Tổng hợp MAE theo từng tập video

| Hệ thống Video | Penguins (3) MAE ↓ | Crystals (7) MAE ↓ | MOT20-01 (1) MAE ↓ | **TỔNG 11 VIDEO MAE ↓** | So với CountVid gốc |
|---|---:|---:|---:|---:|---|
| **`OV-DQUO v2 + SAM 2.1 Large` *(Chủ đạo)*** | 6.33 | **37.71** | 13.00 | **26.91** | **Thắng `-19.64 MAE` (-42.2%)** |
| **`OV-DQUO v3.1 + SAM 2.1 Large`** | 9.00 | 43.29 | **8.00** | **30.73** | **Thắng `-15.82 MAE` (-34.0%)** |
| **`OV-DQUO v1 + SAM 2.1 Large`** | 7.33 | 53.00 | 12.00 | **36.82** | Thắng `-9.73 MAE` (-20.9%) |
| **SAM 3 native** (`Ultralytics`, conf `0.5`) | 20.67 | 48.86 | 12.00 | 37.82 | Thắng `-8.73 MAE` (-18.8%) |
| **CountVid gốc** (`CountGD-Box + SAM 2.1`) | 3.33 | 69.14 | 18.00 | 46.55 | *(baseline)* |
| **CountYOLO cũ** (`countyolo_old + SAM 2.1`) | **2.33** | 74.71 | **4.00** | 48.55 | Kém `+2.00 MAE` (+4.3%) |

### 2.2. Chi tiết từng video (`pred (sai số)`)

| Video | Prompt | GT | **OV-DQUO v2** (`L/14`) | **OV-DQUO v3.1** (`L/14`) | OV-DQUO v1 (`B/16`) | CountVid gốc | CountYOLO cũ | SAM 3 native |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `penguins_1` | `penguin` | 28 | **27 (`-1`)** | 25 (`-3`) | 24 (`-4`) | 26 (`-2`) | **28 (`0`)** | 39 (`+11`) |
| `penguins_2` | `penguin` | 59 | 42 (`-17`) | 38 (`-21`) | 44 (`-15`) | 51 (`-8`) | **55 (`-4`)** | 92 (`+33`) |
| `penguins_3` | `penguin` | 38 | **37 (`-1`)** | 35 (`-3`) | 35 (`-3`) | **38 (`0`)** | 35 (`-3`) | 56 (`+18`) |
| `ma2035_023` | `white crystal` | 111 | **79 (`-32`)** | 61 (`-50`) | 35 (`-76`) | 1 (`-110`) | 0 (`-111`) | 8 (`-103`) |
| `ma2035_024` | `white crystal` | 154 | 1 (`-153`) | 0 (`-154`) | 0 (`-154`) | 1 (`-153`) | 0 (`-154`) | 6 (`-148`) |
| `ma2035_039` | `white crystal` | 27 | **22 (`-5`)** | 20 (`-7`) | 17 (`-10`) | 10 (`-17`) | 0 (`-27`) | 29 (`+2`) |
| `ma2035_040` | `white crystal` | 37 | **29 (`-8`)** | 26 (`-11`) | 17 (`-20`) | 9 (`-28`) | 0 (`-37`) | 33 (`-4`) |
| `ma2035_069` | `white crystal` | 109 | **84 (`-25`)** | 80 (`-29`) | 64 (`-45`) | 11 (`-98`) | 0 (`-109`) | 62 (`-47`) |
| `ma2035_072` | `white crystal` | 77 | **38 (`-39`)** | 31 (`-46`) | 17 (`-60`) | 2 (`-75`) | 0 (`-77`) | 49 (`-28`) |
| `ma2035_169` | `white crystal` | 8 | **6 (`-2`)** | 2 (`-6`) | 2 (`-6`) | 11 (`+3`) | 0 (`-8`) | 18 (`+10`) |
| `MOT20-01` | `human` | 80 | 93 (`+13`) | **88 (`+8`)** | 68 (`-12`) | 98 (`+18`) | **76 (`-4`)** | 92 (`+12`) |

---

## 3. Danh sách Checkpoint đã lưu cố định trong `checkpoints/`

| Vai trò | Đường dẫn checkpoint | Dung lượng | Ghi chú sử dụng |
|---|---|---:|---|
| **Chủ đạo (Flagship)** | `checkpoints/ov_dquo_v2_vitl14_mm_ep12_fullval10.85.pth` | 90.9 MB | Dùng cùng `OVDQUO_ViT-L14_LVIS.pth` qua `models/ov_dquo_video_detector.py` (`load_ov_dquo_video_detector`) |
| **Phụ (Absent thấp ở `@0.26`)** | `checkpoints/ov_dquo_v31_vitl14_mm_ep8_fullval11.08.pth` | 93.7 MB | Dùng cùng `OVDQUO_ViT-L14_LVIS.pth`, giảm lỗi absent 8.3× ở ngưỡng `@0.26` |
| **Bản nhẹ (`ViT-B/16`)** | `checkpoints/ov_dquo_v1_vitb16_mm_ep4_fullval13.32.pth` | 81.4 MB | Dùng cùng `OVDQUO_ViT-B16_LVIS.pth`, `absent @0.26 = 0.00` |
| **Tracker Video SAM 2.1** | `checkpoints/sam2.1_hiera_large.pt` | 856.5 MB | Dùng cho pipeline 4 tầng `video/ov_dquo_sam2_pipeline.py` và CountVid |
| **CountYOLO (Density)** | `checkpoints/density_head_best_vit_exnoise.pth` | 2.3 MB | Head nhẹ cho nhánh YOLO-World + DINOv2 (`exmix2`, `seed3` giữ đối chứng) |
| **CountVid Baseline** | `checkpoints/countgd_box.pth` | 910.5 MB | Trọng số CountGD-Box đối chứng chính thức |
| **Lưu trữ đối chiếu DINO** | `checkpoints/llmdet_v3_text_fullval10.42.pth` | 136.8 MB | Giữ làm mốc tham chiếu (code thử nghiệm đã dọn sạch khỏi nhánh chính) |
