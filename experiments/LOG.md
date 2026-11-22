# Nhật ký tối ưu CountYOLO

Vòng lặp: **xem xét hệ thống → tối ưu → fine-tune nếu cần → đo → ghi lại → lặp**. Mỗi vòng thêm một mục ở cuối file.
Theo dõi kernel: `python tools/kaggle_watch.py` (trạng thái) · `python tools/kaggle_watch.py --fetch <tài khoản>/<kernel>`.

## Mục tiêu (so với bộ đếm Stage 1 của CountVid = CountGD-Box)

| # | Tiêu chí | Đo bằng | Mốc CountGD-Box / CountVid | Mục tiêu |
|---|---|---|---|---|
| T1 | Độ chính xác đếm ảnh | FSC-147 test MAE (chỉ text) | 15.01 | **≤ ~15–16** |
| T2 | Không nhầm loại đối tượng | `tools/prompt_eval.py`: đếm TB khi hỏi sai lớp · sai số tương đối mosaic 2 lớp | SAM 3: 0.4 · 21% | **≤ 1 · ≤ 25%** |
| T3 | Tốc độ lớp detect | `tools/benchmark_speed.py`, T4, batch 1, AMP | 422 ms (2.4 FPS) | **≤ 42 ms (≥ 10×)** |
| T4 | Video (khi T1–T3 đạt) | VideoCount: Crystals / Penguins / MOT20 MAE | 69.1 / 3.33 / 50.0 | ngang CountVid |

## Hiện trạng (cập nhật 2026-10-20 tối)

| Mô hình | T1 MAE | T2 sai lớp · mosaic | T3 (T4, AMP) | Ghi chú |
|---|---|---|---|---|
| CountYOLO `vit` (seed3) | **18.90** (17.98 có chia ô) | 65.1 · 107% ❌ | 43.5 ms | đếm tốt, bỏ qua text |
| CountYOLO `vit` + `--fast` (fp16 + compile) | 19.27 (seed 1; thường 19.25) | 65.1 · 107% ❌ | **23.1 ms ✅** | T3 đạt, 18× CountGD-Box |
| CountYOLO `vit` + exemplar đúng lớp (GT, giới hạn trên của SAM 3 cấp) | 17.95 | — · 98% (seed 3) / 75% (seed 1) | 23 ms + SAM 3 | exemplar giúp T1 ~1 MAE |
| `exmix` bản cuối + exemplar đúng lớp (GT; thật: SAM 3) | 24.37 (exemplar GT) · 19.99 (tự chọn) | 53.0 (không SAM 3) · **22% ✅** | 23 ms + SAM 3 | mosaic đạt, ảnh 1 lớp đếm thiếu |
| **`exmix2` (ep 20) + exemplar đúng lớp (SAM 3)** | 19.12 (tự chọn) · 22.76 (exemplar GT) | 55.1 (không SAM 3) · **21% ✅** | 23 ms + SAM 3 | cân bằng nhất; `checkpoints/density_head_best_vit_exmix2_ep20.pth` |
| **`exmix2` + CLIP kiểm tra (mix, ngưỡng val sai lớp ≤ 0.5)** | 20.86 (chặn nhầm 4.6%) | **1.58** · **26%** (exemplar CLIP) | ~48 ms + lượt exemplar | không cần SAM 3 |
| **`exmix2` + CLIP một lượt (auto, ngưỡng 0.1279) — trong UI** | 20.61 (chặn nhầm 4.5%; không cổng 18.98) | **1.37** · **26%** | 49.9 ms ❌ | UI (Windows: CLIP không compile, 13 hay 25 vùng như nhau) |
| **`exnoise` (test 17.89, chia ô 16.53) + CLIP K=6, từ vựng lớn (ngưỡng 0.0198) — UI** | **18.38** (chặn nhầm 3.0%; không cổng 17.16) | **0.77 ✅** · 28.9% | **39.5 ms ✅** | tốt nhất; T1 tốt nhất từ trước |
| `exmix2` + CLIP K=6, điểm có mặt theo từ vựng base+LVIS+O365 (ngưỡng 0.0267) | 20.70 (chặn nhầm 4.4%) | **0.41 ✅** · 27–29% | 41.2 ms ✅ | hỏi sai lớp ngang SAM 3 (0.4) |
| **`exmix2` + CLIP một lượt K=6, CLIP compile (ngưỡng 0.1166)** | 21.09 (chặn nhầm 5.0%; không cổng 19.03) | **1.15** · 28% | **38.8 ms ✅** | cấu hình đạt T3 (Linux/T4) |
| CountYOLO `vitpa` | 24.99 | 19.1 · 41% | ~44 ms | nghe text hơn, đếm kém đi |
| SAM 3 (zero-shot) | 26.61 | 0.4 · 21% ✅ | ~680 ms (RTX 4050) | hỏng ở ảnh > 200 vật |
| CountGD-Box | 15.01 | — | 422 ms | mốc |

## Đã thử, không lặp lại
Mở neck YOLO-World · DINOv2-base · ViT 840px · mosaic 4×4 không chuẩn hoá loss · Canonical-Scale Zoom ·
tracker video chỉ theo vị trí (đếm thừa) · luôn chia ô cho video · chia ô 3×3 trên FSC-147 ·
`vitpa` (một head học cả đếm lẫn lớp) · `vitgate` chỉ tín hiệu YOLO-World + CLIP · chuẩn hoá loss theo mẫu + mosaic 4×4 ·
DINOv2 448 (không giảm độ trễ batch 1) · EMA 0.999 · TTA lật ngang (−0.1 MAE, ×2 thời gian) ·
cổng text nhân từng ô (vitgate, kể cả chưng cất SAM 3) · cổng mức có/không (chặn nhầm 16% ảnh đúng lớp) · exemplar âm DINOv2 (CLIP) · loại tên đồng nghĩa τ 0.9/0.95 · số exemplar CLIP thích ứng (alpha).

## Các vòng

### Vòng 1 — `vitgate` (đang chạy: `nadthien/countyolo-density-vit-gate`)
- Vấn đề: T2 (bỏ qua text) mà `vitpa` sửa được một phần nhưng làm hỏng T1.
- Thay đổi: density của `vit` đóng băng × cổng text (logit YOLO-World + DINOv2 điều chế bởi CLIP text, FiLM),
  chỉ train cổng 56K tham số; prompt âm 0.3, mosaic trộn lớp 0.5; chọn checkpoint theo val MAE + đếm khi hỏi sai lớp.
- Kỳ vọng: T1 ≈ 18.9, T2 cải thiện rõ.

### Vòng 1b — chuẩn hoá loss theo mẫu (đang chạy: `ttdan2006/countyolo-density-vit-lossnorm`)
- Nhắm T1: sai số dồn vào ~9 ảnh > 500 vật; mosaic 4×4 lần trước thất bại vì mẫu cực đông chi phối loss (loss ×5).
- Thay đổi: `--loss-norm-ref 100` (trọng số mẫu 1/sqrt(1 + count/100), chuẩn hoá trung bình 1) cho MSE + region loss,
  cùng mosaic lưới 2 và 4; arch `vit`, 50 epoch. Khi tắt (ref 0) loss trùng khớp bản cũ (đã kiểm tra).
- Kỳ vọng: nhóm ≥ 500 vật bớt đếm thiếu mà không kéo lệch nhóm 50–200 như lần trước.

### Vòng 2 — kết quả `vitgate` + 2 hướng mới (2026-10-15 tối)
- `vitgate` (`density_head_best_vitgate_ep11.pth`): T1 26.28 ❌ (cổng chặn nhầm cả lớp đúng), T2 hỏi sai lớp 13.5
  (tốt nhất trong bản YOLO, vitpa 19.1), mosaic 44% (vitpa 41%). Không thay `vit`.
  Nguyên nhân: tín hiệu text (logit YOLO-World + CLIP) quá yếu để cổng tách đúng/sai lớp.
- T2 → **chưng cất từ SAM 3**: `tools/sam3_pseudo.py` sinh mask "khái niệm được hỏi" cho train/val, gồm negative khó
  (20 lớp COCO hỏi trên mọi ảnh, giữ lớp có vật; 34% prompt lớp khác có vật thật, so với 2% khi lấy ngẫu nhiên).
  Cổng học BCE theo mask SAM 3 (`--pseudo --hard-prob --gate-bce`). Đang sinh nhãn trên máy (RTX 4050).
- T3 → DINOv2 đầu vào 448 (`thiend27/countyolo-density-vit-v448`), đo T1 trước, tốc độ đo sau bằng benchmark.
- Sửa lỗi launcher: `--vit-name/--vit-input` bị lồng nhầm trong nhánh `--loss-norm-ref` (do lần vá trước) nên lần push
  đầu của v448 thiếu `--vit-input 448`; đã sửa và push lại (TRAIN_EXTRA có `--vit-input 448`). Các kernel trước không ảnh hưởng.
- Đổi kế hoạch T2: dừng sinh nhãn SAM 3 trên máy (6 s/ảnh, máy quá nóng). Người dùng yêu cầu hạn chế chạy local →
  upload `sam3.pt` thành dataset riêng tư `nadthien/sam3-weights`; kernel `nadthien/countyolo-density-vit-distill`
  (`--sam3-distill`) sẽ sinh nhãn + train cổng + đánh giá trọn trên Kaggle. Từ nay mọi việc nặng chạy trên Kaggle.

### Vòng 3 — kết quả chuẩn hoá loss + hướng mới cho T1
- `vit-lossnorm` (ref 100, mosaic 2+4): test MAE **23.61** ❌ (vit 18.90). Nhóm ≥500 vật 633 → 574 (khá hơn chút) nhưng
  <50 vật 7.33 → 11.18 và 50–200 vật 21.1 → 28.3 (đếm thừa +8–9). Không dùng.
- T1 → **EMA trọng số** (`ttdan2006/countyolo-density-vit-ema`, `--ema 0.999`): val MAE mọi lần train dao động 21–30
  giữa các epoch → checkpoint chọn theo may rủi; EMA làm mượt, không tốn thời gian suy luận. Đã kiểm tra EMA trên CPU.

### Vòng 4 — kết quả DINOv2 448 (T3)
- `vit-v448`: val 21.76 (560: 21.68 ± 0.21), test **19.76** (560: 18.90–20.01), có chia ô 18.44 (560: 17.98)
  → độ chính xác trong dao động seed. Đang đo tốc độ trên T4: `thiend27/countyolo-bench-v448` (chỉ CountYOLO,
  thêm biến thể vit-small-448 vào benchmark; CountGD đã đo 422 ms AMP).
- Đo T3 (`thiend27/countyolo-bench-v448`, T4, batch 1): vit-560 AMP 45.7 ms, vit-448 AMP 47.1 ms (fp32 44.3) →
  448 **không giảm độ trễ batch 1**, chỉ tăng thông lượng bs8 (56 → 77/s). Độ trễ bị chi phối bởi số kernel nhỏ /
  chép dữ liệu, không phải FLOPs (YOLO-World một mình đã 15–19 ms).
- T3 → tối ưu suy luận không train lại (`model.optimize_inference`): cache text trên GPU (bỏ chép CPU→GPU mỗi
  forward), DINOv2 attention SDPA (khớp eager, lệch ≤ 4e-5), DINOv2 fp16 thật, tuỳ chọn torch.compile.
  Đo + kiểm tra số đếm không đổi: `thiend27/countyolo-bench-opt`.

### Vòng 6 — T3 đạt về tốc độ, đang kiểm tra độ chính xác
- `thiend27/countyolo-bench-opt` (T4, batch 1): vit-560 AMP 42.9 ms · +DINOv2 fp16 36.1 ms · **+fp16 + torch.compile
  23.1 ms (18× nhanh hơn CountGD-Box 422 ms)** → T3 ✅ về tốc độ. (448 + fp16 36.4 ms: không cần 448.)
  Lưu ý: kiểm tra số đếm trong benchmark vô nghĩa (head khởi tạo ngẫu nhiên → đếm 0) → kiểm tra thật bằng checkpoint:
  `thiend27/countyolo-analyze-fast` (`--fast-eval`: test MAE thường vs `--fast` vs `--fast --ttn-thr 400`).
- torch.compile cần Triton → chạy được trên Kaggle/Linux; trên Windows (UI) dùng fp16 không compile.
- **Kiểm tra độ chính xác chế độ nhanh** (`thiend27/countyolo-analyze-fast`, seed 1): test MAE 19.25 → **19.27** với
  `--fast` (lệch TB mỗi ảnh 0.14%) → **T3 ✅** (23.1 ms, 18× CountGD-Box). Chia ô ≥400 với seed này 19.33 (không giúp;
  seed 3 giúp 18.90 → 17.98) → hiệu quả chia ô phụ thuộc seed.
- T1 → TTA lật ngang dùng phần thời gian dư (23 → ~40–46 ms): `thiend27/countyolo-analyze-tta`.

### Vòng 7 — T1 chững lại
- EMA 0.999 (`ttdan2006/countyolo-density-vit-ema`): val tốt nhất 23.83 (kém vit ~21.7), test 19.19, chia ô 18.25
  → trong dao động seed (18.90–20.01). Không dùng.
- TTA lật ngang (`thiend27/countyolo-analyze-tta`, chế độ nhanh): 19.27 → 19.17. Không đáng ×2 thời gian.
- Nhận định: các mẹo rẻ đã cạn; khoảng cách T1 (≈19 vs 15–16) là do dữ liệu (FSC-147 3659 ảnh train; YOLO-Count /
  RT-Counter dùng LVIS). Hướng tiếp: nhãn giả SAM 3 trên ảnh không nhãn (COCO trên Kaggle) — chờ tốc độ SAM 3 trên T4
  từ kernel distill để ước lượng quy mô.

### Vòng 8 — chưng cất SAM 3 (T2) + cổng mức "có / không"
- `nadthien/countyolo-density-vit-distill`: SAM 3 trên T4 3.93 s/ảnh (23 prompt) → 14190 cặp nhãn, 32% prompt lớp
  khác có vật thật. Cổng học theo mask SAM 3: T2 hỏi sai lớp **10.1** (trung vị 0.07), mosaic **39%** — tốt nhất họ YOLO,
  nhưng T1 28.15 (cổng nhân từng ô chặn nhầm vật đúng lớp). Checkpoint: `density_head_best_vitgate_sam3distill.pth`.
- Ý tưởng: dùng cổng ở mức ảnh — presence = TB cổng theo trọng số density; ≥ ngưỡng thì giữ NGUYÊN density của vit
  (không nhân cổng), dưới thì 0 (`model.presence_thr`). Ngưỡng chọn trên val (MAE đúng lớp + đếm sai lớp).
  Kỳ vọng: T1 ≈ vit (18.9) mà hỏi lớp vắng mặt ≈ 0; mosaic 2 lớp (cả hai có mặt) không cải thiện.
  Đang chạy: `nadthien/countyolo-analyze-presence` (`tools/presence_eval.py`).
- Tốc độ SAM 3 trên T4 (~0.17 s / prompt) → nhãn giả cho T1 (COCO) khả thi: ~1 s/ảnh với vài prompt → ~40k ảnh / kernel.

### Vòng 9 — cổng mức "có / không" + hướng SAM 3 theo video
- `nadthien/countyolo-analyze-presence` (ngưỡng 0.453 chọn trên val): hỏi sai lớp **7.06** (trung vị 0), nhưng
  **15.9% ảnh đúng lớp bị chặn nhầm** → MAE đúng lớp 27.19 (không cổng 18.90); mosaic 100%. Presence TB 0.65 (đúng) vs
  0.175 (sai) nhưng chồng lấn → giới hạn là khả năng nhận khái niệm từ text của mô hình nhẹ, không phải cách gắn cổng.
- Kết luận T2: trong họ YOLO nhẹ, T2 luôn đổi bằng T1. Hướng mới cho bài toán video (mục tiêu chính): **SAM 3 một lần
  / video** (vài frame chính) để xác nhận có mặt + lấy exemplar đúng lớp → CountYOLO mọi frame dùng `exemplar_points`.
  Chi phí SAM 3 chia đều cho cả video, T3 theo frame giữ nguyên.
- Đo giới hạn trên: mosaic 2 lớp với exemplar = điểm GT đúng lớp (`--oracle-exemplars`):
  `thiend27/countyolo-analyze-oracle`. Nếu sai số mosaic giảm mạnh → xây pipeline SAM 3 keyframe + CountYOLO.
- Kết quả giới hạn trên (`thiend27/countyolo-analyze-oracle`, vit seed 1): mosaic 2 lớp với exemplar đúng lớp
  **75%** (tự chọn 107%), MAE 89.5 (128.6) → exemplar giúp nhưng Refine Head chưa từng học cảnh nhiều lớp.
- T2 → **fine-tune "đếm theo exemplar"** (`ttdan2006/countyolo-density-vit-exmix`): khởi tạo vit tốt nhất, mosaic 0.5
  (trộn lớp 0.7), luôn dùng exemplar = điểm GT của lớp được hỏi (`--gt-exemplar-prob 1`; lúc chạy thật SAM 3 cấp),
  25 epoch lr 1e-4; val chọn checkpoint cũng dùng exemplar GT; sau train: prompt_eval `--oracle-exemplars`.

### Vòng 10 — pipeline video "SAM 3 frame chính + CountYOLO mọi frame" (T2/T4)
- `exmix` (`ttdan2006`) vẫn đang chạy.
- Mô hình: thêm **cache prototype exemplar** — `forward(..., exemplar_protos=aux["protos"])` dùng prototype (P3/P4/ViT)
  tính ở ảnh khác thay cho feature tại vị trí exemplar của ảnh hiện tại; `density_tiled_count(..., protos=)`.
  Kiểm tra CPU: prototype dùng lại cho đúng kết quả như truyền exemplar trực tiếp (lệch 0.0).
- `countvid_ext/countyolo_stage1.py`: tuỳ chọn `sam3_weights`, `kf_every` — mỗi `kf_every` frame chạy SAM 3 một lượt:
  không thấy vật → không box tới frame chính sau (chống nhầm lớp); có → tâm 5 box điểm cao nhất làm exemplar →
  prototype dùng cho mọi frame tới frame chính sau. Cờ CountVid: `--countyolo_sam3`, `--countyolo_kf_every`;
  launcher: cấu hình `countyolo_sam3kf`, `--kf-every`, `--ckpt-url` (dataset `nadthien/sam3-weights` tự gắn).
- Đang chạy: `nadthien/countvid-tiling-sam3kf` — Penguins, CountYOLO vit + SAM 3 mỗi 30 frame + SAM 2.1
  (so với CountYOLO thường 6.67, CountVid 3.33). Khi `exmix` xong sẽ chạy lại với checkpoint đó.

### Vòng 11 — kết quả fine-tune "đếm theo exemplar" (`exmix`)
- Val (exemplar GT) không lần nào tốt hơn bản khởi tạo (20.64 ở epoch 0; các epoch 21.3–29.6) → `density_head_best`
  = bản khởi tạo vit seed 3 → số test trùng vit: 18.90, **exemplar GT 17.95**, chia ô 17.98.
- Giới hạn trên exemplar đúng lớp trên mosaic với vit seed 3: **98%** (seed 1: 75%) → hiệu ứng exemplar phụ thuộc seed,
  Refine Head chủ yếu dựa density thô (bỏ qua text + exemplar).
- Chưa biết bản cuối (epoch 25, đã học trên mosaic trộn lớp) có "chỉ đếm vật giống exemplar" không → đo:
  `ttdan2006/countyolo-analyze-exmixlast` (launcher `--prompt-only` giờ đo cả best + last, last thêm test exemplar GT).
- `nadthien/countvid-tiling-sam3kf` (Penguins) vẫn đang chạy.

### Vòng 12 — "đếm theo exemplar" đạt mosaic 2 lớp; cần chọn checkpoint đúng
- `ttdan2006/countyolo-analyze-exmixlast`, bản cuối (epoch 25) của `exmix`: **mosaic 2 lớp với exemplar đúng lớp 22%**
  (MAE 48.9; vit 97% / 115.9) → **T2 mosaic ✅ khi exemplar đúng lớp** (lúc chạy thật do SAM 3 cấp). Hỏi sai lớp (exemplar
  tự chọn) 53.0 (vit 65.1) — phần này dành cho cờ có mặt của SAM 3 (0.4).
  Đổi lại ảnh một lớp: exemplar GT 24.37 (đếm thiếu, bias −14.7 ở vật ≥40px), tự chọn 19.99 (vit 18.90).
- Nguyên nhân chọn nhầm: val chỉ đo ảnh một lớp → best = bản khởi tạo. Thêm `--val-mosaics N` cho train
  (`evaluate_mosaics`: N mosaic val 2 lớp cố định, exemplar GT đúng lớp; tiêu chí = val MAE + MAE mosaic) + log mỗi epoch.
  Kiểm tra CPU (4 ảnh, 3 mosaic) chạy đúng.
- Đang chạy: `ttdan2006/countyolo-density-vit-exmix2` — như exmix + `--val-mosaics 100`, 25 epoch; sau train: test,
  exemplar GT, chia ô, prompt_eval `--oracle-exemplars`.
- Đã chép `checkpoints/density_head_vit_exmix_last_ep25.pth` (dùng cùng exemplar SAM 3, không dùng một mình).

### Vòng 13 — T4: SAM 3 frame chính + CountYOLO + SAM 2 vượt CountVid trên Penguins
- `nadthien/countvid-tiling-sam3kf` (vit seed 3, SAM 3 mỗi 30 frame, SAM 2.1 theo dõi): **Penguins MAE 2.67**
  (27/28, 55/59, 35/38; bias −2.67) — CountVid 3.33, CountYOLO không SAM 3 6.67. Stage 1 = 90 + 212 + 179 s
  (gồm SAM 3); tổng 3.1 giờ do SAM 2 (stage 2 + 3).
- Đang chạy: `nadthien/countvid-tiling-sam3kf-mot20` (MOT20-01; CountVid 97, CountYOLO 102, GT 80).
  `ttdan2006/countyolo-density-vit-exmix2` vẫn đang train.

### Vòng 14 — kết quả cuối trước khi dừng vòng lặp (người dùng yêu cầu ngưng)
- `ttdan2006/countyolo-density-vit-exmix2` (chọn best theo val MAE + MAE mosaic val): best epoch 20 (val 22.93, mosaic val
  39.5 / 20%). Test: tự chọn exemplar **19.12**, exemplar GT 22.76, chia ô 18.36; prompt_eval `--oracle-exemplars`:
  **mosaic 2 lớp 21% ✅** (MAE 49.7; vit 100% / 116.1), hỏi sai lớp 55.1 (vit 65.1). Cân bằng hơn `exmix` bản cuối
  (19.99 / 24.37 / 22%). Đã chép `checkpoints/density_head_best_vit_exmix2_ep20.pth`.
- `nadthien/countvid-tiling-sam3kf-mot20` (vit seed 3 + SAM 3 mỗi 30 frame + SAM 2.1): **MOT20-01 76 / GT 80** (sai 4;
  CountVid 97 → sai 17, CountYOLO không SAM 3 102 → sai 22). Stage 1 212 s, tổng 1827 s.
- T4 hiện trạng (CountYOLO + SAM 3 frame chính + SAM 2): Penguins 2.67 (CountVid 3.33) ✅, MOT20-01 sai 4 (CountVid 17) ✅,
  Crystals 49.9 không SAM 3 (CountVid 69.1) ✅ — chưa chạy Crystals với SAM 3 frame chính, chưa chạy đủ 3 video MOT20.
- Việc còn mở: T1 (19 vs 15–16); hỏi sai lớp trên ảnh đơn (cần SAM 3 xác nhận có mặt); exemplar GT làm ảnh 1 lớp kém đi
  (22.76 vs 19.12) → trên ảnh 1 lớp nên dùng exemplar tự chọn, chỉ dùng exemplar SAM 3 khi cảnh nhiều lớp.

### Vòng 15 — tập trung T2 (nhầm lớp): kiểm tra bằng CLIP, không cần SAM 3
- Ý tưởng: YOLO-World đã nạp sẵn **CLIP ViT-B/32 đầy đủ** (cả encoder ảnh). Sau khi CountYOLO đếm: K=12 đỉnh density →
  vùng cắt (cạnh theo Scale Head × 1.5) → CLIP p(prompt) so với từ vựng chung (FSC-147 train + COCO, 168 tên, bỏ prompt).
  Có mặt nếu điểm ≥ ngưỡng (chọn trên val như presence_eval) → hỏi sai lớp đếm 0; exemplar = tâm 5 vùng giống prompt nhất
  → `exmix2` đếm lại (mosaic 2 lớp không cần SAM 3). Điểm: crop_max, crop_top3, global (cả ảnh), mix.
- `utils/clip_verify.py` (ClipVerifier), `tools/clip_verify_eval.py`, launcher `--analyze ... --clip-verify`.
  Chạy thử CPU 6 ảnh: chạy đúng; AUC đúng/sai lớp 0.92–0.97 (val), mẫu quá nhỏ để kết luận.
- Đang chạy: `ttdan2006/countyolo-analyze-clipverify` (checkpoint `exmix2`, đủ val/test + 300 mosaic).
- Kết quả bản 1 (`ttdan2006/countyolo-analyze-clipverify` v2, `exmix2`, ứng viên = đỉnh density): AUC đúng/sai lớp
  0.98–0.99 (cổng presence cũ chồng lấn nặng). TEST hỏi sai lớp: không cổng 54.7 → crop_top3 6.00 / mix 3.53 /
  **global 1.35**; đổi lại chặn nhầm ảnh đúng lớp 6.2% / 6.7% / 10.6% → MAE 22.37 / 21.80 / 24.92 (không cổng 19.12).
  Mosaic 2 lớp: tự chọn 37%, exemplar CLIP 34%, exemplar GT 20%, có cổng 66% (cổng chặn lớp ít vật trong mosaic).
  80 ms / câu hỏi (CountYOLO fp32 + CLIP fp32, chưa tối ưu). Bản 1 lần đầu chạy CLIP trên CPU (CLIP không theo
  model.to(device)) → đã sửa.
- Nguyên nhân dự đoán: (a) từ vựng COCO số ít trùng lớp FSC số nhiều (test: apples, elephants; val: birds, books, chairs,
  horses) → chặn nhầm; (b) vùng ứng viên chỉ từ đỉnh density (không nghe text) → trong mosaic thiếu vùng của lớp được hỏi.
- Bản 2: bỏ tên trùng số ít/nhiều, trung bình 3 câu mẫu, thêm K box YOLO-World cho đúng prompt làm ứng viên, CLIP fp16,
  `--fast`, CSV từng câu hỏi (`--dump`). Đang chạy: `ttdan2006/countyolo-analyze-clipverify2`.
- `ttdan2006/countyolo-analyze-clipverify2` nằm QUEUED > 1 giờ (Kaggle thiếu GPU / hết quota tuần?). Chạy song song bản
  giống hệt trên thiend27: checkpoint `exmix2` đưa lên dataset riêng tư `thiend27/countyolo-ckpts`
  (`exmix2/checkpoints/density_head_best.pth`), launcher `--ckpt-dataset` thay cho output kernel nguồn
  (nhúng checkpoint vào script → Kaggle trả 400, script 3.4 MB). Kernel: `thiend27/countyolo-analyze-clipverify2`.
- 2026-10-19: cả hai bản 2 hỏng. ttdan2006: `expected Float but found Half` — `nn.Module.half()` đổi cả LayerNorm của CLIP
  (vốn ép đầu vào fp32) → dùng `clip.model.convert_weights` (giữ LayerNorm fp32), kiểm tra CPU fp16 ≈ fp32.
  thiend27: không thấy checkpoint trong dataset `thiend27/countyolo-ckpts` → lookup rộng hơn + in danh sách .pth khi thiếu.
  Push lại trên ttdan2006 (`countyolo-analyze-clipverify2` v2).

### Vòng 16 — kết quả CLIP kiểm tra bản 2 (`ttdan2006/countyolo-analyze-clipverify2` v2)
- AUC đúng/sai lớp test 0.992–0.995 (bản 1: 0.978–0.989). Ngưỡng theo tiêu chí cũ (val MAE + đếm sai lớp): mix → MAE 20.62,
  chặn nhầm 3.1%, sai lớp 3.04 (lọt 3.6%); global → sai lớp 1.98.
- Mosaic 2 lớp: tự chọn 37% → **exemplar CLIP 26%** (exemplar GT 21%; bản 1: 34%) nhờ thêm box YOLO-World theo prompt.
- Thời gian CountYOLO `--fast` + CLIP fp16 (≤ 24 vùng): 47.7 ms (chưa gồm lượt đếm lại có exemplar).
- Điểm vận hành (ngưỡng chọn trên val theo "đếm TB sai lớp ≤ X", từ CSV): mix, X=0.5 → TEST MAE 20.86 (exemplar CLIP 22.30),
  chặn nhầm 4.6%, **sai lớp 1.58** (lọt 2.0%); global, X=1 → MAE 22.08, sai lớp 1.41. Val → test lệch (test lọt nhiều hơn).
- So với các cách trước: cổng presence vitgate (MAE 27.19, sai lớp 7.06), vitpa (24.99 / 19.1 / 41%) → CLIP kiểm tra tốt hơn hẳn.
- Việc tiếp: gộp thành một lượt (chọn exemplar từ box YOLO-World trước Refine Head, bỏ lượt đếm lại) để về ≤ 42 ms; đưa vào UI.

### Vòng 17 — gộp CLIP kiểm tra vào một lượt + đưa vào UI
- `DensityYOLOWorld.forward(..., exemplar_fn=)`: hook gọi sau bước 1 (density thô) → chọn exemplar trước Refine Head.
- `utils/clip_verify.VerifiedCounter`: ứng viên = 12 đỉnh density thô (cạnh ≈ 640/sqrt(đếm thô), kẹp 32–160) + 12 box
  YOLO-World cho prompt; CLIP một batch (vùng + cả ảnh); điểm = 0.5 TB top-3 vùng + 0.5 cả ảnh; dưới ngưỡng → 0.
  Exemplar: self / clip / auto (dùng CLIP khi p TB của 5 đỉnh tự chọn < 0.5 × p TB 5 vùng CLIP chọn — cố định trước,
  không chỉnh theo test). Không còn lượt đếm lại.
- `tools/clip_verify_eval.py` viết lại theo đúng đường chạy này (ngưỡng = nhỏ nhất sao cho val đếm TB sai lớp ≤ 0.5;
  báo self/clip/auto; thời gian lượt thật). Đang chạy: `ttdan2006/countyolo-analyze-clipverify3`.
- UI (`app/app.py`): ô "Kiểm tra lớp bằng CLIP" (mặc định bật), báo điểm/ngưỡng, đánh dấu exemplar CLIP; checkpoint
  mặc định `exmix2`; chia ô dùng lại prototype exemplar CLIP. `CLIP_THR` sẽ đặt theo kết quả val của kernel trên.
  Thử CPU: "seagulls" → đếm 25.8 (GT 13, như không cổng), "cars" → điểm 0.003 → 0.
- Kết quả `ttdan2006/countyolo-analyze-clipverify3` (một lượt, `--fast`): ngưỡng score 0.1279 (val: đếm TB sai lớp 0.47).
  TEST auto: MAE 20.61, chặn nhầm 4.5%, **hỏi sai lớp 1.37** (lọt 1.9%); không cổng auto 18.98 < self 19.12 (auto dùng
  exemplar CLIP ở 18.8% ảnh 1 lớp, 60% câu hỏi mosaic). Mosaic: auto **26%**, clip 25%, GT 22%, self 37%; có cổng 29%.
  Thời gian **49.9 ms** (p90 54.5) — vượt 42 ms; CountYOLO một mình 23.1 ms → CLIP + ứng viên ~27 ms.
- UI: `CLIP_THR = 0.1279`. Đo từng phần trên T4: `ttdan2006/countyolo-analyze-benchverify` (`tools/bench_verify.py`).
- `ttdan2006/countyolo-analyze-benchverify` (T4, ảnh test thật): CountYOLO 26.2 ms · VerifiedCounter 51.1 ms · ứng viên
  1.8 ms (24 vùng) · CLIP 13 vùng 19.2 / 25 vùng 19.5 ms (bị chi phối bởi khởi chạy kernel) · CLIP compile
  (reduce-overhead) 13 vùng **8.6 ms** / 25 vùng 16.0 ms.
- → `ClipVerifier(compile_batch=2K+1)`: CLIP compile batch cố định (đệm 0; kiểm tra CPU lệch 1e-7), K=6 (6 đỉnh + 6 box
  + cả ảnh = 13 vùng); ước tính ~40 ms. Đo lại độ chính xác + thời gian: `ttdan2006/countyolo-analyze-clipverify-k6`.
- `thiend27/countyolo-analyze-clipverify-k6` (K=6, CLIP compile batch 13; checkpoint từ dataset `thiend27/countyolo-ckpts`
  — Kaggle bỏ thư mục ngoài khi giải nén → nguồn tên `countyolo-ckpts`): **38.8 ms** (p90 40.6) ✅ T3; ngưỡng val 0.1166.
  TEST auto: MAE 21.09 (chặn nhầm 5.0%), **hỏi sai lớp 1.15** (lọt 1.8%); mosaic auto 28% (clip 27%, GT 21%).
  So với K=12: nhanh hơn 11 ms, sai lớp tốt hơn chút, mosaic kém 2 điểm. ttdan2006 hết quota GPU tuần (kernel QUEUED ~4 h).
- Hiện trạng T2 (họ CountYOLO, không SAM 3): hỏi sai lớp 1.15–1.37 (mục tiêu ≤ 1), mosaic 26–28% (≤ 25%) — sát mục tiêu,
  từ 65 / 107% của vit ban đầu.

## Vòng lặp mới (2026-10-20): cải thiện CountYOLO + CLIP về chống nhầm lớp (mục tiêu sai lớp ≤ 1, mosaic ≤ 25%, MAE ~19–20, ≤ 42 ms)

### Vòng C1 — phân tích lỗi + từ vựng lớn / loại từ đồng nghĩa
- Phân tích CSV (`clipverify3`, K=12, ngưỡng 0.1279): chặn nhầm dồn vào vài lớp — test: finger foods 18/50, marbles 16/73,
  red beans 8/10; val: fresh cut 11/11, flamingos 10/102. Lọt khi hỏi sai lớp: nhiều câu thực chất đúng nghĩa
  (flamingos → birds, birds → seagulls, oysters → oyster shells); lọt thật khi lớp thật không có trong từ vựng
  (legos → green peas 261, green peas → apples 130).
- Độ giống text CLIP: đồng nghĩa/lớp cha 0.75–0.90 (marbles~balls 0.84, red beans~beans 0.90) nhưng khác lớp 0.63–0.93
  (cars~trucks 0.93) → không có ngưỡng tách gọn, để val quyết định.
- Thử 9 biến thể (chấm lại từ cùng đặc trưng CLIP, `VARIANTS` trong tools/clip_verify_eval.py): từ vựng base (168) /
  LVIS (1198) / base+LVIS+Objects365 (1447) × loại tên có embedding giống prompt ≥ τ ∈ {không, 0.9, 0.95}; ngưỡng mỗi
  biến thể chọn trên val (đếm sai lớp ≤ 0.5), chọn biến thể theo val MAE. Lưu ý: LVIS là từ vựng chung, có thể chứa
  tên lớp test (không dùng nhãn test để xây).
- Đang chạy: `thiend27/countyolo-analyze-clipvocab` (K=6, CLIP compile).
- Kết quả `thiend27/countyolo-analyze-clipvocab` (K=6, 41.2 ms): loại tên đồng nghĩa τ 0.9/0.95 không đổi gì (hầu như
  không tên nào giống prompt đến mức đó). Từ vựng lớn (base+LVIS+O365, 1447 tên) trội hơn base ở mọi điểm vận hành
  (ngưỡng chọn trên val theo "đếm TB sai lớp ≤ X"):
  X=0.75 → base: MAE 21.09 / chặn nhầm 5.0% / sai lớp 1.15 · **lớn: 20.70 / 4.4% / 0.41**;
  X=1.0 → lớn: 19.88 / 2.4% / 0.99. Mosaic có cổng 29% (không cổng, exemplar CLIP 27%).
  Chọn X=0.75 (ngưỡng 0.0267) cho UI — lưu ý X chọn sau khi xem test của vài điểm (hơi lạc quan).
- `VerifiedCounter(presence_vocab="all")`: điểm có mặt theo từ vựng lớn, exemplar vẫn chọn theo base (như lúc đánh giá).
  `ClipVerifier(cache_path=)`: mã hoá text theo khúc 128 + lưu đĩa (`outputs/clip_text_cache.pt`; CPU mã hoá 4341 câu > 15 phút).
- Hiện trạng chống nhầm lớp: **sai lớp 0.41 ✅** (SAM 3 0.4), MAE 20.70, 41.2 ms ✅; còn **mosaic 27–29%** (≤ 25%).

### Vòng C2 — exemplar nhiễu khi train (mosaic)
- Exemplar GT → mosaic 21–22%, exemplar CLIP chọn → 27%: mô hình chưa chịu được exemplar chọn nhầm. `--ex-noise q`:
  mỗi điểm exemplar GT bị thay bằng điểm ngẫu nhiên trong ảnh với xác suất q.
- Đang chạy: `thiend27/countyolo-density-vit-exnoise` — fine-tune từ exmix2 (dataset), 15 epoch, lr 5e-5, mosaic 0.5,
  trộn lớp 0.7, exemplar GT + nhiễu 0.25, chọn best theo val MAE + mosaic val; sau train: test, prompt_eval oracle, CLIP K=6.

### Vòng C3 — exemplar âm (không train, chạy song song với C2)
- Trong mosaic mô hình đếm cả vật lớp kia. `utils/clip_verify.negative_mask`: vùng ứng viên CLIP chắc chắn không phải prompt
  (p < neg_p) = exemplar âm; mỗi ô DINOv2 so cosine với exemplar dương (5 vùng CLIP chọn) và âm, giống âm hơn dương quá
  margin → xoá density ô đó. Chỉ áp khi auto chọn exemplar CLIP (cảnh nhiều lớp). `VerifiedCounter(neg_margin=)`.
- Eval đo 5 biến thể (neg_p, margin) ∈ {(0.05,0), (0.05,0.05), (0.05,0.1), (0.02,0), (0.1,0)} từ cùng density + đặc trưng
  (không chạy lại mô hình). Thử CPU 1 mosaic: red beans GT 52: CLIP 55 → âm 26 (m=0) / 40 (m=0.05) — cần số đủ.
- Checkpoint exmix2 cũng đưa lên dataset riêng tư `nadthien/countyolo-ckpts` (file nằm ở gốc dataset).
  Đang chạy: `nadthien/countyolo-analyze-clipneg` (K=6, --fast).
- Kết quả C3 (`nadthien/countyolo-analyze-clipneg`): exemplar âm **không giúp** — mosaic 27–34% (auto 28%, clip 27%,
  GT 22%), MAE ảnh 1 lớp tệ đi 0.9–3.2 (sai lớp giảm nhẹ 1.15 → 0.8–1.05 chỉ vì xoá bớt density). Không dùng.
  Nhận xét: auto chỉ dùng exemplar CLIP ở 54.7% câu hỏi mosaic; phần còn lại dùng exemplar tự chọn (37%).
  Khoảng cách chính: chất lượng exemplar CLIP (27%) vs GT (22%).

### Vòng C4 — số exemplar CLIP thích ứng (không train)
- Luôn lấy đủ 5 vùng CLIP điểm cao nhất → trong mosaic vài vùng cuối là lớp kia. `select_exemplars(alpha)`: chỉ giữ vùng có
  p ≥ alpha × p cao nhất (1–5 vùng). Eval: alpha ∈ {0.3, 0.5, 0.7} (đếm lại cùng điểm CLIP), **alpha chọn trên 150 mosaic
  VAL** (rng riêng, không đổi chuỗi mosaic TEST); báo ảnh 1 lớp + mosaic TEST. Exemplar âm (C3) tắt trong eval (`NEG = []`).
  Thêm `--vocabs` (chạy thử CPU chỉ base). Đang chạy: `nadthien/countyolo-analyze-clipalpha` (exmix2, K=6).
- Kết quả C4 (`nadthien/countyolo-analyze-clipalpha`, 40.2 ms): mosaic VAL auto alpha 0 / 0.3 / 0.5 / 0.7 = 26.5 / 27.6 /
  28.8 / 30.6% → **chọn alpha 0 (đủ 5 exemplar)**; TEST ảnh 1 lớp cũng tệ dần (19.03 → 19.37 / 19.74 / 20.36). Không dùng.
- Phân tích CSV (không GPU): cho qua thêm khi cảnh nhiều lớp và crop_top3 ≥ tc (tc chọn trên val, giữ val sai lớp ≤ 0.75)
  → tc = 0.536, gần như không đổi gì (mosaic có cổng 29.5%, chặn 4.5 → 4.3%). Không dùng.
- Hiện trạng: mosaic auto không cổng 28%, có cổng 29.5%; exemplar GT 22%. Chờ C2 (exnoise).

### Vòng C5 — nguồn exemplar CLIP
- Exemplar CLIP đang chọn từ cả đỉnh density (không nghe text, trong mosaic hay rơi vào lớp kia) lẫn box YOLO-World cho
  prompt. Thử: chỉ box YOLO-World / chỉ đỉnh density; chọn trên 150 mosaic VAL. Alpha (C4) tắt (`ALPHAS = ()`).
  Đang chạy: `nadthien/countyolo-analyze-clipsrc` (exmix2, K=6).

### Kết quả C2 + C5
- C5 (`nadthien/countyolo-analyze-clipsrc`): mosaic VAL auto: cả hai nguồn 26.5% / chỉ YOLO-World 29.2% / chỉ đỉnh density
  42.6% → giữ cả hai. Không dùng.
- **C2 (`thiend27/countyolo-density-vit-exnoise`, exemplar nhiễu 0.25)** — tốt nhất từ trước tới nay cho đếm:
  FSC-147 test **17.89** (vit 18.90; exmix2 19.12), exemplar GT 18.67, **chia ô 16.53** (vit 17.98); prompt_eval oracle
  mosaic 24% (MAE 50.7). Exemplar nhiễu như một regularizer.
  CLIP K=6 (39.5 ms): không cổng auto 17.16; ngưỡng từ vựng lớn theo val sai lớp ≤ 0.75 → 0.0198: **TEST MAE 18.38,
  chặn nhầm 3.0%, sai lớp 0.77 ✅**, mosaic có cổng 28.9% (không cổng 28.1%, GT 23%).
- UI: checkpoint mặc định `checkpoints/density_head_best_vit_exnoise.pth`, `CLIP_THR = 0.0198`; tab Mô hình cập nhật số.
- Hiện trạng chống nhầm lớp: sai lớp 0.77 ✅, MAE 18.38 ✅, 39.5 ms ✅; còn mosaic 28–29% (≤ 25%; exemplar GT chỉ 23%
  → cần cả exemplar tốt hơn lẫn đếm theo exemplar tốt hơn).

### Vòng C6 — exemplar nhiễu "chọn nhầm lớp" (mosaic)
- Lỗi thật lúc chạy: CLIP chọn nhầm VẬT LỚP KIA (không phải điểm ngẫu nhiên). Dataset: mosaic trộn lớp trả thêm
  `other_points` (vật các ô lớp khác); train `--ex-noise-other`: điểm exemplar bị thay (xác suất 0.25) lấy từ vật lớp kia
  khi có, ngược lại điểm ngẫu nhiên. Kiểm tra CPU: 5/10 điểm bị thay bằng vật lớp khác; train 1 epoch chạy đúng.
- Đang chạy: `nadthien/countyolo-density-vit-exother` — fine-tune từ exnoise (dataset riêng tư `nadthien/countyolo-exnoise`),
  15 epoch, lr 3e-5; sau train: test, prompt_eval oracle, CLIP K=6.
- C6 nằm QUEUED > 1 giờ trên nadthien → chạy song song bản giống hệt trên thiend27 (`thiend27/countyolo-density-vit-exother`, dataset riêng tư `thiend27/countyolo-exnoise`); lấy bản xong trước.

## 2026-11-07 — VideoCount: đối chứng video thật và nhánh CountYOLO ưu tiên độ chính xác

- Xác nhận baseline người dùng muốn: CountVid gốc = CountGD-Box + SAM 2; không gọi nhầm là DINOv2 + SAM 2.
- Thêm CountYOLO proposal fusion: box YOLO-World chỉnh hình học đỉnh density, box SAM 3 trên keyframe,
  NMS và lọc khối lượng đỉnh; dùng exnoise, keyframe mỗi 5 frame. Các biến thể cũ/exnoise/kf5 được giữ làm đối chứng.
  Không khẳng định cải thiện trước khi có số video.
- CountVid không nạp CountGD khi dùng Stage 1 CountYOLO (giảm bộ nhớ); baseline giữ nguyên model gốc.
- SAM 3: bỏ fallback ID theo thứ tự box; thiếu ID thật thì báo lỗi. Ghi trace ID/box từng frame,
  đếm ID thô và xuất hiện >=3 frame riêng; frame list được đóng gói FFV1 lossless theo luồng, dọn file tạm.
- Evaluator: lưu từng video/prompt ngay khi xong, không ghi đè các prompt cùng video, lưu đầy đủ log,
  có trạng thái incomplete/failed và manifest sampling. Lấy mẫu MOT20 đúng CountVid (ceil + round linspace).
- Launcher mới và hàng đợi `experiments/video_tasks.json`: mỗi job độc lập, checksum nguồn/checkpoint,
  xử lý quota, tự fetch, chốt dừng khi lỗi. Pilot Penguins-1 + MOT20-01; sau pilot chọn CountYOLO theo MAE rồi đánh giá
  13 video (11 video heldout báo riêng). Baseline mới dùng ttdan2006 để chia tải; trọng số riêng tư ở nadthien.
- Kết nối cả ba tài khoản đã kiểm tra thành công. Lỗi 401 ban đầu do access_token toàn cục cũ ghi đè legacy key;
  sửa trong client, không sửa/xóa credential toàn cục. Hai tài khoản kia không có quyền đọc dataset trọng số của nadthien.
- Lượt SAM 3 ngưỡng detection nội bộ 0.3 (Penguins-1, 34 frame, T4): GT 28, pred raw 77, pred hits>=3 71,
  256.0 giây gồm nạp model/encode; đây là kết quả chẩn đoán một video, không phải MAE toàn bộ Penguins.
  Đã xếp lượt `sam3_native` ngưỡng detection 0.5 (mặc định native) để đối chiếu công bằng với nhánh trước đây.
- Kiểm tra: 7 kiểm tra CPU local đạt, 2 kiểm tra geometry đã đạt trên T4 ở job SAM 3;
  bản vá áp vào nguồn CountVid chính thức và compile thành công. Bộ 9 kiểm tra sẽ chạy trước model ở các job mới.
- Đã tải kết quả C6 exother cũ: checkpoint tự báo arch=simcentric, test MAE 23.49 và mosaic oracle 43%; không thay exnoise.
- Theo dõi nền mỗi 60 giây: `outputs/video_benchmark/watcher.log`; tiến trình gửi tiếp khi có slot,
  tự mở rộng sau pilot, dừng dispatch khi lỗi. Máy local chỉ đóng gói/API/tổng hợp, không chạy model.

- Lượt baseline Penguins-1 đã suy luận xong (log ghi Total Number of Objects: 26, GT 28), nhưng lỗi serializer vì
  CountVid cần file output JSON tồn tại sẵn. Đã khôi phục khởi tạo `{}` trước subprocess và bổ sung test mô phỏng
  đúng contract này. Lượt lỗi được giữ trong history/archive và chạy lại; không tính vào metric chuẩn.
- Sửa test packaging dùng fixture checkpoint (không đòi asset local ở bước kiểm tra trên Kaggle),
  xử lý tranh chấp khóa để watcher chờ thay vì thoát. Các job đã gửi bằng evaluator cũ được thử lại đúng một lần.
- Kết quả đối chiếu `sam3_native` (detection 0.5, output conf 0.3), Penguins-1: GT 28, raw 39,
  hits>=3 37; 34 frame, 191.1 giây. Hoàn tất và đã tải JSON/trace; không coi là kết quả toàn bộ Penguins.
  Bộ 8 test trong snapshot này đạt toàn bộ trên Kaggle. Snapshot sau bổ sung test khóa hàng đợi (tổng 9).
- Watcher đã khởi động lại sau khi sửa khóa; đang theo dõi 2 tài khoản, chốt quota 2 session/tài khoản.
  Trạng thái mới nhất và history là `experiments/video_tasks.json`; lỗi setup/serializer được lưu archive riêng.

## 2026-11-07 — Kiểm tra tiến độ và sửa giới hạn tải dữ liệu

- Penguins-1 (GT 28), các lượt hoàn tất và đã xác minh: CountYOLO cũ 28 (1268.2 s),
  fusion 26 (1167.9 s), CountVid gốc 26 (1144.3 s), SAM 3 native 39 (191.1 s).
  Fusion chưa tốt hơn bản cũ trên video này; chưa chọn cấu hình thắng khi thiếu MOT20.
- Hai lượt tiếp bị Google Drive quota trước suy luận. Watcher dừng đúng chốt lỗi, không tự tiêu GPU lặp lại.
- Tạo hai dataset riêng tư `nadthien/videocount-benchmark`, `ttdan2006/videocount-benchmark`:
  1252 frame JPEG, 354.7 MiB, đủ 13 video. Giữ nguyên ảnh gốc; MOT20 được lấy mẫu đúng CountVid một lần.
  Đã kiểm chứng cả 13 case, danh sách frame/hash, GT và từng byte ảnh trùng dữ liệu gốc.
- Evaluator nhận biết cache presampled và đặt downsample_factor=1 cho CountVid để không lấy mẫu hai lần.
  Mọi job mới gắn cache Kaggle; không tải VideoCount từ Drive nữa. Có hash sampling trong run_manifest.
- Lưu lỗi cũ vào history/archive, gửi lại SAM 3 native MOT20-01, exnoise Penguins-1 và baseline MOT20-01.
- Test: 8 kiểm tra CPU đạt (gồm regression cache không lấy mẫu hai lần); 2 kiểm tra geometry dùng Torch
  đã đạt trên Kaggle ở các lượt trước. Bộ mới tổng 10 kiểm tra sẽ chạy trước mỗi lượt suy luận.

## 2026-11-07 — Cache CountGD-Box sau xác nhận upload

- Người dùng xác nhận upload `checkpoints/countgd_box.pth` lên dataset riêng tư `ttdan2006/countvid-weights`.
- Upload hoàn tất; Kaggle báo ready, file 954749856 byte. SHA-256:
  `dc550ecb35dec9bb6608c53523a250eac84a6208bce00ab444cadb4a0903cfcd`.
- Baseline mới gắn dataset này, kiểm checksum rồi nạp model; bỏ tải CountGD-Box từ Google Drive.
  Ghi checksum vào run_manifest và kiểm lại khi tải kết quả. Lượt lỗi quota cũ được lưu history/archive.
- Kiểm tra CPU: 9 đạt, 2 geometry cần Torch vẫn được kiểm trên Kaggle. Bộ hiện tại 11 kiểm tra.

## 2026-11-08 — Kết quả qua đêm và chẩn đoán tiếp

- Hàng đợi chính đã chuyển sang full sau pilot, chọn `countyolo_old` theo MAE pilot 2.0
  (fusion 6.0, kf5 12.5, exnoise 18.0). Tổng 43/47 lượt hoàn tất; 4 lượt lỗi đều timeout 14400 s:
  CountVid và CountYOLO cũ trên MOT20-02, MOT20-05. Benchmark toàn bộ chưa hoàn tất.
- Penguins đủ 3 video: MAE CountYOLO cũ 2.333, CountVid 3.333, SAM 3 Ultralytics 20.667.
- Crystals đủ 7 video: MAE CountYOLO cũ 74.714, CountVid 69.143, SAM 3 Ultralytics 48.857.
  Cả 7 clip CountYOLO cũ đếm 0. Đối chiếu log: mọi SAM 3 keyframe đều không có detection, nên hard gate
  chặn mọi proposal trước SAM 2. Đây là lỗi chính sách của pipeline, chưa đo được năng lực density head khi không bị chặn.
- Thêm `absence_policy=fallback` tùy chọn: SAM 3 không thấy vật -> chạy CountYOLO với exemplar tự chọn;
  mặc định block giữ nguyên đối chứng. Dùng seed3, không đổi checkpoint hoặc thêm fusion.
- Gửi 2 lượt chẩn đoán riêng trên Crystals ma2035_023 và ma2035_039. Đây là development sau khi quan sát heldout,
  không trộn với benchmark cũ. Ledger: experiments/video_diagnostics.json.
- Thêm baseline_batch30 trên MOT20-01 (pilot) để kiểm tra tính tương đương số đếm và tốc độ trước khi retry video dài.
  Không chạy lại mù 4 lượt timeout. Model/prompt/frame/threshold giữ nguyên, chỉ đổi obj_batch_size 10 -> 30.
- Cả 3 lượt chẩn đoán đã được gửi. Quota tại thời điểm kiểm tra: nadthien dùng ~19.32/30 GPU giờ,
  ttdan2006 ~14.28/30; thời gian reserved được ghi bởi Kaggle, không phải số giờ đã thực dùng.
- Kiểm tra: 9 kiểm tra CPU local đạt, 3 cần Torch chạy trên Kaggle (gồm regression absence gate);
  bản vá áp đúng nguồn CountVid chính thức và compile thành công. Tổng hiện tại 12 kiểm tra.
- Tách thư mục export theo ledger để chẩn đoán không ghi đè báo cáo benchmark chính.


## 2026-11-08 — SAM3 setup audit

- Inspected exact PyPI Ultralytics 8.4.204 and Meta SAM3 source at commit
  f1647e8d0c1d280991f4d1b7d974c2704537fe66. Confirmed BGR input -> RGB
  preprocessing, image size 1008 and mean/std 127.5 are consistent.
- Ultralytics default tracker uses association .5, new detection threshold 0,
  hotstart delay 0, while Meta temporal builder uses .1, .7, 15 respectively.
  Other exposed Meta settings are recorded in video/sam3_policy.py. This does
  not imply implementation parity with Meta (including lifetime heuristics).
- Added delayed CPU frame records. Revocations affect pending frames only;
  historical sightings of legitimately departed objects remain in the union.
- Aligned postprocess drops final empty masks and recomputes remaining boxes;
  checkpoint loading fails on missing required model parameters instead of
  silently accepting an incomplete mapping with strict=False.
- Added paired FP16/FP32 video jobs and a bounded image numerical probe. SAM3
  weight checksum, precision, tracker settings and peak allocated CUDA memory
  are recorded. Keep old benchmark results unchanged.
- 16 checks: 13 passed locally, 3 torch-dependent checks reserved for Kaggle.
  Three SAM3 diagnostics prepared in the existing diagnostic queue; watcher
  PID 59380 polls every 60 s. GPU run results remain pending at preparation.


## 2026-11-08 — Three-account SAM3 routing prepared

- Verified thiend27 authentication, no active kernels in recent runs, 0/30 GPU
  hours used; ttdan2006 14.84/30 hours, batch30 CountVid job complete.
- Prepared two aligned SAM3 pilots for thiend27, FP32 plus image probe for
  ttdan2006. Staging plan: outputs/sam3_three_accounts/routing.json. Main queue
  is unchanged until access is approved and verified.
- Auto-review rejected private dataset reader grants because explicit dataset
  and destination account approval was required. Requested that approval:
  nadthien/sam3-weights -> thiend27, ttdan2006 (READER);
  nadthien/videocount-benchmark -> thiend27 (READER). Datasets stay private.
  No sharing or duplicate upload was performed. Metadata snapshots saved.
- Added explicit cross-account data_dataset/sam3_dataset packaging, preserved
  during refresh. 17 checks: 14 pass locally; 3 require Kaggle torch.


## 2026-11-08 — Three-account SAM3 execution activated

- User explicitly approved the exact READER grants. Applied them, verified
  both SAM3 weight readers and thiend27 benchmark access; datasets stay private.
- Routed only the three still-prepared SAM3 jobs, preserving routing history:
  thiend27: aligned Penguins-1, aligned MOT20-01;
  ttdan2006: aligned FP32 Penguins-1 with paired image precision probes.
- All three Kaggle submissions accepted. Existing two CountYOLO diagnostics
  remain on nadthien. Diagnostic watcher PID 59380 continues every 60 seconds.
- Completed CountVid batch30 pilot: 98/80 on MOT20-01 (same count as batch10),
  1777.93 seconds for 52 frames. Parity verified on this pilot only.


## 2026-11-08 — SAM3 setup results and next validation

- All three SAM3 pilot diagnostics completed, all 17 remote tests passed;
  both detector and tracker checkpoint audits report zero missing parameters
  and buffers. Extra checkpoint keys are ignored by the upstream submodel maps.
- Penguins-1, 34 identical sampled frames, GT28: legacy39 (191.07s),
  aligned FP16 27 (178.03s), aligned FP32 27 (631.77s). Raw aligned IDs30;
  delayed emitted IDs27. FP32 did not improve this count and cost ~3.55x.
- MOT20-01, 52 identical sampled frames, GT80: legacy92 (442.59s),
  aligned FP16 70 (428.90s); raw IDs74. Two-case MAE11.5 ->5.5.
  These are pilot/development observations, not a full benchmark claim.
- Numerical probes: Penguins first-frame image count33 in both precisions;
  two Crystals first-frame image counts0 in both. All logits/boxes finite.
  Low image presence logits suppress the image scores below conf .3. No
  evidence of a precision failure on these probes; do not bypass presence
  gating or change text/thresholds to fit known counts.
- CountYOLO absence fallback ma2035_023 completed:46/111 in4050.01s,
  vs old hard-gated0. Second fallback Crystals task is still running.
- Continue frozen aligned FP16 settings on Penguins-2/3 (thiend27) and
  Crystals ma2035_023/039 (ttdan2006). All four submissions accepted;
  keep old benchmark and app defaults for now. No new threshold fitting.


## 2026-11-08 — Validation exposed tracker memory failure

- Diagnostic queue terminal:9 complete,1 failed; watcher59380 exited normally.
- Aligned Penguins3:39/38; Penguins2 failed CUDA OOM in RoPE memory attention
  (1.25GiB allocation with ~995MiB free on a14.56GiB T4). Do not report a full
  Penguins metric for aligned until Penguins2 completes.
- Aligned Crystals023:1/111;039:8/27. Fixed Meta-like settings are not a general
  solution: severe undercount remains on these clips. Preserve those results.
- CountYOLO fallback039:73/27 in10089.07s;023:46/111. Removing hard absence
  restores proposals but is not sufficient for accurate crystal counting.
- Added sam3_aligned_batch8, preserving resolution/thresholds/all detections,
  splitting newly detected objects into ordered tracker states of at most8.
  Test all IDs/masks/order preserved; no cap on total objects.18 tests:
  15 passed locally;3 require torch on Kaggle.
- Separate ledger experiments/sam3_memory_tasks.json retains the old OOM task,
  runs Penguins1 count parity plus Penguins2 OOM recovery on thiend27.


## 2026-11-08 — SAM3 tracker batch8 passed OOM recovery

- Both memory diagnostics complete; all18 remote tests passed on each job.
- Penguins1:27/28 unchanged from aligned unbatched;34 frames;177.37s;
  peak allocated CUDA4156.77MiB vs7518.77MiB before (about45% reduction).
- Penguins2:54/59 across all94 frames;839.38s;peak7267.48MiB. This clip
  previously failed at~13.24GiB allocated in tracker attention. Batch8 completed
  with no frame skipping, total-object cap, resolution or threshold change.
- No full same-profile3-video Penguins score yet: Penguins3 is still the
  unbatched aligned run. Crystals accuracy remains unresolved. Keep legacy
  benchmark and app defaults, identify batch8 explicitly in new evaluations.
- Queue experiments/sam3_memory_tasks.json:2/2 complete, no pending tasks.
  Watcher85620 exited normally after downloading and verifying both outputs.
# 2026-11-11 — UI CountYOLO chạy đủ ba stage với SAM2

- Thay tracker nhẹ trong tab video bằng proposal CountYOLO → mask độc lập SAM2
  → lọc ngắn hạn w=3 (chuỗi khớp mask IoU >0.5) → tracking dài hạn và kiểm tra
  mask mới không chồng lấn với vật đã theo dõi. SAM 2.1 large, object batch 8,
  offload video/state sang CPU; SAM2 cài thêm vào `.venv` hiện có, không đổi torch.
- Adapter giữ ý tưởng/tiêu chí ba stage của CountVid, có thêm lọc mask rỗng,
  mask độc lập trùng IoU ≥0.85, xử lý frame đầu rỗng và cấp ID qua nhiều batch.
  Đây không phải cấu hình benchmark cũ; không gán lại các MAE cũ cho UI mới.
- Clip người dùng `vid_000001.mp4`: 30 FPS, 151 frame; lấy 3 FPS, chỉ 10 frame
  đầu, prompt `cube`, head exnoise, min_mass=0.2. Mắt thường thấy 5 cube.
  Tracker appearance cũ:25; position:30. Pipeline SAM2 mới:10, 140.44 giây
  (Stage1 20.17s, Stage2 96.24s, Stage3 11.60s; tổng gồm nạp SAM2).
- Proposal/frame:19,19,21,21,17,20,20,16,17,22. Sau temporal filter:
  13,13,12,12,16,16,14,11,11,8. Preview mask cho thấy các mặt của cùng cube
  vẫn bị phân đoạn thành các ID riêng; SAM2 không tự gộp các mask rời nhau.
  Không chỉnh ngưỡng để ép kết quả về 5 trên clip này.
- Artifacts local: `outputs/cube_video_diagnostic/sam2/` (JSON, GIF và ảnh mask).
  5 kiểm tra logic mới + 18 kiểm tra pipeline cũ đạt; Gradio build thành công.
# 2026-11-13 — Đối chiếu UI và terminal trên cùng đầu vào

- UI trước đây mặc định tối đa 120 frame, CLI dùng toàn bộ; UI/CLI còn có hai
  cách sắp xếp tên ảnh. Penguins có 34/94/94 frame nên mức mặc định 120 không
  cắt chúng, nhưng UI còn giữ giới hạn 10 từ thử cube sẽ khác CLI không giới hạn.
  Không lưu được cấu hình UI của lần người dùng báo thiếu nhiều, nên chưa quy
  nguyên nhân cụ thể cho lần chạy đó.
- Kiểm tra riêng CLIP trước/sau chuyển precision không đổi density/box trên
  frame Penguins đầu tiên; không có bằng chứng CLIP là nguyên nhân.
- Thống nhất đọc frame và suy ra prompt qua `video/frame_input.py`; UI dùng
  0=toàn bộ như CLI, không fallback âm thầm sang video khác khi path frame sai.
  Thêm JSON UI, hash nội dung frame/checkpoint/SAM2 và hash mã pipeline; cache
  model UI nạp lại nếu checkpoint trên đĩa được thay thế.
- Chạy thật 3 frame đầu penguins_1, exnoise, penguin, min_mass=.2, w=3, batch=8:
  UI có startup warmup và CLI đều ra16; proposal_counts, independent_counts,
  filtered_counts, frame_counts và fingerprint giống hệt nhau. Đây là kiểm tra
  parity ngắn, không phải benchmark lại toàn bộ clip.
- Artifacts: `outputs/ui_terminal_parity/{ui,terminal,comparison}.json`.


## 2026-11-17 ? Exemplar reliability development experiment

- Added opt-in reliability selection: positive CLIP target-vocabulary margin,
  local density mass and detector or confirmed temporal support. Scores are
  relative evidence, not calibrated probabilities; YOLO-World and CLIP are
  correlated sources. Fixed rules are declared before GPU evaluation.
- Select distinct feature cells without repeating weak candidates to fill K.
  Rejected candidates have zero weight in P3/P4/DINOv2 matching. If all are
  rejected, use the separately supervised coarse density instead of forcing
  a prototype or returning zero. This fallback may still produce false positives.
- A short-lived CLIP feature anchor requires two consecutive detector-supported
  frames. Temporal-only support cannot refresh its own TTL. No cross-frame
  P3/P4/DINOv2 prototype reuse and no change to SAM2 tracking/thresholds.
- Default UI/self-exemplar path is retained; CLI enables the experiment with
  --exemplar-policy reliable. Added per-frame exemplar/proposal/mask/new-ID
  traces and cumulative count, plus paired evaluation CSVs and mask snapshots.
- Local: 10 reliability tests, 5 SAM2 logic tests, 5 frontend tests and 18 video
  pipeline tests passed. GPU smoke verified identical control decoder and
  exact coarse fallback. One repeated frame probe is not accuracy evidence.
- Prepared private ttdan2006/countyolo-exemplar-reliability-20261117: full
  Penguins-1 (34 frames) and Crystals ma2035_039 (41), control versus reliable,
  same exnoise checkpoint, pixel fingerprints, decoder and SAM2 backend.
  This is a development diagnostic, not O2 evaluation or a heldout benchmark.

- GPU integration smoke on first 3 Penguins frames completed: reliable count11,
  proposals20/20/19, selected5/5/5, final cumulative trace11. No prefix GT was
  assigned; this is integration evidence only, not an accuracy score.
- Kaggle accepted version1, kernel145604859, private ttdan2006 account. Initial
  remote state QUEUED. Bounded watcher polls60s and fetches only small artifacts;
  paired results require matching frame/checkpoint/SAM2 hashes and settings.

- User approved routing exemplar diagnostic to nadthien. Private version1 accepted (kernel145607767); source payload byte-identical to ttdan2006 job, using existing nadthien/videocount-benchmark. Active local watcher moved to nadthien; original ttdan2006 queued job has not been cancelled and is retained in experiments/exemplar_reliability_ttdan2006_20261117.json.

- User requested cancellation of ttdan2006 duplicate exemplar. Verified private kernel145604859 was QUEUED, preserved metadata/source locally, then deleted that duplicate via official API. CountVid baseline on ttdan2006 was still RUNNING afterward; nadthien exemplar remains the active evaluation.

- Follow-up: nadthien version1 failed before inference while building pinned CLIP under isolated packaging on Kaggle Python3.13. Verified upstream setup imports pkg_resources; pinned setuptools80.9.0/wheel0.45.1 and disabled CLIP build isolation. Version2 accepted, initial QUEUED. Algorithm payload identical, no new tuning. Failed attempt and logs preserved in outputs/exemplar_reliability_kaggle_attempt1; bounded watcher restarted for v2.

### 2026-11-17: opt-in openworld-v2, user-approved logic improvements

- Added independent density peak proposals (no integral-count cap), detector
  geometry fusion, coarse proposal retention and a reported 256-region budget.
  Density-only boxes remain square estimates; this does not solve full bbox
  localization or small-object resolution. No new detector training occurred.
- Detector-free exemplar initialization, explicit equivalent-name aliases,
  absolute margin coarse/refine blending, final-region CLIP inspection and
  one-to-one spatial/appearance confirmation. Pending regions do not enter SAM2.
  Region IDs are temporary associations, not global identities.
- New CLI mode `--exemplar-policy openworld`; evaluator supports
  `--modes control reliable openworld`. Existing default and running Kaggle
  payload are unchanged; no v2 accuracy claim or new remote submission.
- Nine new behavioral regressions and twenty existing exemplar/SAM2/frontend
  checks passed. Four-frame real Penguins-1 CUDA integration passed in 22.6s:
  raw proposals 25/22/23/23, confirmed 0/7/8/9, SAM2 filtered 0/1/3/2,
  final unique count 3. First-frame density 16.91 produced 25 raw proposals.
  No prefix GT assigned: this is execution evidence, not accuracy validation.
  The combined confirmation plus SAM2 persistence is potentially too strict;
  thresholds were not tuned on this smoke clip. Local SAM2 extension `_C`
  unavailable, so optional hole-filling postprocessing was skipped.
- Artifact: outputs/exemplar_reliability_local/openworld_v2_smoke.json.
  Remaining limits: CLIP parent/child ambiguity, relative uncalibrated evidence,
  short/fast appearances, coarse square geometry and no global ReID.

### 2026-11-17: FSC147 validation selects a simpler proposal improvement

- Executed a fixed-seed 58-image pilot (2/class), then all 1,286 validation
  images, 29 classes, exnoise checkpoint, self-exemplars, no GT exemplar input.
  Test split untouched. Count errors and one-to-one point localization at
  radii 8/16 pixels in resized 640-space are separate; not bbox AP.
- Full v2 failed to improve: density MAE 28.1015 vs original 23.3608;
  static CLIP hard-gated proposal recall@16 only 21.7962%. 494 images selected
  no exemplar. Do not promote v2 or hard CLIP gating based on these results.
- Isolated uncapped density decoder: initial 256-candidate compute budget
  truncated 24 images. Removed that CLIP-specific budget from density-only
  `recall`, then reran all validation. Paired image hashes and density values
  matched exactly across both full runs.
- Original capped proposals -> all valid density peaks (min mass .2):
  proposal-count MAE 25.0389 -> 23.3188; RMSE 89.7105 -> 86.1309;
  point precision@16 81.2515% -> 78.2002%; recall 56.3419% -> 62.8617%;
  F1 66.5419% -> 69.6970%. F1@8 also improved 52.3969% -> 54.5881%.
  Original density integral is unchanged (MAE 23.3608), so do not describe this
  as improved density prediction. No thresholds fitted in these runs.
- Keep `--exemplar-policy recall` opt-in for the next video ablation; default
  stays self. More proposals can increase SAM2 cost/false tracks. Geometry
  fusion did not outperform unfused peaks on these point metrics; actual
  box geometry remains unvalidated without box annotations.
- Added independent static-image evaluation (no temporal confirmation/memory),
  source/checkpoint/image fingerprints, per-image diagnostics and source copies.
  31 regression tests passed including >256 density peaks and unchanged counts.
- Summary: experiments/fsc147_openworld_validation_20261117.json.
  Full artifacts: outputs/fsc147_openworld/full_val and full_val_uncapped_all.

### 2026-11-17: v3 remediation implemented and evaluated

- Fixed connected flat maxima (one representative per plateau, valleys remain
  separate). Added one-to-one geometry fusion without global NMS on density
  boxes; a detector containing multiple density centers cannot erase them.
  Density anchors survive geometry replacement and are logged separately.
- Added peak-centered adaptive exact ROIAlign crops (nearest-center spacing,
  side 8..96, 64-crop semantic budget independent of proposal recall), no final
  CLIP veto, self fallback, and isolated rank/weight/blend/coarse-fallback variants.
  Tested optional ROI-average prototypes separately. No checkpoint retraining.
- Added coarse/unblended/final counts, crop coordinates, feature centers,
  proposal sources, detector rejection reasons and exact float32 crop exports.
- 40 regression checks passed. Full 1,286-image validation baseline hashes and
  density predictions exactly matched previous results. Plateau removed zero
  cells in this real set (an edge-case fix, not a measured gain). Geometry
  rejected 543 multi-center detector boxes. V3 fallback occurred on 574 images.
- Full validation: recall/plateau proposal MAE 23.3188, F1@16 .69697;
  rank 23.8048/.69368, weighted 24.7271/.68659, blend 24.1407/.69270,
  coarse fallback 24.6034/.68530. Geometry anchors F1 .69701, box centers .69604.
  Region-pooling pilot MAE 20.7586/F1 .62846 vs point-rank 19.3793/.65720;
  not promoted. Exemplar changes did not outperform existing recall overall.
- Froze plateau+original self-exemplar before one full 1,190-image FSC147 test.
  Policy hashes verified unchanged after test. Original capped -> selected:
  MAE 17.67395 -> 17.59244, RMSE 114.26135 -> 114.90968;
  P@16 .79656 -> .76451, R .69007 -> .74323, F1 .73950 -> .75372.
  Recall/F1 improved, precision and RMSE worsened; no uniform superiority claim.
- Important unresolved semantic failure: on two visually inspected validation
  images with no elephants, recall/rank/region still produced 437 and 264
  proposals. In the synthetic mixed image, the polka-dots query put 311/312
  proposals on the bottle-cap half. These tiny sanity checks are not general
  benchmarks, but prevent claiming open-world correctness from FSC147 alone.
  No thresholds fitted on these checks or on test; no model promotion/default
  change. Semantic training needs verified negatives and mixed-class supervision;
  random alternate FSC147 prompts alone do not establish absence.
- Six-frame Penguins-1 paired SAM2 integration completed for recall, plateau,
  geometry: all returned 19 IDs. Prefix GT intentionally null and accuracy
  metrics disabled. Local SAM2 optional hole-fill extension unavailable.
  This validates integration, not full-video accuracy or global ReID.
- Final summary: experiments/proposal_v3_results_20261117.json.
  Frozen config: experiments/proposal_v3_frozen_20261117.json.
  Artifact folders: outputs/fsc147_openworld/v3_*.

### 2026-11-17: original CountVid detection-only comparison

- Ran original CountGD stage from pinned CountVid commit
  690ca3e48369c35f5e23a3ab5d2b04b56881a343, original countgd_box checkpoint,
  text-only, short-side800/max1333, max token sigmoid >.23, no NMS/tiling/SAM2.
  Local CUDA float32, upstream PyTorch deformable-attention fallback (no custom
  extension). BERT initialized from matching config then fully overwritten by
  original checkpoint; local BERT tokenizer used. All model keys loaded; only
  unused feature_map_encoder checkpoint keys ignored as by upstream strict=False.
- Pilot only: same 58 validation images, 29 classes, verified identical pixels
  and GT against saved CountYOLO v3 pilot. Compare native preprocessing, not
  matched-resolution architecture ablation. Point metrics in common 640-space.
- CountGD vs CountYOLO plateau: MAE 12.6379 vs 20.3966; RMSE 48.4471 vs 60.1408;
  P@16 .79705 vs .72105, R .69518 vs .57365, F1 .74264 vs .63896.
- Two visually reviewed absent-elephant images: CountGD 0/0 versus CountYOLO
  437/264. Mixed bottle caps: CountGD399 (GT436), zero wrong-half boxes;
  mixed polka dots: CountGD410 (GT443), but393 boxes on wrong half. CountYOLO
  produced312 with311 on wrong half. Both fail that fine-grained mixed prompt;
  close aggregate count alone is not evidence of correct object recognition.
- Runner: tools/eval_countvid_detect.py. Raw boxes/scores and environment audit:
  outputs/countvid_detect/pilot58. Comparison:
  experiments/countvid_detect_comparison_20261117.json.
  No full-validation/test or full CountVid tracking accuracy claimed here.

### 2026-11-17: controlled semantic training pilot (rejected)

Video v3 submission follow-up: private Penguins full-video paired control/plateau
job submitted to nadthien/countyolo-video-v3-penguins-20261117. Original exnoise
checkpoint; no rejected semantic weights. 24 proposal/SAM2/runner tests passed.
Crystals submission hit Kaggle maximum2 batch GPU sessions; MOT20 not submitted.
Switch to thiend27 was rejected by automatic approval review pending explicit
account authorization. Those two job packages remain prepared locally.
User subsequently explicitly authorized thiend27. Both jobs were submitted
successfully: thiend27/countyolo-video-v3-crystals-20261117 (7 videos) and
thiend27/countyolo-video-v3-mot20-20261117 (3 videos), initially QUEUED.
Each runs full sampled sequences for control and plateau with original exnoise
and SAM2. Submission records: experiments/video_v3_{crystals,mot20}_20261117.json.

2026-11-18 check: Penguins v3 session log confirms all6 video/mode cases complete.
GT28/59/38; control21/46/35, plateau21/47/36. Control MAE7.6667 RMSE8.6987;
plateau MAE7.0000 RMSE8.1035. CountVid baseline26/51/38, MAE3.3333 RMSE4.7610.
Counts extracted from structured JSON log lines; remote JSON output download
intermittently times out. Saved extracted rows in
experiments/video_v3_penguins_log_results_20261118.json.
Both Crystals and MOT20 v3 ended ERROR before inference: data discovery raises
Expected exactly one pre-sampled VideoCount cache. No model metrics for them.
No job resubmitted in this status-check turn.

### 2026-11-18: DINOv3 real-weight static pilot

Follow-up requested by user: deleted experiment/dinov3-exemplar-detect branch,
standalone DINOv3 detector/head/train/test scripts and unsubmitted Kaggle package
and weight staging cache. Returned to master with unrelated workspace changes
preserved. Historical experiment results retained. No DINOv3 Kaggle job submitted.

Replaced frozen feature extractor inside main CountYOLO in an opt-in paired test,
same exnoise trained heads and self-exemplars,58 fixed validation images, no GT
exemplars, no training or threshold tuning. Channel dimensions remain384.
DINOv2@560: proposal MAE20.3966 RMSE60.1408, density MAE20.8906,
P@16 .72105 R .57365 F1 .63896.
DINOv3@560 (same input pixels,35x35 patches): proposal MAE26.2069 RMSE56.0668,
density MAE22.8574, P .59498 R .65786 F1 .62484.
DINOv3@640 (same40x40 patch grid): proposal MAE25.8103 RMSE55.9700,
density MAE22.8635, P .59939 R .65026 F1 .62379.
All trainable head weights checked unchanged by hash during evaluation.19 existing
matcher/proposal tests pass. Recall improves but precision/MAE worsen; no promotion.
Exported both DINOv3 swap candidates with vit_revision and explicit transfer-only
metadata under outputs/backbone_swap_dinov3/pilot58. This is a distribution-transfer
diagnostic with DINOv2-trained heads, not a retrained backbone ranking.
Summary: experiments/dinov3_backbone_swap_20261118.json.

### 2026-11-18: main-pipeline backbone adaptation started

- tools/adapt_backbone.py retains original CountYOLO architecture and exnoise
  initialization. Two matched arms: frozen DINOv2-S@560 continued-training
  control, frozen DINOv3-S@640 with adapted heads. Existing trainable heads,
  projections, matchers and scale head updated; neither backbone nor YOLO trained.
- Full3659 train /1286 validation,3 epochs per arm, batch4, lr1e-5, float32,
  self-exemplars only, no GT exemplar input, no random negative labels, no test.
  Original multi-scale count, density/coarse and scale losses retained. Seed
  reset after backbone construction to match augmentation/dropout RNG sequence.
- Select by validation point F1@16 with original DINOv2 density-MAE retention
  guard1.10; initial candidate retained. No automatic default promotion.
- Smoke test completed both arms through training, validation, checkpoint saving;
  verified actual head parameter/buffer change and DINOv3 checkpoint metadata.
  Existing19 matcher/proposal regression tests passed.
- Kaggle private package prepared: nadthien/countyolo-backbone-adaptation-20261118.
  Upload stalled with API connection/DNS errors; canceled retry process before
  submission. Ledger experiments/backbone_adaptation_kaggle_20261118.json records
  blocked_network_not_submitted. No claim of remote training.
- Full offline-cache local GPU run started instead, PID32512, output
  outputs/backbone_adaptation/full_train_val. Initial DINOv2 full-val proposal
  MAE23.3087, density MAE23.3462, F1 .696853; training epoch1 started915 steps.
  Run writes live_progress.json, epoch metrics, best/last head checkpoints and
  comparison.json. This log records launch status, not completed training results.

Retry requested by user: private cache upload completed successfully after adding
bounded connection timeouts/reconnect retries. Submitted
nadthien/countyolo-backbone-adaptation-20261118; Kaggle initially QUEUED. Same
paired3-epoch main-pipeline head-adaptation experiment, no new standalone detector.
Weights in private nadthien/countyolo-dinov3-adapt-cache, verified by SHA256 in
runner; no HF token included. Ledger updated from network-blocked to submitted.

First Kaggle adaptation attempt failed before training: pathlib.rglob found no
cache manifest although dataset status ready and all8 expected files existed.
Changed input discovery to recursive glob following symlink mounts, deduplicated
resolved paths; applies to both cache and FSC147 split discovery. Resubmitted
without reuploading weights. Updated code SHA256
2b7a7884ab8209d5b68755c696e274ca74862a7461e0831321b1fc128e0128e2;
Kaggle returned RUNNING. Runtime estimate1-2h is provisional, not measured Kaggle
epoch throughput yet; local control epochs took roughly390-442seconds each.

Completion check: Kaggle and local both finished all3 epochs per arm on3659
train/1286 val images. Downloaded16 small JSON records plus final session log.
Kaggle initial DINOv2: proposal MAE23.3002, density MAE23.3515, F1 .69707.
Unadapted DINOv3: proposal MAE30.5078, density MAE29.2671, F1 .63796.
Adapted DINOv3 epoch3: proposal MAE23.8235, RMSE86.1903, density MAE26.1452,
P .75417 R .63720 F1 .69077. Improves strongly over raw swap but does not
beat original DINOv2 on MAE/F1. Same-budget DINOv2 epoch3 MAE24.0498/F1 .69871;
best F1 candidate epoch2 MAE23.9852/F1 .70073. These criteria trade off count
error and localization; no automatic promotion.
All DINOv3 candidates fail predeclared density MAE guard1.10*23.3515=25.6866;
remote comparison.json best=null confirms rejection. Local final DINOv3
MAE23.9199/F1 .69018, similar direction. Three epochs do not establish convergence.
Results: experiments/backbone_adaptation_results_20261118.json and
outputs/kaggle/nadthien__countyolo-backbone-adaptation-20261118/adaptation/.

### 2026-11-18: requested10-epoch experiments submitted

- nadthien/countyolo-backbone-adaptation-20261118-e10: fresh matched10 epochs
  per DINOv2/DINOv3 arm, original exnoise initialization. Same3-epoch protocol
  and retention guard; frozen image backbones, adaptation of existing heads.
  Kept original completed job and records; not a3-to10 optimizer continuation.
- nadthien/countyolo-dinov3-only-finetune-e10-20261118: DINOv3-S/16 sole image
  backbone, new exemplar-conditioned density/point head. Official within-image
  exemplar boxes, no text grounding, YOLO, CLIP or SAM in model/inference path.
  Head warm-up epochs1-2, last2 backbone blocks and final norm unfrozen3-10;
  head lr1e-4/backbone1e-6, batch4. Full3659train/1286val, no test.
- DINOv3-only3 unit tests pass: freeze/unfreeze gradients, tiny exemplars,
  checkpoint round-trip. Pretrained GPU batch4 smoke completed2 epochs through
  both stages and validation/saving. Compared saved backbone against original:
  36 tensors changed, all confined to last2 blocks/norm. Smoke is not accuracy.
- Both jobs independently confirmed RUNNING. Reuse private checksummed cache;
  no HF tokens in packages. Only-model scores have GT exemplar assistance and
  cannot be directly ranked against main pipeline text/self-exemplar scores.
- Ledgers: backbone_adaptation_kaggle_20261118-e10.json and
  dinov3_only_kaggle_20261118.json. No checkpoint promotion before results.

Completion check of both10-epoch jobs: Kaggle COMPLETE, all10 epochs per arm,
full1286 validation cases. Downloaded JSON histories/per-image validation metrics.
Main-pipeline DINOv3 epoch10 proposal MAE23.0210, F1@16 .704165 versus original
DINOv2 MAE23.3002/F1 .697070; same-budget DINOv2 epoch10 MAE23.9767/F1 .699805.
This is a small development-validation detection improvement, not test evidence.
However DINOv3 epoch10 density MAE26.1995 exceeds1.10*23.3515 retention guard.
Guard-selected DINOv3 checkpoint is epoch5: MAE24.0645/density25.2766/F1 .690542.
No default promotion; keep epoch10 results distinct from guarded best.pth.
DINOv3-only best by F1 is epoch9: proposal MAE30.4984/RMSE108.0420,
density MAE24.8109, P .79445/R .46530/F1 .586872; epoch10 F1 .585011.
Only-model's official exemplar input differs from main self-exemplar/text task;
do not present this as a controlled backbone architecture ranking.
Summary: experiments/dinov3_e10_results_20261118.json. Both ledgers COMPLETE.

### 2026-11-18: semantic-path audit, original and adapted DINOv3

- tools/audit_semantic_path.py ran58 fixed positive validation images plus6
  sanity cases (two original positives, two reviewed absent elephants, two
  synthetic mixed prompts). All source images validation-only. No training,
  no threshold tuning, no general absent-class benchmark claim.
- Compared YOLO text score maps, coarse density, refined density; decoded YOLO
  at .05/.15, density3x3 mass .05/.2. Fixed top64 refined proposals per image
  tested with existing CLIP margin>=0 separately from baseline predictions.
- Mixed polka-dot target on right: YOLO text-map wrong-half mass fraction .0421
  (low absolute max confidence .0508). Original coarse .8517 ->refined .995;
  adapted DINOv3 coarse .9397 ->refined .962. Thus some prompt-dependent spatial
  signal is present upstream but not preserved by density/refinement. This is
  observed behavior; does not isolate an individual layer or prove sole cause.
- No elephants: original refined proposals437/264, adapted432/211, despite
  YOLO having0 boxes at .15. Correct classes can also have low YOLO confidence,
  so no unconditional detector-presence veto introduced.
-58 positive images, adapted refined P .74117/R .58917/F1 .65649 at mass .2.
  YOLO center-matching recall .07530 at .05 and .02609 at .15; sparse YOLO boxes
  do not supply sufficient instance centers for these dense counting cases.
- Fixed64 candidate-pool CLIP gate: original TP1274 ->454, P .6996 ->.6647,
  recall .4207 ->.1499. Adapted TP1345 ->598, P .7460 ->.7410, recall .4442
  ->.1975. Gate discards valid detections without precision benefit on this
  pool; not evidence that CLIP encoder generally fails. Point labels do not
  give full semantic box-content annotations and large crops may mix concepts.
- Repeated identical prompt after intervening prompts gives max abs density
  difference0 for both models; no stale text-cache symptom in this test.
- Raw maps/proposals: outputs/semantic_path_audit/{original,dinov3_e10}.
  Summary: experiments/semantic_path_audit_20261118.json. No model/default changes.

### 2026-11-18: proposal ceiling, region/prompt pilots and code logic review

- Oracle audit58 positive val images: refined mass .2 one-to-one center recall
  .58917 at16px; mass0 maxima .69914. Union of coarse/refined mass0 and YOLO .01
  after exact-center dedup: .76618. Candidate count8458/GT3028; still23.38% GT
  unmatched even with perfect selection. Containment coverage is reported
  separately and never treated as instance recall. Radius8 union recall .57232.
- Added RegionPromptHead (small region/text adapters), frozen DINOv3 inside and
  context ROI features plus CLIP crops.23 reviewed training images,12 heldout,
  six seen classes; own GT exemplar boxes positive, other five classes reviewed
  absent.37 heldout exemplar regions classified correctly, but this is an easy
  regional pilot, not whole-pipeline unseen-image or open-world evidence.
- V1 fixed top128 proposal pools on12 heldout images: learned point F1 .54355
  vs unfiltered .54072/CLIP .42144. Unfiltered absent mean58.5, learned0, CLIP0.
  Three heldout mixed scenes: unfiltered R .36697/F1 .40134 vs learned R .17125/
  F1 .28105; wrong-half proposals271 ->0, but many valid detections also removed.
- V2 adds GT/context crops from15 train-only class-pair mosaics. Context crops
  must stay within reviewed source half; no heldout training. Same single-image
  R .49820/F1 .54355; absent mean .01667. Mixed R/F1 unchanged .17125/.28105;
  absent mean .75 ->.25. Not promoted: semantic region recognition does not yet
  preserve candidate recall under mixed-scene/proposal crop distribution shift.
- Logic fixes: random absent labels removed from default negative/mixed sampler;
  verified manifest required, unknowns skipped with counts, mosaic absence
  intersection enforced across all tiles. Wrong-prompt evaluation accepts only
  reviewed per-image mappings and reports sample count; mixed validation uses
  mutually reviewed pools. Legacy prompt/presence evaluators require explicit
  unverified-reproduction flag and label their output invalid as absence GT.
- Fixed arbitrary-grid exemplar mapping to use patch centers; current80->40
  mapping unchanged, noninteger ratios corrected. ViT reload key now includes
  checkpoint revision, preventing stale same-name/same-size backbone reuse.
-52 tests pass; actual dataset smoke confirms reviewed negatives zeroed and
  unreviewed images stay positive. Main trainer real-checkpoint4-image smoke
  completed train/val/save after logic changes; smoke scores not accuracy claims.
- No production verifier integration or new full training launch: mixed-scene
  recall gate failed. Next prerequisite is training on realistic proposal crops
  with spatial/objectness supervision and broader reviewed semantic labels.
- Summary: experiments/semantic_logic_review_20261118.json. Artifacts:
  outputs/proposal_ceiling/dinov3_e10 and outputs/region_prompt_pilot/{v1,v2_mixed_train}.

- Official DINOv3-S/16 access now works. Frozen backbone, same-image official
  positive exemplar boxes; no text model, no trained detection head.
- First run54/58 with4 tiny-exemplar pooling failures. Fixed fractional-area
  box/patch pooling; separate full58 run retained original threshold0.65,
  size640, local maximum radius1. Five decoder tests passed.
- Full58: MAE41.2759, RMSE73.2485, P@16 .18328, R@16 .16513, F1@16 .17373.
  Qualitative previews show background false peaks and crowded-object misses.
- Raw cosine baseline rejected for promotion. Not a comparison of trained
  DINOv3 vs GroundingDINO; prompt inputs differ from prior text-only experiments.
- Results: outputs/dinov3_static/pilot58_area_pool/results.json;
  experiments/dinov3_static_results_20261118.json. No test-set tuning.

- Runner: tools/train_semantic_pilot.py. Two arms from identical exnoise weights,
  seed1811, AdamW lr1e-5, 90 updates, batch2; frozen backbone and BN statistics,
  self-exemplars, existing refined/coarse density and regional count losses.
- Assistant visually reviewed six contact sheets, not exhaustive human labels.
  Excluded ambiguous market image77. Official train images only: 23 training,
  12 image-heldout, six seen classes. Verified absence restricted to these six
  classes. 60 heldout absent prompts and 24 synthetic two-class prompt cases.
  Separate fixed58 validation positives guard retention; no FSC147 test used.
- Positive-only control vs semantic cycle (positive, positive/absent pair,
  two prompts on same mixed image). Matched optimizer steps, not image exposure.
  Two target tests pass: paired image equality, mass preservation, spatial/point
  transforms, negative labels without mutating positives.
- Baseline -> positive control -> semantic: absent mean density75.2923 ->
  61.5571 ->48.1283; all three still produce proposals for100% of absent prompts.
  Mixed wrong-half fraction .42544 ->.44993 ->.41862; mixed point F1
  .42528 ->.43238 ->.41664. Lower wrong-half mass alone is not recognition gain.
- Fixed58 validation density MAE20.8906 ->23.3448 ->28.8221; proposal MAE
  20.3966 ->20.8276 ->21.4828; point F1 .63896 ->.62500 ->.61638.
  Seen-class positive density MAE18.1658 ->15.6683 ->22.9860.
- Semantic checkpoint FAILS predeclared retention guard. No promotion, defaults
  or active checkpoint changed. Evidence suggests suppression/generalization
  loss; this tiny90-step pilot does not prove semantic training cannot work.
- Audit: experiments/semantic_pilot_manifest_20261117.json;
  experiments/semantic_pilot_results_20261117.json. Full per-case outputs,
  source snapshot and both checkpoints: outputs/semantic_training/pilot_v1.

### 2026-11-18: Proposal point head and center-offset pilot

- Added models/proposal_point_head.py and reproducible train/eval tools. Frozen
  best DINOv3+YOLO-World checkpoint SHA256 remains
  05a90c22c05e51590e4e7b0d4b121753f8c306c5379fa09efcc57c093d6a39ef.
- Cached actual coarse/refined maxima plus low-confidence YOLO proposals,
  two-scale DINO ROI features and spatial density/text-map cues. One-to-one
  point assignment, ignored ambiguous centers, duplicate/background labels,
  BCE objectness and center-offset regression. These are localization labels,
  not verified absent-class labels. No explicit text embedding in this head.
- 326 official train images, 1000 updates per variant, seed1811; frozen final
  thresholds. Fixed58 val: baseline MAE20.3966/F1@16 .65649; v1 filter
  MAE34.9828/F1 .52878; offset plus4px suppression MAE29.1034/F1 .56212.
- Follow-up bounded residual initialized at baseline membership, with baseline
  positives prioritized during assignment: MAE29.9310/F1 .57030 including
  offsets. This is a second development experiment after inspecting v1,
  not independent confirmation. Both proposal selection variants rejected.
- V1 point loss train .22060 vs val .74647; residual .21737 vs .44914.
  Generalization gap supports concern about class transfer. Does not prove
  larger training or semantic supervision cannot help. This was a small pilot,
  not full FSC147 training.
- Fixed original proposal set plus v1 center offsets: pilot58 F1 .65649 ->
  .66090; full1286 val F1 .70425 -> .70932, precision .79568 -> .80140,
  recall .63168 -> .63622. Fullval paired MAE23.0047 unchanged by construction.
  Use this paired baseline, not older evaluator MAE23.0210. Fullval includes
  the58 previously inspected images; no independent test claim.
- Reviewed absent-elephant cases still return437/89 points with v1 offset-NMS
  vs baseline432/211. Mixed polka-dot prompt still localizes mostly wrongly.
  Center supervision alone did not solve semantic grounding.
- 60 regression tests pass, including8 assignment/head tests. Existing best
  checkpoint and production defaults retained. No new Kaggle run launched.
- Summary: experiments/proposal_point_head_20261118.json. Per-image results,
  source metadata, checkpoints and fullval evidence: outputs/proposal_point_head/.

### 2026-11-18: Retired failed experimental paths

- User requested cleaning the code and removing unsuccessful experiments.
- Removed standalone DINOv3-only, region/text heads, proposal-point/residual
  heads (including the small offset-only ablation), and semantic fine-tuning
  pilot sources, launchers and tests specific to retired modules.
- Deleted their local weights, feature caches, abandoned standalone training
  smoke weights and stale bytecode:56 files total, 2.054 GiB reclaimed.
  Small evaluation JSON files remain. Earlier commands and deleted weight/source
  paths above are historical records and are no longer runnable instructions.
- Retained main CountYOLO, original baselines, best DINOv3+YOLO-World checkpoint,
  backbone adaptation, CountVid, SAM2/SAM3, general evaluators and verified-absence
  safeguards. Removed only the retired region-head test from the shared semantic
  logic suite; retained its9 checks of the maintained pipeline.
- 49 remaining regression tests pass; compileall and git diff --check pass.
  Protected checkpoint SHA256 hashes verified unchanged after cleanup.
- No Git branches or remote Kaggle jobs/datasets were deleted. Only master and
  sam3 exist locally; sam3 remains an active supported path.
- Exact deleted files, sizes and hashes: experiments/cleanup_20261118.json.

### 2026-11-18: Opt-in SAM3 temporal counting identity

- Implemented video/identity.py: offline stitching of native SAM3 tracklets.
  Keep tracker IDs separate from counting identities. No simultaneous-track
  merges; require3 consecutive endpoint observations, DINO appearance, motion
  for short gaps, boundary exit/entry for long gaps, area consistency and mutual
  candidate margin. Ambiguous cases remain distinct. Gap limits count sampled
  frames. Native within-track identity switches cannot be corrected here.
- Frozen pinned DINOv3-S/16 ROI embeddings via video/identity_features.py.
  Generic features are not instance-ReID training. Optional engine integration
  and evaluator --identity-mode dinov3; default none. CPU feature extraction
  default avoids adding SAM3 GPU memory pressure. Callback IDs remain native;
  stitched counts and mappings are available only at finalization.
- Delayed lifecycle output filters feature rows together with revoked IDs.
  Zero-area thin-mask boxes retain native counts without identity evidence.
- Paired replay of existing aligned_batch8 traces on identical original frames:
  Penguins1,34frames27->27 (GT28); Penguins2,94frames54->54 (GT59).
  No new links, no count MAE improvement demonstrated. No SAM3 rerun performed.
- Controlled3-frame absence and new ID on first3 eligible native tracks per
  video:6/6 injected fragments recovered with no unrelated accepted links.
  Uses real DINO image features and native ID as proxy truth; not natural
  occlusion accuracy. No threshold tuning on these outputs.
- Long-gap boundary reentry and competing-lookalike cases pass logic tests,
  but natural long-term ReID remains unvalidated without labeled footage.
- 50 targeted regression tests pass, including13 identity/engine tests;
  compileall and git diff --check pass. No default switch or remote job launch.
- Summary: experiments/sam3_identity_20261118.json; per-case mappings, feature
  records, hashes and controlled probes: outputs/sam3_identity/.

### 2026-11-18: Submitted full VideoCount SAM3 identity evaluation

- User explicitly requested a Kaggle video evaluation. Submitted one private GPU
  job: nadthien/countyolo-sam3-identity-video-20261118. Initial state QUEUED,
  submitted2026-11-18T11:09:04Z; execution/results not yet confirmed.
- Full13 video/prompt cases and1252 sampled frames: Penguins3/222,
  Crystals7/246, MOT20-Count3/784. Compare native-ID count and stitched identity
  count from one SAM3 pass per case; no retraining or validation threshold search.
- Aligned Ultralytics8.4.204, fp16, object batches8, output/detection confidence.5,
  DINOv3 appearance on CPU to avoid extra SAM3 GPU memory pressure. Fixed configs.
- Attached existing private VideoCount, SAM3 and DINOv3 adaptation-cache datasets.
  Verify local-declared SAM3/DINOv3 hashes and exact frame manifests remotely;
  resolve gated DINOv3 from attached local model files without embedded tokens.
-31 local association/packaging regression tests passed; payload140425 bytes.
  Sources and run parameters recorded in run_manifest.json; incremental results,
  paired_summary.json and traces retained even if another dataset fails.
- Launcher: launchers/kaggle_sam3_identity.py. Submission ledger:
  experiments/sam3_identity_kaggle_20261118.json.

- Follow-up check: first attempt ERROR before video evaluation. All13 identity
  tests passed remotely; parent runner then failed importing video.eval_common
  because chdir('/kaggle/working') did not add that directory to sys.path when
  Kaggle executed /kaggle/src/script.py. Not a model-accuracy failure.
- Fixed explicit parent import path, added a regression executing the actual
  runner prefix from a separate script directory.14 local tests passed.
- Retried only after confirming remote ERROR; same13 cases/settings, new source
  hash d4ab365c7eb5a0745de089765711f1d4197ea468923b318546ee92bd2271239e,
  submitted2026-11-18T11:55:03Z, initial QUEUED. Attempt history retained in ledger;
  first attempt log/status archived under the downloaded output/attempt1 folder.

- User requested trying other Kaggle accounts. After initial automatic-review
  rejection of private input sharing, user explicitly approved read-only access:
  DINOv3 adaptation cache -> thiend27 and ttdan2006; VideoCount -> ttdan2006.
  Verified access after update; datasets remain private, SAM3 access already held.
- Submitted identical payload hash d4ab365c7eb5a0745de089765711f1d4197ea468923b318546ee92bd2271239e
  to thiend27 and ttdan2006, with independent per-account ledgers. Original
  nadthien job retained. No model/config changes between accounts.
- Latest check: thiend27 RUNNING (empty log at check; video inference not yet
  confirmed), ttdan2006 QUEUED, nadthien QUEUED. No accuracy result available.

- Final check: all3 accounts COMPLETE,13/13 video/prompt cases and1252 frames
  each, zero failed cases. Verified model hashes, DINO revision, exact per-case
  frame manifests and identical payload sources/counts across accounts.
- Paired native SAM3 -> identity MAE: Penguins2.3333 ->2.3333; Crystals72 ->72;
  MOT20-Count429 ->429. Zero accepted and zero uncertain candidate links.
  Association did not improve the native counts; not promoted.
- Penguins predictions27/54/39 vs GT28/59/38. Crystals predictions1/1/8/1/4/1/3
  vs GT111/154/27/37/109/77/8. MOT20 predictions70/197/7 vs GT80/278/1203.
  Severe native undercount, particularly MOT20-05, requires detector/prompt/
  numerics/geometry diagnosis; total-count metrics do not identify its cause.
- Small records downloaded for all accounts. Canonical verified results:
  experiments/sam3_identity_kaggle_results_20261118.json. Per-case results and
  paired_summary.json: outputs/kaggle/<account>__countyolo-sam3-identity-video-20261118/.

### 2026-11-18: SAM3 undercount audit in progress

- Raw IDs already scarce: MOT20-05 raw8 -> emitted7; Crystals023 raw1 ->1.
  Association and delayed final filtering cannot explain most of the deficit.
  Detector/tracker load reports contain zero missing parameters/buffers.
- Earlier FP16/FP32 Crystals image probes were both zero with low concept
  presence, but first Crystals023 frame is nearly empty; unique-video GT must
  not be treated as per-frame GT. Need later frames and crowded MOT20 controls.
- Found local Ultralytics8.4.177 differs from Kaggle8.4.204. Prepared isolated
  audit dependency; local initialization remained checkpoint-load-bound and
  produced no accuracy evidence. Stopped local model runs when user explicitly
  requested SAM3 testing on Kaggle only. No pipeline dependency upgrade applied.
- Submitted thiend27/countyolo-sam3-detection-audit-20261118, status RUNNING at
  submission (not yet proof of completed probe). Fixed first/middle/last frames
  of MOT20-05/Crystals023, first MOT20-01/Penguins1 controls; canonical prompts
  plus explicit person/crystal diagnostic alternatives, FP16/FP32,3-frame
  video prefixes, score/presence/mask stats and visual previews. No GT tuning.
- Verify actual image-byte hashes against sampled dataset manifest, matching
  SAM3 checksum and pinned Ultralytics8.4.204. Capture detector threshold counts,
  candidate masklets, native postprocess and removed IDs to localize losses.
- tools/audit_sam3_detection.py; launchers/kaggle_sam3_audit.py;
  experiments/sam3_detection_audit_kaggle_20261118.json. Root cause pending probe.

- Probe COMPLETE:28/28 image runs,14 FP16/FP32 pairs,4 prefix diagnostics.
  All logits/input/masks finite.13 pairs have identical output counts; one
  MOT20-05/person middle-frame case69 vs68. No catastrophic FP16 failure seen.
- MOT20-05/human at first/middle/last frames: image counts0/0/0 in both
  precisions. Person diagnostic alias:49/69/53 FP16,49/68/53 FP32, same.5
  output threshold. Strong prompt sensitivity confirmed, not full-video MAE gain.
  Human raw sigmoid>.5:1/3/4 FP16; none>.7. Existing new-track.7 further limits
  births after initialization. No thresholds or canonical prompts changed.
- Crystals023/white crystal presence first/middle/last:.06199/.01628/.00982;
  image counts0/0/0. Crystal alias likewise zero. Low contrast/domain recognition
  remains unresolved; presence indicates model confidence, not true absence.
- Prefix MOT20-05 detector counts1/0/2, native output1/1/1; Crystals023
  detector1/2/1, native1/1/1. Scarcity precedes final identity stitching.
- Confirmed implementation difference: Ultralytics image postprocess multiplies
  raw class sigmoid by presence; video extractor does not. Port detector uses
  supervise_joint_box_scores=False, while audited Meta video builder enables it.
  This can expose false positives in low-presence videos; adding presence would
  suppress more and is not an established remedy for undercount.
- Kaggle verified actual bytes of all521 frames in the4 selected clips. Local
  CPU-only FFV1 roundtrip on9 sampled frames gives exact equality, zero pixel
  difference. No local SAM3 model inference used for these conclusions.
- Summary: experiments/sam3_detection_audit_results_20261118.json. Probe artifacts:
  outputs/kaggle/thiend27__countyolo-sam3-detection-audit-20261118/audit/.

### 2026-11-19: Submitted detector-only FSC147 and VID evaluation

- User requested one Kaggle job evaluating SAM3 detector alone on FSC147 and
  video data. Submitted nadthien/countyolo-sam3-detector-fsc-vid-20261119,
  initial QUEUED,2026-11-19T02:05:52Z (09:05:52 Asia/Saigon).
- SAM3SemanticPredictor image detector/segmentation via Ultralytics8.4.204,
  fixedconf.5, FP16, input1008, max_grid1. No tracker, temporal association,
  tiling, fine-tuning, visual exemplars, threshold selection or FSC test access.
- Entire FSC147 val1286 images: count MAE/RMSE/bias and one-to-one box-center
  point precision/recall/F1 at8/16 pixels in640-space. Not bbox AP.
- All1252 exact sampled VideoCount frames, per-frame GT. One missing annotation:
  Crystals ma2035_039/ma2035_03900364.jpg; predict but exclude from errors, never
  assign zero.1251 canonical frame labels. Separate person_alias arm on all784
  MOT20 frames, retaining canonical human.3322 independent image/prompt calls.
- Report per-video frame errors, dataset frame-weighted MAE and video-macro MAE.
  No unique-video counts or summed per-frame detections presented as unique count.
- Input preflight checks SAM3/FSC metadata/VID GT hashes and all1252 JPEG hashes.
  Audited detector checkpoint loading. Incremental JSONL, periodic results and
  final detector/results.json retained. Model inference only on Kaggle GPU.
- Six CPU-only protocol tests passed, including missing GT, point matching,
  partial failure and fake-counter integration that refuses tracker calls.
- Launcher: launchers/kaggle_sam3_detector_only.py; evaluator:
  tools/eval_sam3_detector_only.py; ledger:
  experiments/sam3_detector_only_kaggle_20261119.json. No results yet.

### 2026-11-19: YOLO branch diagnosis and bounded improvement pilots

- User authorized improving/testing the YOLO branch while leaving SAM3 jobs
  untouched. Retained best DINOv3+YOLO checkpoint SHA256
  05a90c22c05e51590e4e7b0d4b121753f8c306c5379fa09efcc57c093d6a39ef;
  hash verified unchanged afterward. No application/default/checkpoint changes.
- Reused prior58-image evidence:31 images have no YOLO boxes atconf.05;
  YOLOconf.01 point recall@16 .18197; refined baseline .58917; wide candidate
  oracle .76618. Fixed-pool semantic filtering cannot recover missing centers.
- Tested contextual2x2 YOLO crops, unique tile ownership and one-to-one support
  for coarse/refine density candidates. Kept every original proposal unchanged;
  no CLIP gate, new learned head, GT inference input or threshold search.
  Baseline MAE20.3966/F1 .65649 -> recovery20.3448/.65601. Added7 proposals,
  only1 additional true positive. Rejected for retention; no fullval promotion.
- Detector-only control S-v1 vs S-v2 vs L-v2, same58 images and prompt/CLIP
  precision setup. Atconf.01 recall .18197/.17734/.20079; F1 .29560/.27954/.31068.
  Larger detector gives limited recall gain; not enough evidence to replace
  the trained density pipeline. New weights kept separate in experiment output.
- Official exemplar-center diagnostic (annotation input, not deployable):
  baseline MAE20.3966 ->18.5345, precision .74117 ->.81781, recall .58917 ->
  .58851, F1 .65649 ->.68446. Location selection contributes to false positives;
  this is not evidence of a text-only deployment improvement or perfect oracle.
- Tested fixed GT-free detector-supported exemplar policy,conf.15, unique
  coarse peak within16px, maximum3 exemplars, self fallback otherwise. Activated
  on12/58 images; MAE20.8966/F1 .65208, recall .58058. Rejected. Do not use
  detector confidence alone as a reliable exemplar selector.
- No training, new full1286 validation or video test launched: deployable
  candidates failed pilot retention.58 images were already used in development,
  not an independent test. Further training needs labeled region/text alignment
  for exemplar selection, not unverified whole-image absence labels.
- Summary: experiments/yolo_improvement_pilot_20261119.json. Results:
  outputs/yolo_recovery/{pilot_20261119,exemplar_oracle_20261119,
  exemplar_policy_20261119}/ and outputs/yolo_detector_compare/20261119/.

### 2026-11-19: Learned region/text exemplar ranker submitted

- User selected improving prompt-conditioned region exemplar selection. Added
  a separate ExemplarRanker: frozen CLIP region base plus norm-bounded DINOv3
  adapter into CLIP space, explicit text similarity and4-cue quality residual.
  Zero-initialized output preserves exact self fallback; unchanged rank order
  also falls back. No learned text-presence veto or final detection filtering.
- Train3,647 official FSC train images,89 classes; exclude12 reviewed images
  from new selector training. Base counter had already seen official train, so
  image-heldout status applies only to this selector.30 verified cross-class
  mosaics, both prompts, using mutual reviewed absence; all35 reviewed image
  hashes checked locally and again in training.
- Rank real coarse-peak candidate locations (budget128, original baseline slots
  retained). Candidate center<=8px from target GT is positive; >24px negative
  localization anchor; intervening candidates ignored. These are not absent
  whole-image labels. Positive candidate class CE uses train classes only.
- AdamW lr3e-4, fixed2,000updates,7 class-balanced single images +1 mixed per
  batch when available. Ranking loss +.1 class CE +.01 residual penalty. Train
  only selector; YOLO/DINO/counting heads/matchers and CLIP remain frozen.
- Final-only checkpoint; pilot58 must show improvement with MAE/F1/recall/density
  retention; reviewed mixed and absence guards must also pass before full1286
  validation. No test-set use and no automatic production promotion.
-17 local CPU tests passed: ranker/conditioning, semantic safeguards and complete
  materialized Kaggle module import. No pretrained model test needed locally.
- Submitted nadthien/countyolo-exemplar-region-text-20261119, initial QUEUED,
  2026-11-19T02:55:55Z (09:55:55 Asia/Saigon). No accuracy result yet.
  Payload SHA2560259579667bc2c36c98ca08b354d6e88af0f8164bc941e244b0ed1667bd6cf33.
- Reuse private DINO cache and original e10 kernel output on the same account;
  checkpoint hash verified05a90c...d6a39ef. Public YOLO asset bytes verified
  identical to local095f5266...ee251cf. No new private-data readers and no SAM3
  job changes. Ledger: experiments/exemplar_ranker_kaggle_20261119.json.

- Ranker final check COMPLETE:3,647 cached single images,60 mixed records,
  94,104 positive anchors, fixed2,000 updates. Verified payload sources and
  frozen counting checkpoint against submission manifest.
- Pilot58 paired Kaggle baseline -> ranked: MAE20.3966 ->19.6034;
  F1 .65636 ->.66667; recall .58884 ->.58620; density MAE23.4774 ->23.5412.
  Mixed6 prompts: MAE89.1667 ->91.5, F1 .54954 ->.53773,
  recall .59786 ->.56116. Reviewed absent2: mean false proposals321.5 ->337.
  Mixed and absence guards FAIL. No full1286 evaluation or promotion.
- Downloaded ranker_final.pth536596 bytes, SHA256
  7d50816d0b813b71ae99a00dd8dc6e77fcf168113f9fe4941dbd9737408e0175.
  Existing baseline/default remains unchanged. Canonical summary:
  experiments/exemplar_ranker_results_20261119.json.

- Concurrent detector-only SAM3 job ERROR before evaluation: constructed
  mot20-frame-level-counts-gt.json instead of upstream MOT20-frame-level-counts-gt.json.
  Case-insensitive Windows hid this path bug; Linux correctly rejected it.
  Added shared case-preserving frame_count_annotation() and filename regression.
 20 protocol/identity tests passed; retried only after verifying remote ERROR.
  Same evaluation configuration, new payload
  afbb5bb0a62dd0eafc00a127c49f00513e8b24f84bb9ba21416a7525cccb4b32,
  submitted2026-11-19T04:22:49Z (11:22:49 Asia/Saigon), initial QUEUED.
  Initial error logs archived under downloaded output/attempt1.


## 2026-11-22 — Archived Best Open-World Counting Checkpoints (LLMDet v3 & OV-DQUO v1) + OV-DQUO v3.1

- **LLMDet v3 Text-Only (`ttdan2006/countyolo-llmdet-text-v3-20261121`, commit `45b75d5`):**
  - Full-Val (1,190 images): `fullval_tiled @0.26 MAE = 10.42, F1 = 0.818`, `fullval_raw @0.26 MAE = 10.94, F1 = 0.810`, `absent = 0.00`.
  - Archived checkpoint: `checkpoints/llmdet_v3_text_fullval10.42.pth` (143.5 MB, SHA256 `c25bb7b034d2b202`), result: `experiments/llmdet_v3_text_result_20261121.json`.
- **OV-DQUO v1 Multimodal (`nadthien/countyolo-ov-dquo-mm-v1-20261122`, `EVA02-CLIP-B/16 @768`, commit `592a8d1`, `best_epoch=4`):**
  - Full-Val (1,190 images): `fullval_mm_tiled @0.35 MAE = 13.32, F1 = 0.802`, `fullval_mm_raw @0.26 MAE = 13.77, F1 = 0.793`, `fullval_ex_only_raw @0.35 MAE = 14.29, F1 = 0.790`, `fullval_text_raw @0.26 MAE = 14.56, F1 = 0.786`, `absent @0.26 = 0.00`.
  - Archived checkpoint: `checkpoints/ov_dquo_v1_vitb16_mm_ep4_fullval13.32.pth` (85.4 MB, SHA256 `cf9b85cd9685302c`), results: `experiments/ov_dquo_v1_vitb16_mm_result_20261122.json`, manifest: `experiments/ov_dquo_v1_vitb16_mm_manifest_20261122.json`.
- **OV-DQUO v3.1 (`ttdan2006/countyolo-ov-dquo-mm-v3-20261122`, `EVA02-CLIP-L/14 @672`):**
  - Fixed v3 Epoch 1–2 dynamics (5% absent with `normalizer=30.0`, embedding-space stride-8 text residual replacing `softplus(-3.0)`, active-slope `1.5*tanh` visual gate + elementwise `clip_grad_value_(cal_p, 1.0)`, removed muted `mid_mem_fusion`, `best_th_mae` checkpoint selection); 6/6 CPU tests PASS; submitted on `ttdan2006`.
- Index file: `experiments/BEST_MODELS.md`.
