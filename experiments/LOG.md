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
