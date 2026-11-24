# Best Saved Counting Models (Flagship: OV-DQUO v2 ViT-L/14)

All checkpoints below are permanently archived in `checkpoints/` with verified SHA-256 hashes, Kaggle kernel references, and full validation/video JSONs in `experiments/`.

Active repository branches retained: **OV-DQUO (`v2` flagship)**, **YOLO-World / CountYOLO**, **CountVid**, and **SAM 3**.

---

## 1. FSC-147 Full-Val (1,190 images, 29 unseen classes)

| Role | Model | Local Checkpoint | SHA-256 (prefix) | Full-Val Best MAE | F1 | Absent MAE | Kaggle Ref |
|---|---|---|---|---:|---:|---:|---|
| **PRIMARY FLAGSHIP (Non-DINO)** | **OV-DQUO v2** (`EVA02-CLIP-L/14` @672, `ep12`) | `checkpoints/ov_dquo_v2_vitl14_mm_ep12_fullval10.85.pth` (95.3 MB) | `1bb9af9066056d07` | **10.85** MM tiled `@0.35` / **11.72** MM raw / **12.05** Ex-Only / **13.52** Text `@0.26` | **0.830** MM / **0.806** Text | **0.50** (`@0.35`) | `thiend27/countyolo-ov-dquo-mm-v2-20261122` |
| **Low-Threshold Absent Variant** | **OV-DQUO v3.1** (`EVA02-CLIP-L/14` @672, `ep8`) | `checkpoints/ov_dquo_v31_vitl14_mm_ep8_fullval11.08.pth` (98.9 MB) | `archived` | **11.08** MM tiled `@0.35` / **12.26** MM raw / **12.93** Ex-Only / **13.79** Text `@0.26` | **0.824** MM / **0.798** Text | **0.50** (`@0.35`), **3.50** (`@0.26`) | `ttdan2006/countyolo-ov-dquo-mm-v3-20261122` |
| **Lightweight Non-DINO** | **OV-DQUO v1** (`EVA02-CLIP-B/16` @768, `ep4`) | `checkpoints/ov_dquo_v1_vitb16_mm_ep4_fullval13.32.pth` (85.4 MB) | `cf9b85cd9685302c` | **13.32** MM tiled `@0.35` / **13.77** MM raw / **14.29** Ex-Only / **14.56** Text `@0.26` | **0.802** MM / **0.786** Text | **0.00** (`@0.26`) | `nadthien/countyolo-ov-dquo-mm-v1-20261122` |
| **Archived Reference (DINO)** | **LLMDet v3 Text-Only** (`llmdet_tiny`, 2×2 tiled `@0.26`) | `checkpoints/llmdet_v3_text_fullval10.42.pth` (143.5 MB) | `c25bb7b034d2b202` | **10.42** (`10.94` raw) | **0.818** (`0.810` raw) | **0.00** | `ttdan2006/countyolo-llmdet-text-v3-20261121` |

---

## 2. VideoCount Benchmark (11 shared videos: 3 Penguins + 7 Crystals + MOT20-01)

| Pipeline | Penguins (3) MAE | Crystals (7) MAE | MOT20-01 (1) MAE | **Overall 11-Video MAE** | Delta vs CountVid |
|---|---:|---:|---:|---:|---|
| **OV-DQUO v2 + SAM 2.1 Large** *(Flagship)* | **6.33** | **37.71** | 13.00 (`93/80`) | **26.91** | **-19.64 MAE (-42.2%)** |
| **OV-DQUO v3.1 + SAM 2.1 Large** | 9.00 | 43.29 | **8.00** (`88/80`) | **30.73** | **-15.82 MAE (-34.0%)** |
| **OV-DQUO v1 + SAM 2.1 Large** | 7.33 | 53.00 | 12.00 (`68/80`) | 36.82 | -9.73 MAE (-20.9%) |
| **SAM 3 native** | 20.67 | 48.86 | 12.00 (`92/80`) | 37.82 | -8.73 MAE |
| **CountVid** (`CountGD-Box + SAM 2.1`) | 3.33 | 69.14 | 18.00 (`98/80`) | 46.55 | *(baseline)* |
| **CountYOLO** (`countyolo_old`) | 2.33 | 74.71 | 4.00 (`76/80`) | 48.55 | +2.00 MAE |

---

## 3. Canonical Inference & Evaluation Entry Points
- **OV-DQUO (v2 flagship, auto-loads v1 / v2 / v3.1 strictly):**
  - Detector loader: `models/ov_dquo_video_detector.py` (`load_ov_dquo_video_detector`)
  - 4-stage Video + SAM 2.1 + Tracking + TemporalIdentity: `video/ov_dquo_sam2_pipeline.py`
  - VideoCount CLI evaluator: `tools/eval_ov_dquo_video.py`
  - Kaggle launcher (default `v2`): `launchers/kaggle_ov_dquo_video.py`
  - Archived VideoCount summaries: `experiments/ov_dquo_v2_sam2_video_result_20261124.json`, `experiments/ov_dquo_v31_sam2_video_result_20261124.json`, `experiments/ov_dquo_v1_sam2_video_result_20261123.json`
