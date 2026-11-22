# Best Saved Counting Models (FSC-147 Full-Val — 1,190 images, 29 unseen classes)

All checkpoints below are archived locally in `checkpoints/` with verified SHA-256 hashes, exact Kaggle kernel references, exact Git commits, and full validation result JSONs in `experiments/`.

| Rank / Family | Model | Local Checkpoint | SHA-256 (prefix) | Full-Val Best MAE | F1 | Absent MAE | Git Commit & Branch | Kaggle Ref |
|---|---|---|---|---:|---:|---:|---|---|
| **Best Overall (DINO family)** | **LLMDet v3 Text-Only** (`llmdet_tiny`, 2×2 tiled `@0.26`) | `checkpoints/llmdet_v3_text_fullval10.42.pth` (143.5 MB) | `c25bb7b034d2b202` | **10.42** (`10.94` raw) | **0.818** (`0.810` raw) | **0.00** | `45b75d5` (`experiment/llmdet-countgd-adapt`) | `ttdan2006/countyolo-llmdet-text-v3-20261121` |
| **Best Non-DINO (Multimodal + Text + Ex-Only)** | **OV-DQUO v1** (`EVA02-CLIP-B/16` @768, `ep4`) | `checkpoints/ov_dquo_v1_vitb16_mm_ep4_fullval13.32.pth` (85.4 MB) | `cf9b85cd9685302c` | **13.32** MM tiled `@0.35` / **13.77** MM raw `@0.26` / **14.29** Ex-Only `@0.35` / **14.56** Text-Only `@0.26` | **0.802** MM tiled / **0.793** MM raw / **0.786** Text | **0.00** (`@0.26`) | `592a8d1` (`experiment/ov-dquo-counting`) | `nadthien/countyolo-ov-dquo-mm-v1-20261122` |

## Archived Result Files
- **LLMDet v3 Text-Only:** `experiments/llmdet_v3_text_result_20261121.json` (full output folder: `outputs/kaggle/ttdan2006__countyolo-llmdet-text-v3-20261121/`)
- **OV-DQUO v1 Multimodal (`ViT-B/16`):**
  - `experiments/ov_dquo_v1_vitb16_mm_result_20261122.json`
  - `experiments/ov_dquo_v1_vitb16_mm_manifest_20261122.json`
  - Full output folder (includes `ovdquo_vitb16_lvis.pth`, `history.json`, `last.pth`): `outputs/kaggle/nadthien__countyolo-ov-dquo-mm-v1-20261122/`

## Active Follow-up Runs (ViT-L/14)
- `thiend27/countyolo-ov-dquo-mm-v2-20261122` (OV-DQUO v2 `ViT-L/14` @672)
- `ttdan2006/countyolo-ov-dquo-mm-v3-20261122` (OV-DQUO v3.1 `ViT-L/14` @672, calibrated 5% absent `norm=30`, embedding-space stride-8 text residual, active-slope visual gate, elementwise scalar clip, `best_th_mae` selection)
