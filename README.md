# CountYOLO v4

Mô hình đếm vật thể thế hệ mới kết hợp **CLIP Vision-Language** + **ResNet-50 Backbone** + **SAM 2.1 Video Tracking**, hỗ trợ đếm zero-shot thông qua exemplar crops hoặc text prompt từ LLM.

## Kiến trúc

```
Image → ResNet-50 Backbone → RepVL-PAN Neck ← CLIPTextEncoder (category text)
                                    ↓
                          AttributeFusion ← ExemplarEncoder (exemplar crops)
                             [hoặc CLIPTextEncoder(LLM attributes) lúc infer]
                                    ↓
                    DensityHead + NMSFreeHead
                                    ↓
                           SAM 2.1 Video Tracking (inference)
```

## Cài đặt

```bash
pip install torch torchvision transformers open-clip-torch
pip install git+https://github.com/facebookresearch/sam2.git
pip install hydra-core iopath wandb tqdm scipy pyyaml
```

## Cấu trúc thư mục

```
countyolo/
├── configs/               # YAML configs
├── datasets/              # FSC-147 Dataset loader
├── models/
│   ├── backbone/          # ResNet-50 Backbone
│   ├── neck/              # RepVL-PAN Neck
│   ├── fusion/            # AttributeFusion (Cross-Attention)
│   ├── heads/             # DensityHead + NMSFreeHead
│   └── language/          # CLIPTextEncoder + ExemplarEncoder
├── utils/                 # Loss, Matcher, Router, SAT
├── tests/                 # Unit tests + Integration tests
├── train.py               # Script training chính
└── count_in_videos_countyolo.py  # Inference + SAM 2.1 tracking
```

## Dữ liệu (Google Drive)

Dữ liệu **không được lưu trong repo này** (quá lớn). Tải về và đặt vào:

```
data/
├── FSC147/
│   ├── images_384_VarV2/
│   ├── annotation_FSC147_384.json
│   └── Train_Test_Val_FSC_147.json
└── VideoCount/
    └── MOT20-Count/
        ├── frames/MOT20-01/
        └── anno/MOT20-frame-level-counts-gt.json
```

## Training (Google Colab / GPU)

```bash
# Kết nối Google Drive (Colab)
from google.colab import drive
drive.mount('/content/drive')

# Copy data
!cp -r /content/drive/MyDrive/countyolo_data/FSC147 data/

# Chạy training
python train.py --config configs/countyolo_fsc147.yaml
```

## Inference với LLM

```python
model = CountYOLO(use_clip=True)

# TRAIN: exemplar crops từ FSC-147
output = model(images, texts=["person"], exemplar_crops=crops)

# INFER: LLM parsed attributes (không cần exemplar ảnh)
output = model(images, texts=["bike"], attr_texts=["red", "parked"])
count = output["density_map"].sum().item()
```

## Tests

```bash
python -m pytest tests/
python tests/test_llm_inference.py
python train.py --sanity-check
```
