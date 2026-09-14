import os
import torch
import pytest
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.language.clip_encoder import CLIPTextEncoder
from models.count_yolo import CountYOLO

def test_clip_encoder_shape():
    """Verify that CLIPTextEncoder outputs correctly shaped tensors."""
    encoder = CLIPTextEncoder(out_channels=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    
    texts = ["birds", "many people in a crowd"]
    feats = encoder(texts, device)
    
    assert feats.shape == (2, 256)
    assert feats.requires_grad # The projection layer should require grad

def test_countyolo_clip_forward():
    """Verify that CountYOLO handles string inputs properly."""
    model = CountYOLO(use_clip=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    images = torch.rand((2, 3, 384, 384)).to(device)
    texts = ["objects", "cars"]
    
    # Text input only, attr_feats is implicitly generated as dummy
    outputs = model(images, texts=texts)
    
    assert 'density_map' in outputs
    assert 'cls_scores' in outputs
    assert 'box_preds' in outputs
    assert outputs['density_map'].shape == (2, 1, 384, 384)
