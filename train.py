import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from datasets.fsc147_dataset import FSC147Dataset
from models.count_yolo import CountYOLO
from utils.matcher import DensityGuidedMatcher
from utils.loss import CountYOLOLoss, UncertaintyLoss


def extract_exemplar_crops(images: torch.Tensor, boxes_list: list, crop_size: int = 224) -> torch.Tensor:
    """
    Cắt và resize các exemplar crops từ ảnh gốc dựa trên exemplar bounding boxes.
    
    Args:
        images:     (B, 3, H, W) - Ảnh gốc
        boxes_list: list[(N_ex, 4)] - exemplar boxes [x1, y1, x2, y2] cho mỗi ảnh
        crop_size:  Kích thước resize (mặc định 224 cho CLIP)
    Returns:
        (B, 3, 3, crop_size, crop_size) - Tối đa 3 exemplar crops mỗi ảnh
    """
    B = images.size(0)
    all_crops = []
    
    for b in range(B):
        img = images[b]  # (3, H, W)
        boxes = boxes_list[b]  # (N_ex, 4)
        crops = []
        
        for box in boxes[:3]:  # Lấy tối đa 3 exemplar
            x1 = max(0, int(box[0].item()))
            y1 = max(0, int(box[1].item()))
            x2 = min(img.shape[2], int(box[2].item()))
            y2 = min(img.shape[1], int(box[3].item()))
            
            if x2 > x1 and y2 > y1:
                crop = img[:, y1:y2, x1:x2].unsqueeze(0)  # (1, 3, h, w)
                crop_resized = F.interpolate(crop, size=(crop_size, crop_size),
                                             mode='bilinear', align_corners=False).squeeze(0)
            else:
                crop_resized = torch.zeros(3, crop_size, crop_size, device=img.device)
            crops.append(crop_resized)
        
        # Pad về 3 nếu ít hơn 3 exemplars
        while len(crops) < 3:
            crops.append(torch.zeros(3, crop_size, crop_size, device=img.device))
        
        all_crops.append(torch.stack(crops))  # (3, 3, H, W)
    
    return torch.stack(all_crops)  # (B, 3, 3, H, W)

def collate_fn(batch):
    # Trích xuất và pad batch
    images = [item['image'] for item in batch]
    density_maps = [item['density_map'] for item in batch]
    points = [item['points'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    image_names = [item['image_name'] for item in batch]
    
    # Pad width to max width in the batch and ensure it's a multiple of 32
    max_w = max(img.shape[2] for img in images)
    if max_w % 32 != 0:
        max_w = max_w + (32 - (max_w % 32))
    
    padded_images = []
    padded_density_maps = []
    
    for img, dm in zip(images, density_maps):
        pad_w = max_w - img.shape[2]
        # pad format for F.pad is (pad_left, pad_right, pad_top, pad_bottom)
        img_padded = torch.nn.functional.pad(img, (0, pad_w, 0, 0))
        dm_padded = torch.nn.functional.pad(dm, (0, pad_w, 0, 0))
        padded_images.append(img_padded)
        padded_density_maps.append(dm_padded)
        
    return torch.stack(padded_images), torch.stack(padded_density_maps), points, boxes, image_names

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/countyolo_fsc147.yaml')
    parser.add_argument('--sanity-check', action='store_true', help='Overfit on a single mini-batch')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = FSC147Dataset(data_dir=cfg['dataset']['data_dir'], split='train', sigma=cfg['dataset']['sigma'])
    dataloader = DataLoader(dataset, batch_size=cfg['dataset']['batch_size'], shuffle=True, 
                            num_workers=0 if args.sanity_check else cfg['dataset']['num_workers'], 
                            collate_fn=collate_fn)

    # Models
    model = CountYOLO(use_clip=True).to(device)
    
    # Fine-Tuning Setup
    fine_tune_cfg = cfg.get('fine_tune', {})
    freeze_clip = fine_tune_cfg.get('freeze_clip', False)
    freeze_backbone = fine_tune_cfg.get('freeze_backbone', False)
    use_amp = fine_tune_cfg.get('use_amp', False)
    grad_accum_steps = fine_tune_cfg.get('gradient_accumulation_steps', 1)
    
    if freeze_clip or freeze_backbone:
        model.freeze_modules(freeze_clip=freeze_clip, freeze_backbone=freeze_backbone)
        print(f"Fine-Tuning Mode: Frozen CLIP={freeze_clip}, Frozen Backbone={freeze_backbone}")
        
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
    matcher = DensityGuidedMatcher(lambda_dens_max=cfg['matcher']['lambda_dens_max'], 
                                   warmup_epochs=cfg['matcher']['warmup_epochs']).to(device)
    
    # Loss & Optimizer
    base_criterion = CountYOLOLoss(w_density=cfg['loss']['w_density'], 
                                   w_cls=cfg['loss']['w_cls'], 
                                   w_box=cfg['loss']['w_box']).to(device)
                                   
    use_uncertainty = cfg['loss'].get('use_uncertainty', False)
    # Lọc ra các parameter không bị đóng băng để huấn luyện
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if use_uncertainty:
        uncertainty_loss = UncertaintyLoss(num_losses=3).to(device)
        optimizer = optim.Adam(trainable_params + list(uncertainty_loss.parameters()), lr=cfg['training']['lr'])
    else:
        uncertainty_loss = None
        optimizer = optim.Adam(trainable_params, lr=cfg['training']['lr'])

    # Sanity Check setup
    epochs = 1 if args.sanity_check else cfg['training']['epochs']
    iterations = 200 if args.sanity_check else len(dataloader)
    
    # WandB setup
    if not args.sanity_check:
        if os.environ.get("WANDB_API_KEY"):
            wandb.init(project=cfg['wandb']['project'], name=cfg['wandb']['name'], config=cfg)
        else:
            print("⚠️ Không tìm thấy WANDB_API_KEY. Tắt đồng bộ Weights & Biases để tránh bị treo.")
            os.environ["WANDB_MODE"] = "disabled"
            wandb.init(mode="disabled")

    # Fetch 1 batch for sanity check
    if args.sanity_check:
        print("SANITY CHECK MODE: Overfitting on 1 mini-batch...")
        sanity_batch = next(iter(dataloader))
    
    # Khởi tạo data_iter TRƯỚC vòng for epoch → sửa lỗi NameError khi it=0
    data_iter = iter(dataloader)
    
    best_loss = float('inf')
        
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(range(iterations), desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        ema_loss = None   # Exponential Moving Average de progress bar muot hon
        ema_alpha = 0.98  # Momentum: 0.98 = smooth manh, 0.9 = nhanh hon
        
        for it in pbar:
            if args.sanity_check:
                images, density_targets, points, boxes, image_names = sanity_batch
            else:
                try:
                    images, density_targets, points, boxes, image_names = next(data_iter)
                except:
                    data_iter = iter(dataloader)
                    images, density_targets, points, boxes, image_names = next(data_iter)
                    
            images = images.to(device)
            density_targets = density_targets.to(device)
            points = [p.to(device) for p in points]
            boxes = [b.to(device) for b in boxes]
            B = images.size(0)
            
            # Trích xuất exemplar crops từ ảnh gốc (thay thế random noise trước đây)
            # TRAIN: ExemplarEncoder(crop) → đúng cùng CLIP space với CLIPTextEncoder(text) lúc infer LLM
            exemplar_crops = extract_exemplar_crops(images, boxes)  # (B, 3, 3, 224, 224)
            
            # Text prompt: lấy từ metadata FSC-147 (tạm thời dùng category chung)
            texts = ["objects"] * B  # TODO: thay bằng category thật từ FSC-147 metadata khi có
            
            # Gradient Accumulation: chỉ zero_grad ở đầu mỗi accumulation window
            if it % grad_accum_steps == 0:
                optimizer.zero_grad()
            
            # Forward và Loss phải nằm gọn trong autocast context
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Truyền exemplar_crops vào model: CountYOLO sẽ tự gọi ExemplarEncoder
                outputs = model(images, texts=texts, exemplar_crops=exemplar_crops)
                pred_density = outputs['density_map']
                pred_cls = outputs['cls_scores']
                pred_boxes = outputs['box_preds']
                
                from utils.box_utils import decode_fcos_boxes
                
                # Decode boxes từ (l,t,r,b) sang tọa độ ảnh gốc tuyệt đối
                pred_boxes_scaled = decode_fcos_boxes(pred_boxes, stride=4.0) # (B, N_preds, 4)
                # Sau khi fix decode_fcos_boxes dùng exp(), x2 > x1 và y2 > y1 luôn đúng.
                # Clamp về 0 vẫn giữ lại để an toàn khi box vượt biên ảnh.
                pred_boxes_scaled = pred_boxes_scaled.clamp(min=0)
                N_preds = pred_boxes_scaled.size(1)
                
                # Xây dựng base cost dựa trên khoảng cách L1
                matcher_cost_matrix = []
                for b in range(B):
                    N_gt = len(points[b])
                    if N_gt == 0:
                        matcher_cost_matrix.append(torch.zeros((N_preds, 0), device=device))
                        continue
                    
                    # Tính tâm của Bounding Box từ tọa độ ảnh gốc
                    pred_cx = (pred_boxes_scaled[b, :, 0] + pred_boxes_scaled[b, :, 2]) / 2.0
                    pred_cy = (pred_boxes_scaled[b, :, 1] + pred_boxes_scaled[b, :, 3]) / 2.0
                    pred_ctrs = torch.stack([pred_cx, pred_cy], dim=1) # (N_preds, 2)
                    
                    gt_pts_xy = points[b][:, [1, 0]]
                    
                    # Convert sang float32 vì torch.cdist CUDA không hỗ trợ FP16 (Half)
                    cost_dist = torch.cdist(pred_ctrs.float(), gt_pts_xy.float(), p=1.0)
                    matcher_cost_matrix.append(cost_dist)
                
                cost_dens = matcher.compute_cost_dens(density_targets, pred_boxes_scaled) # (B, N_preds, 1)
                lambda_dens = matcher.get_lambda_dens(epoch)
                
                final_cost_list = []
                for b in range(B):
                    if matcher_cost_matrix[b].shape[1] > 0:
                        final_cost = matcher_cost_matrix[b] + lambda_dens * cost_dens[b]
                    else:
                        final_cost = matcher_cost_matrix[b]
                    final_cost_list.append(final_cost)
                    
                # Tính Loss
                loss_dict = base_criterion(pred_density, pred_cls, pred_boxes_scaled,
                                           density_targets, points, boxes, final_cost_list)
                
                if use_uncertainty:
                    losses_to_balance = [loss_dict['loss_density'], loss_dict['loss_cls'], loss_dict['loss_box']]
                    total_loss = uncertainty_loss(losses_to_balance)
                    loss_dict['total_loss'] = total_loss
                    loss_dict['sigma_dens'] = torch.exp(uncertainty_loss.log_vars[0]).item()
                    loss_dict['sigma_cls']  = torch.exp(uncertainty_loss.log_vars[1]).item()
                    loss_dict['sigma_box']  = torch.exp(uncertainty_loss.log_vars[2]).item()
                else:
                    total_loss = loss_dict['total_loss']
                    
                # Scale loss theo gradient accumulation steps
                total_loss = total_loss / grad_accum_steps
                
            # Backward với AMP Scaler
            scaler.scale(total_loss).backward()
            
            if (it + 1) % grad_accum_steps == 0 or (it + 1) == iterations:
                scaler.step(optimizer)
                scaler.update()
            
            # Logging
            log_stats = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
            pbar.set_postfix({'loss': f"{log_stats['total_loss']:.4f}", 
                              'den': f"{log_stats['loss_density']:.4f}",
                              'cnt': f"{log_stats.get('loss_count', 0):.4f}",
                              'cls': f"{log_stats.get('loss_cls', 0):.4f}"})
            
            if not args.sanity_check and wandb.run is not None:
                wandb.log(log_stats)
                
            # epoch_loss dung unscaled loss (log_stats['total_loss']) de nhat quan
            # voi gia tri hien thi tren progress bar.
            # KHONG dung total_loss.item() vi no da bi chia grad_accum_steps.
            step_loss = log_stats['total_loss']
            epoch_loss += step_loss
            
            # EMA loss de hien thi running average thay vi loss cua buoc hien tai
            if ema_loss is None:
                ema_loss = step_loss
            else:
                ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * step_loss
            pbar.set_description(f"Epoch {epoch+1}/{epochs} | avg={ema_loss:.3f}")
                
        # End epoch - Save Checkpoints
        if not args.sanity_check:
            epoch_loss /= iterations   # epoch_loss bay gio la trung binh UNSCALED loss
            # Luu model tot nhat
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), "checkpoint_best.pth")
                print(f"[Epoch {epoch+1}] Luu checkpoint tot nhat (avg_loss={best_loss:.4f}, ema_loss={ema_loss:.4f})")
            else:
                print(f"[Epoch {epoch+1}] avg_loss={epoch_loss:.4f}, ema_loss={ema_loss:.4f} (best={best_loss:.4f})")
            
            # Lưu model mới nhất
            torch.save(model.state_dict(), "checkpoint_latest.pth")
            
    print("Training finished!")

if __name__ == '__main__':
    train()
