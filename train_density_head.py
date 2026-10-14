import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.density_yolo import DensityYOLOWorld, DENSITY_SCALE
from datasets.fsc147_dataset import FSC147Dataset, collate_fsc147


REGION_SCALES = (4, 8, 16, 80)  # cạnh vùng (ô lưới P3); 80 = toàn ảnh 640


def region_count_loss(pred, gt, scales=REGION_SCALES):
    """L1 của số đếm trong từng vùng s x s (sum-pool), chia cho s để các tỷ lệ cùng độ lớn.
    Ép đúng số lượng theo vùng thay vì ép từng pixel khớp hình Gaussian của GT."""
    diff = pred - gt
    return sum((F.avg_pool2d(diff, s) * s * s).abs().mean() / s for s in scales) / len(scales)


def scale_loss(log_size, gt_log_size, gt_density):
    """L1 trên log2 kích thước vật, trọng số theo density GT: chỉ học ở nơi có vật."""
    w = gt_density.clamp(min=0)
    return ((log_size - gt_log_size).abs() * w).sum() / w.sum().clamp(min=1e-6)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def evaluate(model, loader, device, amp):
    """Trên một split (zero-shot: exemplar = đỉnh density thô): MAE / RMSE của count = sum(density map),
    MAE của density thô bước 1, và chất lượng ước lượng kích thước vật của Scale Head
    (sai số |log2| trung vị, Spearman với kích thước exemplar GT)."""
    model.eval_mode()
    abs_err, sq_err, coarse_err, n = 0.0, 0.0, 0.0, 0
    est_size, gt_size = [], []
    for batch in loader:
        images = batch['image'].to(device)
        with torch.autocast(device.type, enabled=amp):
            _, pred, aux = model(images, batch['class_name'], return_aux=True)
        pred = pred.float()
        diff = pred.sum(dim=(1, 2, 3)).cpu() - batch['count']
        abs_err += diff.abs().sum().item()
        sq_err += (diff ** 2).sum().item()
        coarse_err += (aux['coarse'].float().sum(dim=(1, 2, 3)).cpu() - batch['count']).abs().sum().item()
        n += len(diff)
        if aux['log_size'] is not None:
            w = pred.clamp(min=0)
            est = (aux['log_size'].float() * w).sum(dim=(1, 2, 3)) / w.sum(dim=(1, 2, 3)).clamp(min=1e-6)
            est_size += est.cpu().tolist()
            gt_size += batch['size_map'][:, 0, 0, 0].tolist()
    res = {"mae": abs_err / n, "rmse": (sq_err / n) ** 0.5, "coarse_mae": coarse_err / n}
    if est_size:
        e, g = np.array(est_size), np.array(gt_size)
        res.update(scale_err=float(np.median(np.abs(e - g))), scale_rho=spearman(e, g))
    return res


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"
    print(f"Bắt đầu huấn luyện trên: {device} (AMP={amp})")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Mô hình: YOLO-World đóng băng, chỉ train Density Head
    model = DensityYOLOWorld(args.weights, arch=args.arch, vit_name=args.vit_name,
                             vit_input=args.vit_input).to(device)
    if args.unfreeze_from is not None:
        # Gọi trước load_head: mốc L2-SP là YOLO-World pretrained, không phải neck của checkpoint --init
        model.unfreeze_yolo(args.unfreeze_from)
        print(f"Mở đóng băng YOLO-World từ layer {args.unfreeze_from}: {sum(p.numel() for p in model.yolo_parameters()):,} "
              f"tham số, lr x{args.yolo_lr_mult}, L2-SP w={args.l2sp_weight}")
    if args.init:
        ckpt = model.load_head(args.init, map_location=device)
        print(f"Fine-tune từ {args.init} (epoch {ckpt.get('epoch')}, val MAE {ckpt.get('val_mae')})")

    # 2. Dữ liệu
    train_set = FSC147Dataset(args.data_dir, split="train", hflip=True, mosaic_prob=args.mosaic_prob,
                              mosaic_grids=args.mosaic_grids)
    val_set = FSC147Dataset(args.data_dir, split="val")
    # Tập con train không augment để theo dõi khoảng cách train/val MAE
    train_probe = Subset(FSC147Dataset(args.data_dir, split="train"), range(0, len(train_set), 4))
    print(f"GT density: {'adaptive chính thức' if train_set.use_official_gt else 'Gaussian từ points'}")
    if args.limit:
        train_set = Subset(train_set, range(min(args.limit, len(train_set))))
        val_set = Subset(val_set, range(min(args.limit, len(val_set))))
        train_probe = Subset(train_probe, range(min(args.limit, len(train_probe))))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              collate_fn=collate_fsc147, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            collate_fn=collate_fsc147, pin_memory=True)
    probe_loader = DataLoader(train_probe, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                              collate_fn=collate_fsc147, pin_memory=True)
    print(f"Train: {len(train_set)} ảnh (mosaic p={args.mosaic_prob}, lưới {args.mosaic_grids}) | Val: {len(val_set)} ảnh | "
          f"Train probe: {len(train_probe)} ảnh | arch={model.arch} | region loss w={args.region_weight}")

    # 3. Optimizer (Density Head + Self-Exemplar Matcher + Refine Head [+ neck YOLO-World, lr nhỏ, không weight
    # decay vì L2-SP đã kéo về trọng số pretrained thay vì về 0])
    groups = [{"params": model.trainable_modules().parameters()}]
    if model.yolo_parameters():
        groups.append({"params": model.yolo_parameters(), "lr": args.lr * args.yolo_lr_mult, "weight_decay": 0.0})
    optimizer = optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=amp)

    best_mae = float("inf")
    if args.init:
        # Mốc ban đầu = checkpoint khởi tạo: best chỉ bị ghi đè khi fine-tune thực sự tốt hơn
        best_mae = evaluate(model, val_loader, device, amp)["mae"]
        model.save_head(os.path.join(args.out_dir, "density_head_best.pth"), epoch=0, val_mae=best_mae)
        print(f"Val MAE của checkpoint khởi tạo: {best_mae:.2f}")
    for epoch in range(1, args.epochs + 1):
        model.train_mode()
        epoch_loss, epoch_l2sp = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            images = batch['image'].to(device)
            gt = batch['density_map'].to(device)

            # Một phần ảnh dùng GT points làm exemplar để Refine Head học tin vào kênh similarity;
            # phần còn lại dùng đỉnh density thô như lúc suy luận zero-shot
            exemplars = [p if torch.rand(1).item() < args.gt_exemplar_prob else None for p in batch['points']]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                _, pred, aux = model(images, batch['class_name'], exemplar_points=exemplars, return_aux=True)
            # MSE trên map đã nhân DENSITY_SCALE để loss có độ lớn dễ đọc; loss phụ cho density thô;
            # loss số đếm theo vùng đa tỷ lệ cho density cuối; loss kích thước vật cho Scale Head
            pred, coarse = pred.float(), aux['coarse'].float()
            loss = F.mse_loss(pred * DENSITY_SCALE, gt * DENSITY_SCALE) \
                + args.coarse_weight * F.mse_loss(coarse * DENSITY_SCALE, gt * DENSITY_SCALE) \
                + args.region_weight * region_count_loss(pred, gt)
            if aux['log_size'] is not None:
                loss = loss + args.scale_weight * scale_loss(aux['log_size'].float(),
                                                             batch['size_map'].to(device), gt)
            if model.yolo_parameters():
                l2sp = model.yolo_l2sp()
                loss = loss + args.l2sp_weight * l2sp
                epoch_l2sp += l2sp.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        v = evaluate(model, val_loader, device, amp)
        val_mae = v["mae"]
        msg = (f"Epoch [{epoch}/{args.epochs}] loss={epoch_loss / len(train_loader):.4f} "
               f"| val MAE={val_mae:.2f} RMSE={v['rmse']:.2f} | coarse MAE={v['coarse_mae']:.2f}")
        if model.yolo_parameters():
            msg += f" | L2-SP={epoch_l2sp / len(train_loader):.4g}"
        if "scale_rho" in v:
            msg += f" | scale |log2|={v['scale_err']:.3f} rho={v['scale_rho']:.3f}"
        if epoch % 5 == 0 or epoch == args.epochs:
            msg += f" | train-probe MAE={evaluate(model, probe_loader, device, amp)['mae']:.2f}"
        print(msg)

        model.save_head(os.path.join(args.out_dir, "density_head_last.pth"), epoch=epoch, val_mae=val_mae)
        if val_mae < best_mae:
            best_mae = val_mae
            model.save_head(os.path.join(args.out_dir, "density_head_best.pth"), epoch=epoch, val_mae=val_mae)
            print(f"  -> Lưu best (val MAE={val_mae:.2f})")

    print(f"Huấn luyện hoàn tất. Best val MAE = {best_mae:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Density Head (text-conditioned) trên FSC-147")
    parser.add_argument("--data-dir", default="data/FSC147")
    parser.add_argument("--weights", default="yolov8s-world.pt")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--init", default=None, help="Checkpoint head để fine-tune tiếp (mặc định: train từ đầu)")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ dùng N ảnh mỗi split (chạy thử)")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--gt-exemplar-prob", type=float, default=0.5,
                        help="Xác suất dùng GT points làm exemplar khi train (còn lại: đỉnh density thô)")
    parser.add_argument("--coarse-weight", type=float, default=0.5, help="Trọng số loss của density thô")
    parser.add_argument("--region-weight", type=float, default=25.0,
                        help="Trọng số loss số đếm theo vùng đa tỷ lệ (0 = tắt)")
    parser.add_argument("--scale-weight", type=float, default=2.0,
                        help="Trọng số loss log2 kích thước vật của Scale Head")
    parser.add_argument("--mosaic-grids", type=int, nargs="+", default=[2],
                        help="Lưới mosaic chọn ngẫu nhiên (vd 2 4: thêm ảnh ghép 4x4 dày đặc vật nhỏ)")
    parser.add_argument("--mosaic-prob", type=float, default=0.3,
                        help="Xác suất ghép mosaic 2x2 cùng lớp cho một mẫu train (0 = tắt)")
    parser.add_argument("--arch", choices=["simcentric", "vit"], default="simcentric",
                        help="vit: thêm nhánh DINOv2 đóng băng (feature + Self-Exemplar trên ViT)")
    parser.add_argument("--vit-name", default="facebook/dinov2-small", help="DINOv2 trên HF Hub (arch vit)")
    parser.add_argument("--vit-input", type=int, default=560,
                        help="Cạnh ảnh đưa vào ViT, bội của 14 (560 -> lưới 40x40, 840 -> 60x60)")
    parser.add_argument("--unfreeze-from", type=int, default=None,
                        help="Train cả YOLO-World từ layer này đến trước Detect (12 = neck); mặc định đóng băng")
    parser.add_argument("--yolo-lr-mult", type=float, default=0.1, help="lr của phần YOLO-World = lr x hệ số này")
    parser.add_argument("--l2sp-weight", type=float, default=0.01,
                        help="Trọng số L2-SP kéo neck về trọng số pretrained (0 = tắt)")
    train(parser.parse_args())
