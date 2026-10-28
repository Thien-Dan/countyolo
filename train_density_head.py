import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.density_yolo import DensityYOLOWorld, DENSITY_SCALE
from datasets.fsc147_dataset import FSC147Dataset, collate_fsc147


REGION_SCALES = (4, 8, 16, 80)  # cạnh vùng (ô lưới P3); 80 = toàn ảnh 640


def region_count_loss(pred, gt, scales=REGION_SCALES, weights=None):
    """L1 của số đếm trong từng vùng s x s (sum-pool), chia cho s để các tỷ lệ cùng độ lớn.
    Ép đúng số lượng theo vùng thay vì ép từng pixel khớp hình Gaussian của GT.
    weights (B,): trọng số từng mẫu (trung bình 1); None = như nhau."""
    diff = pred - gt
    per = sum((F.avg_pool2d(diff, s) * s * s).abs().mean(dim=(1, 2, 3)) / s for s in scales) / len(scales)
    return per.mean() if weights is None else (per * weights).mean()


def weighted_mse(a, b, weights=None):
    per = ((a - b) ** 2).mean(dim=(1, 2, 3))
    return per.mean() if weights is None else (per * weights).mean()


def sample_weights(gt, ref):
    """Chuẩn hoá loss theo từng mẫu: w = 1 / sqrt(1 + count / ref), chuẩn hoá về trung bình 1 trong batch.
    Ảnh / mosaic cực đông có loss lớn gấp nhiều lần nên chi phối gradient (mosaic 4x4 đã thất bại vì vậy)."""
    w = 1.0 / torch.sqrt(1.0 + gt.sum(dim=(1, 2, 3)).clamp(min=0) / ref)
    return w / w.mean()


def scale_loss(log_size, gt_log_size, gt_density):
    """L1 trên log2 kích thước vật, trọng số theo density GT: chỉ học ở nơi có vật."""
    w = gt_density.clamp(min=0)
    return ((log_size - gt_log_size).abs() * w).sum() / w.sum().clamp(min=1e-6)


class EMA:
    """Trung bình động trọng số các head (val MAE dao động mạnh giữa các epoch, vd 21-30): eval/lưu bằng bản EMA,
    không tốn thêm thời gian suy luận. Buffer (BN running stats) cũng được lấy trung bình."""

    def __init__(self, modules, decay):
        self.modules, self.decay = modules, decay
        self.shadow = {k: v.detach().clone().float() for k, v in modules.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self):
        for k, v in self.modules.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def swap(self):
        """Đổi qua lại trọng số đang train <-> EMA (gọi 2 lần là trở về như cũ)."""
        sd = self.modules.state_dict()
        for k, v in self.shadow.items():
            cur = sd[k].detach().clone().float()
            sd[k].copy_(v.to(sd[k].dtype))
            self.shadow[k] = cur


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def evaluate(model, loader, device, amp, wrong_prompts=None, gt_exemplars=False):
    """Trên một split (zero-shot: exemplar = đỉnh density thô): MAE / RMSE của count = sum(density map),
    MAE của density thô bước 1, và chất lượng ước lượng kích thước vật của Scale Head
    (sai số |log2| trung vị, Spearman với kích thước exemplar GT).
    wrong_prompts: list tên lớp; nếu có, đếm thêm với một lớp sai cho mỗi ảnh -> "wrong_mean" (lý tưởng 0)."""
    model.eval_mode()
    abs_err, sq_err, coarse_err, n = 0.0, 0.0, 0.0, 0
    est_size, gt_size, wrong = [], [], []
    rng = np.random.default_rng(0)  # cùng các lớp sai ở mọi epoch
    for batch in loader:
        images = batch['image'].to(device)
        with torch.autocast(device.type, enabled=amp):
            # gt_exemplars: exemplar = điểm GT (mô hình train luôn có exemplar, vd do SAM 3 cung cấp lúc chạy thật)
            ex = batch['points'] if gt_exemplars else None
            _, pred, aux = model(images, batch['class_name'], exemplar_points=ex, return_aux=True)
        pred = pred.float()
        diff = pred.sum(dim=(1, 2, 3)).cpu() - batch['count']
        abs_err += diff.abs().sum().item()
        sq_err += (diff ** 2).sum().item()
        coarse_err += (aux['coarse'].float().sum(dim=(1, 2, 3)).cpu() - batch['count']).abs().sum().item()
        n += len(diff)
        if wrong_prompts:
            names = [str(rng.choice([c for c in wrong_prompts if c != cn])) for cn in batch['class_name']]
            with torch.autocast(device.type, enabled=amp):
                wrong += model(images, names)[1].float().sum(dim=(1, 2, 3)).cpu().tolist()
        if aux['log_size'] is not None:
            w = pred.clamp(min=0)
            est = (aux['log_size'].float() * w).sum(dim=(1, 2, 3)) / w.sum(dim=(1, 2, 3)).clamp(min=1e-6)
            est_size += est.cpu().tolist()
            gt_size += batch['size_map'][:, 0, 0, 0].tolist()
    res = {"mae": abs_err / n, "rmse": (sq_err / n) ** 0.5, "coarse_mae": coarse_err / n}
    if wrong:
        res["wrong_mean"] = float(np.mean(wrong))
    if est_size:
        e, g = np.array(est_size), np.array(gt_size)
        res.update(scale_err=float(np.median(np.abs(e - g))), scale_rho=spearman(e, g))
    return res


@torch.no_grad()
def evaluate_mosaics(model, ds, device, amp, n, seed=0):
    """n mosaic 2x2 từ 2 lớp val khác nhau (2 ô mỗi lớp, cùng cách dựng với tools/prompt_eval.py, cố định theo seed),
    hỏi từng lớp với exemplar = điểm GT của lớp đó (lúc chạy thật do SAM 3 cấp) -> MAE + sai số tương đối trung vị."""
    model.eval_mode()
    rng = random.Random(seed)
    classes = [c for c in sorted(ds.class_to_idx) if len(ds.class_to_idx[c]) >= 2]
    half = ds.image_size // 2
    err, rel = [], []
    for _ in range(n):
        ca, cb = rng.sample(classes, 2)
        tiles = [(ca, j) for j in rng.sample(ds.class_to_idx[ca], 2)] +                 [(cb, j) for j in rng.sample(ds.class_to_idx[cb], 2)]
        rng.shuffle(tiles)
        image = torch.zeros(3, ds.image_size, ds.image_size)
        gt, pts = {ca: 0.0, cb: 0.0}, {ca: [], cb: []}
        for k, (c, j) in enumerate(tiles):
            smp = ds._load(j)
            oy, ox = divmod(k, 2)
            image[:, oy * half:(oy + 1) * half, ox * half:(ox + 1) * half] =                 F.interpolate(smp['image'][None], size=(half, half), mode='area')[0]
            gt[c] += smp['count']
            pts[c].append(smp['points'] / 2 + torch.tensor([ox * half, oy * half], dtype=torch.float32))
        images = torch.stack([image, image]).to(device)
        with torch.autocast(device.type, enabled=amp):
            pred = model(images, [ca, cb], exemplar_points=[torch.cat(pts[ca]), torch.cat(pts[cb])])[1]
        for c, pr in zip((ca, cb), pred.float().sum(dim=(1, 2, 3)).cpu().tolist()):
            err.append(abs(pr - gt[c]))
            rel.append(abs(pr - gt[c]) / max(gt[c], 1))
    return {"mosaic_mae": float(np.mean(err)), "mosaic_rel": float(np.median(rel))}


def noisy_points(points, q, size, other=None):
    """Thay mỗi điểm exemplar với xác suất q bằng: một vật lớp khác trong ảnh (other, mosaic trộn lớp — giống CLIP chọn
    nhầm lớp) nếu có, ngược lại một điểm ngẫu nhiên."""
    if points is None or len(points) == 0:
        return points
    points = points.clone()
    m = torch.rand(len(points)) < q
    n = int(m.sum())
    if n:
        if other is not None and len(other):
            points[m] = other[torch.randint(len(other), (n,))]
        else:
            points[m] = torch.rand(n, 2) * size
    return points


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
    if args.gate_only:
        model.freeze_base()
        print(f"Chỉ train cổng text: {sum(p.numel() for p in model.gate.parameters()):,} tham số, phần đếm đóng băng")

    # 2. Dữ liệu
    train_set = FSC147Dataset(args.data_dir, split="train", hflip=True, mosaic_prob=args.mosaic_prob,
                              mosaic_grids=args.mosaic_grids, neg_prob=args.neg_prob, mixed_prob=args.mixed_prob,
                              pseudo=args.pseudo, hard_prob=args.hard_prob)
    val_set = FSC147Dataset(args.data_dir, split="val")
    # Có train prompt âm: khi chọn checkpoint, đo thêm số đếm khi hỏi sai lớp (lớp val khác lớp của ảnh)
    wrong_prompts = sorted(val_set.class_to_idx) if args.neg_prob > 0 else None
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
    print(f"Train: {len(train_set)} ảnh (mosaic p={args.mosaic_prob}, lưới {args.mosaic_grids}, trộn lớp {args.mixed_prob}, prompt sai {args.neg_prob}) | Val: {len(val_set)} ảnh | "
          f"Train probe: {len(train_probe)} ảnh | arch={model.arch} | region loss w={args.region_weight}")

    # 3. Optimizer (Density Head + Self-Exemplar Matcher + Refine Head [+ neck YOLO-World, lr nhỏ, không weight
    # decay vì L2-SP đã kéo về trọng số pretrained thay vì về 0])
    groups = [{"params": [p for p in model.trainable_modules().parameters() if p.requires_grad]}]
    if model.yolo_parameters():
        groups.append({"params": model.yolo_parameters(), "lr": args.lr * args.yolo_lr_mult, "weight_decay": 0.0})
    optimizer = optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=amp)
    ema = EMA(model.trainable_modules(), args.ema) if args.ema > 0 else None

    best_mae = float("inf")
    # Mosaic val 2 lớp (exemplar đúng lớp): cộng MAE mosaic vào tiêu chí chọn best
    mosaic_ds = val_set.dataset if isinstance(val_set, Subset) else val_set
    if args.init:
        # Mốc ban đầu = checkpoint khởi tạo: best chỉ bị ghi đè khi fine-tune thực sự tốt hơn
        v0 = evaluate(model, val_loader, device, amp, wrong_prompts, gt_exemplars=args.gt_exemplar_prob >= 1)
        best_mae = v0["mae"] + v0.get("wrong_mean", 0.0)
        if args.val_mosaics:
            m0 = evaluate_mosaics(model, mosaic_ds, device, amp, args.val_mosaics)
            best_mae += m0["mosaic_mae"]
            print(f"Mosaic val của checkpoint khởi tạo: MAE {m0['mosaic_mae']:.2f} | tương đối {m0['mosaic_rel']:.2f}")
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
            if args.ex_noise > 0:
                # Exemplar nhiễu như lúc chạy thật (CLIP / SAM 3 đôi khi chọn nhầm vật lớp khác hoặc nền): thay một phần
                # điểm GT bằng điểm ngẫu nhiên trong ảnh
                exemplars = [noisy_points(p, args.ex_noise, images.shape[-1], o if args.ex_noise_other else None)
                             for p, o in zip(exemplars, batch['other_points'])]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                _, pred, aux = model(images, batch['class_name'], exemplar_points=exemplars, return_aux=True)
            # MSE trên map đã nhân DENSITY_SCALE để loss có độ lớn dễ đọc; loss phụ cho density thô;
            # loss số đếm theo vùng đa tỷ lệ cho density cuối; loss kích thước vật cho Scale Head
            pred, coarse = pred.float(), aux['coarse'].float()
            w = sample_weights(gt, args.loss_norm_ref) if args.loss_norm_ref > 0 else None
            # Mẫu negative khó (lớp khác có thật trong ảnh, nhãn SAM 3) không có density GT -> chỉ học cổng
            dv = batch['density_valid'].to(device)
            if not bool((dv > 0).all()):
                w = dv if w is None else w * dv
            loss = weighted_mse(pred * DENSITY_SCALE, gt * DENSITY_SCALE, w) \
                + args.coarse_weight * weighted_mse(coarse * DENSITY_SCALE, gt * DENSITY_SCALE, w) \
                + args.region_weight * region_count_loss(pred, gt, weights=w)
            if args.gate_bce > 0 and aux.get('gate') is not None:
                g = aux['gate'].float().clamp(1e-6, 1 - 1e-6)
                loss = loss + args.gate_bce * F.binary_cross_entropy(g, batch['gate_mask'].to(device))
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
            if ema is not None:
                ema.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        if ema is not None:
            ema.swap()  # eval + lưu checkpoint bằng trọng số EMA
        v = evaluate(model, val_loader, device, amp, wrong_prompts, gt_exemplars=args.gt_exemplar_prob >= 1)
        val_mae = v["mae"]
        score = val_mae + v.get("wrong_mean", 0.0)  # tiêu chí chọn best
        if args.val_mosaics:
            v.update(evaluate_mosaics(model, mosaic_ds, device, amp, args.val_mosaics))
            score += v["mosaic_mae"]
        msg = (f"Epoch [{epoch}/{args.epochs}] loss={epoch_loss / len(train_loader):.4f} "
               f"| val MAE={val_mae:.2f} RMSE={v['rmse']:.2f} | coarse MAE={v['coarse_mae']:.2f}")
        if model.yolo_parameters():
            msg += f" | L2-SP={epoch_l2sp / len(train_loader):.4g}"
        if "scale_rho" in v:
            msg += f" | scale |log2|={v['scale_err']:.3f} rho={v['scale_rho']:.3f}"
        if epoch % 5 == 0 or epoch == args.epochs:
            msg += f" | train-probe MAE={evaluate(model, probe_loader, device, amp)['mae']:.2f}"
        if "wrong_mean" in v:
            msg += f" | hỏi sai lớp: đếm TB {v['wrong_mean']:.2f}"
        if "mosaic_mae" in v:
            msg += f" | mosaic 2 lớp MAE {v['mosaic_mae']:.2f} (tương đối {v['mosaic_rel']:.2f})"
        print(msg)

        model.save_head(os.path.join(args.out_dir, "density_head_last.pth"), epoch=epoch, val_mae=val_mae)
        if score < best_mae:
            best_mae = score
            model.save_head(os.path.join(args.out_dir, "density_head_best.pth"), epoch=epoch, val_mae=val_mae,
                            ema=args.ema)
            print(f"  -> Lưu best (val MAE={val_mae:.2f}, tiêu chí={score:.2f})")
        if ema is not None:
            ema.swap()  # trả lại trọng số đang train

    print(f"Huấn luyện hoàn tất. Best val MAE = {best_mae:.2f}"
          + (" (tiêu chí = val MAE + đếm TB khi hỏi sai lớp)" if wrong_prompts else "")
          + (" (+ MAE mosaic 2 lớp val)" if args.val_mosaics else ""))


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
    parser.add_argument("--ex-noise", type=float, default=0.0,
                        help="Tỉ lệ điểm exemplar GT bị thay bằng điểm ngẫu nhiên khi train (chịu được exemplar chọn nhầm)")
    parser.add_argument("--ex-noise-other", action="store_true",
                        help="Exemplar nhiễu lấy vật lớp khác trong mosaic trộn lớp (khi có) thay cho điểm ngẫu nhiên")
    parser.add_argument("--val-mosaics", type=int, default=0,
                        help="Số mosaic val 2 lớp (exemplar GT đúng lớp); > 0: cộng MAE mosaic vào tiêu chí chọn best")
    parser.add_argument("--coarse-weight", type=float, default=0.5, help="Trọng số loss của density thô")
    parser.add_argument("--region-weight", type=float, default=25.0,
                        help="Trọng số loss số đếm theo vùng đa tỷ lệ (0 = tắt)")
    parser.add_argument("--scale-weight", type=float, default=2.0,
                        help="Trọng số loss log2 kích thước vật của Scale Head")
    parser.add_argument("--mosaic-grids", type=int, nargs="+", default=[2],
                        help="Lưới mosaic chọn ngẫu nhiên (vd 2 4: thêm ảnh ghép 4x4 dày đặc vật nhỏ)")
    parser.add_argument("--mosaic-prob", type=float, default=0.3,
                        help="Xác suất ghép mosaic 2x2 cùng lớp cho một mẫu train (0 = tắt)")
    parser.add_argument("--neg-prob", type=float, default=0.0,
                        help="Xác suất thay prompt bằng lớp khác, đáp án density 0 (dạy mô hình nghe theo text)")
    parser.add_argument("--mixed-prob", type=float, default=0.0,
                        help="Tỉ lệ mosaic trộn ô của lớp khác (chỉ ô đúng lớp được đếm)")
    parser.add_argument("--ema", type=float, default=0.0,
                        help="> 0: EMA trọng số head với hệ số này (vd 0.999), eval/lưu bằng bản EMA")
    parser.add_argument("--pseudo", default=None, help="Nhãn giả SAM 3 cho train (tools/sam3_pseudo.py, .npz)")
    parser.add_argument("--hard-prob", type=float, default=0.0,
                        help="Xác suất đổi prompt sang lớp khác có thật trong ảnh (theo nhãn SAM 3)")
    parser.add_argument("--gate-bce", type=float, default=0.0, help="Trọng số BCE cổng text vs mask SAM 3")
    parser.add_argument("--loss-norm-ref", type=float, default=0.0,
                        help="> 0: trọng số mẫu 1/sqrt(1 + count/ref) cho MSE + region loss (0 = tắt, như cũ)")
    parser.add_argument("--gate-only", action="store_true",
                        help="arch vitgate + --init <ckpt vit>: đóng băng phần đếm, chỉ train cổng text")
    parser.add_argument("--arch", choices=["simcentric", "vit", "vitpa", "vitgate"], default="simcentric",
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
