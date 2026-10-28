"""
Canonical-Scale Zoom: đếm lại những vùng có vật nhỏ ở kích thước chuẩn.

Sai số đếm tập trung gần như hoàn toàn ở ảnh có vật nhỏ (< ~40px trong ảnh 640): P3 stride 8 không phân giải
được, nên đếm thiếu nặng. Scale Head cho biết kích thước vật ở từng vùng; vùng nào có vật nhỏ hơn kích thước
chuẩn thì cắt từ ảnh gốc, phóng về 640 để vật xấp xỉ kích thước chuẩn, đếm lại và thay đúng phần density của
vùng đó. Density có tính cộng nên thay theo vùng vẫn nhất quán; vùng có vật đủ lớn giữ nguyên, tránh việc
zoom cả ảnh làm đếm dư ở ảnh thưa.
"""
import torch
import torchvision.transforms.functional as TF


def weighted_log_size(log_size, density, fallback=None):
    """Trung bình log2 kích thước vật, trọng số theo density (vùng có vật). Không có vật -> fallback."""
    w = density.clamp(min=0)
    total = w.sum()
    if total < 1e-3:
        return fallback
    return float((log_size * w).sum() / total)


@torch.no_grad()
def canonical_scale_count(model, image, class_name, device, target=44.0, min_zoom=1.5, max_zoom=6, chunk=16,
                          size=640):
    """
    Args:
        model: DensityYOLOWorld có Scale Head
        image: PIL.Image gốc (RGB)
        size: kích thước ảnh đưa vào model
        target: kích thước vật chuẩn (px trong ảnh 640) mà head đếm tốt nhất
        min_zoom: chỉ zoom vùng cần phóng ít nhất chừng này lần
        max_zoom: số ô tối đa mỗi chiều (vùng được phóng tối đa ~max_zoom lần)

    Returns:
        count (float), info dict {"count_full", "log_size", "k", "zoomed"}
    """
    x = TF.to_tensor(image.resize((size, size)))[None].to(device)
    _, density, aux = model(x, [class_name], return_aux=True)
    d = density[0, 0].float()
    count_full = float(d.sum())
    info = {"count_full": count_full, "log_size": None, "k": 1, "zoomed": 0}
    if aux["log_size"] is None:
        return count_full, info
    ls = aux["log_size"][0, 0].float()
    g = weighted_log_size(ls, d)
    info["log_size"] = g
    if g is None or target / 2 ** g < min_zoom:
        return count_full, info

    k = int(min(max_zoom, max(2, round(target / 2 ** g))))
    info["k"] = k
    H, W = d.shape
    ys = [round(i * H / k) for i in range(k + 1)]
    xs = [round(i * W / k) for i in range(k + 1)]
    iw, ih = image.size
    total, crops = 0.0, []
    for i in range(k):
        for j in range(k):
            rd, rl = d[ys[i]:ys[i + 1], xs[j]:xs[j + 1]], ls[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            r = weighted_log_size(rl, rd, fallback=g)
            if target / 2 ** r >= min_zoom:
                box = (xs[j] * iw / W, ys[i] * ih / H, xs[j + 1] * iw / W, ys[i + 1] * ih / H)
                crops.append(TF.to_tensor(image.crop(tuple(round(v) for v in box)).resize((size, size))))
            else:
                total += float(rd.sum())
    for s in range(0, len(crops), chunk):
        batch = torch.stack(crops[s:s + chunk]).to(device)
        _, dz = model(batch, [class_name] * len(batch))
        total += float(dz.float().sum())
    info["zoomed"] = len(crops)
    return total, info


def _peak_boxes(density, log_size, count, stride=8.0, default_side=24.0, min_mass=0.0):
    """round(count) đỉnh cục bộ mạnh nhất của density (H, W) -> box vuông xyxy quanh tâm ô, cạnh = kích thước vật
    Scale Head dự đoán tại ô đó (px ảnh 640). Không có Scale Head -> cạnh default_side.
    min_mass: bỏ đỉnh có tổng density 3x3 quanh nó < min_mass (một vật ~ khối lượng 1); đỉnh yếu trên nền
    thường là phát hiện giả, mỗi cái thành một track giả khi đếm video. Score trả về = khối lượng 3x3."""
    d = density[None, None]
    peaks = torch.where(d == torch.nn.functional.max_pool2d(d, 3, stride=1, padding=1), d, torch.zeros_like(d))[0, 0]
    mass = (torch.nn.functional.avg_pool2d(d, 3, stride=1, padding=1) * 9)[0, 0]
    peaks = torch.where((peaks > 0) & (mass >= min_mass), mass, torch.zeros_like(mass))
    k = min(int(round(count)), int((peaks > 0).sum()))
    if k == 0:
        return torch.zeros(0, 4), torch.zeros(0)
    vals, idx = peaks.flatten().topk(k)
    W = density.shape[-1]
    cx = (idx % W).float() * stride + stride / 2
    cy = (idx // W).float() * stride + stride / 2
    side = 2 ** log_size.flatten()[idx] if log_size is not None else torch.full_like(cx, default_side)
    half = side / 2
    return torch.stack([cx - half, cy - half, cx + half, cy + half], 1).cpu(), vals.cpu()


def _sample_feats(feats, boxes, size):
    """Feature (1, D, h, w) lấy tại tâm box (toạ độ ảnh size x size), chuẩn hoá L2 -> (N, D)."""
    if feats is None:
        return None
    if len(boxes) == 0:
        return torch.zeros(0, feats.shape[1])
    c = (boxes[:, :2] + boxes[:, 2:]) / 2 / size * 2 - 1  # [-1, 1]
    grid = c.to(feats.device, feats.dtype).view(1, -1, 1, 2)
    f = torch.nn.functional.grid_sample(feats, grid, align_corners=False)[0, :, :, 0].T.float()
    return torch.nn.functional.normalize(f, dim=1).cpu()


@torch.no_grad()
def density_tiled_count(model, image, class_name, device, count_thr=0.0, grid=2, size=640, return_boxes=False,
                        min_mass=0.0, protos=None):
    """
    Test-time tiling kích hoạt theo số đếm: đếm toàn ảnh; nếu count >= count_thr (> 0) thì chia ảnh gốc thành
    grid x grid ô không chồng lấn, mỗi ô resize về size, đếm lại và cộng (density có tính cộng). Nhắm vào ảnh
    rất đông mà mô hình đếm thiếu: trong mỗi ô vật to lên grid lần và mật độ mỗi ô gần phân bố lúc train hơn.

    Args:
        image: PIL.Image gốc (RGB)
        return_boxes: trả thêm box vuông tại round(count) đỉnh density (hệ toạ độ ảnh gốc) + score, dùng làm
            box prompt (vd Stage 1 của CountVid)
        min_mass: ngưỡng khối lượng density 3x3 của đỉnh khi tạo box (xem _peak_boxes)
        protos: prototype exemplar cho trước (aux["protos"] của model, vd từ frame chính có exemplar SAM 3)
    Returns:
        count, info {"count_full", "tiled", "density" (lần đếm toàn ảnh, H x W), "feats" (N, D) feature DINOv2
        tại từng box hoặc None} [, boxes (N, 4), scores (N,)]
    """
    iw, ih = image.size

    def run(img):
        x = TF.to_tensor(img.convert("RGB").resize((size, size)))[None].to(device)
        _, density, aux = model(x, [class_name], return_aux=True, exemplar_protos=protos)
        d = density[0, 0].float()
        ls = aux["log_size"][0, 0].float() if aux.get("log_size") is not None else None
        return d, ls, aux.get("vit_feats")

    d, ls, vf = run(image)
    count_full = float(d.sum())
    info = {"count_full": count_full, "tiled": False, "density": d.cpu(), "feats": None}
    if count_thr <= 0 or count_full < count_thr or grid < 2:
        if not return_boxes:
            return count_full, info
        b, s = _peak_boxes(d, ls, count_full, min_mass=min_mass)
        info["feats"] = _sample_feats(vf, b, size)
        return count_full, info, b * torch.tensor([iw / size, ih / size] * 2), s

    info["tiled"] = True
    xs = [round(i * iw / grid) for i in range(grid + 1)]
    ys = [round(i * ih / grid) for i in range(grid + 1)]
    total, boxes, scores, feats = 0.0, [], [], []
    for i in range(grid):
        for j in range(grid):
            x0, y0, x1, y1 = xs[j], ys[i], xs[j + 1], ys[i + 1]
            dt, lt, ft = run(image.crop((x0, y0, x1, y1)))
            c = float(dt.sum())
            total += c
            if return_boxes:
                b, s = _peak_boxes(dt, lt, c, min_mass=min_mass)
                feats.append(_sample_feats(ft, b, size))
                b = b * torch.tensor([(x1 - x0) / size, (y1 - y0) / size] * 2) + torch.tensor([x0, y0, x0, y0])
                boxes.append(b)
                scores.append(s)
    if not return_boxes:
        return total, info
    if all(f is not None for f in feats) and feats:
        info["feats"] = torch.cat(feats)
    return total, info, torch.cat(boxes), torch.cat(scores)
