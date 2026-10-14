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
