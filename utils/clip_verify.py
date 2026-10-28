"""
Kiểm tra "vật CountYOLO thấy có đúng là lớp được hỏi không" bằng CLIP ViT-B/32 mà YOLO-World đã nạp sẵn (encoder text
của YOLO-World = CLIP; phần encoder ảnh đi kèm, không tốn thêm bộ nhớ).

Mỗi vùng cắt (quanh đỉnh density, cạnh theo Scale Head) được so với prompt và một bộ từ vựng chung (lớp FSC-147 train +
80 lớp COCO, bỏ chính prompt): p = softmax(100 * cos)[prompt]. Dùng để
  - có mặt: không vùng nào giống prompt hơn từ vựng -> lớp được hỏi vắng mặt -> đếm 0
  - exemplar: các vùng giống prompt nhất làm exemplar cho CountYOLO (cảnh nhiều lớp)

VerifiedCounter gộp tất cả vào MỘT lượt CountYOLO: vùng ứng viên (đỉnh density thô + box YOLO-World cho prompt) được
kiểm tra ngay sau bước 1 (hook exemplar_fn của DensityYOLOWorld.forward), trước Refine Head.
"""
import os

import torch
import torch.nn.functional as F
import yaml
from torchvision.ops import nms, roi_align

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# Trung bình embedding nhiều câu mẫu (ảnh đếm vật thường có nhiều vật, cận cảnh)
TEMPLATES = ("a photo of {}.", "a photo of many {}.", "a close-up photo of {}.")


def same_name(a, b):
    """Cùng một khái niệm khác số ít / số nhiều (vd apples / apple) -> không dùng làm từ đối chứng của nhau."""
    a, b = a.lower().strip(), b.lower().strip()
    return a == b or a + "s" == b or b + "s" == a or a + "es" == b or b + "es" == a


def default_vocab():
    """Từ vựng đối chứng mặc định: 89 lớp FSC-147 train (utils/fsc147_train_classes.txt) + 80 lớp COCO — giống lúc chọn
    ngưỡng trong tools/clip_verify_eval.py."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fsc147_train_classes.txt")
    return [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()] + coco_names()


def _dataset_names(fname):
    import ultralytics
    p = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "datasets", fname)
    return list(yaml.safe_load(open(p))["names"].values())


def lvis_names():
    """1203 lớp LVIS (lấy tên đầu trong "a/b", bỏ "_" và chú thích trong ngoặc)."""
    out = []
    for n in _dataset_names("lvis.yaml"):
        n = n.split("/")[0].split("(")[0].replace("_", " ").strip()
        if n:
            out.append(n)
    return out


def o365_names():
    return [n.split("/")[0].strip() for n in _dataset_names("Objects365.yaml")]


def coco_names():
    import ultralytics
    p = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "datasets", "coco.yaml")
    return list(yaml.safe_load(open(p))["names"].values())


class ClipVerifier:
    def __init__(self, model, vocab, crop=224, context=1.5, min_side=32.0, half=None, compile_batch=0, cache_path=None):
        """compile_batch > 0: torch.compile(mode="reduce-overhead") encoder ảnh với batch cố định (đệm 0 cho đủ) — CLIP chạy
        thường bị chi phối bởi chi phí khởi chạy kernel (T4: 13 hay 25 vùng đều ~19 ms; compile 13 vùng 8.6 ms).
        Cần Triton (Linux / Kaggle).
        cache_path: file lưu đặc trưng text đã mã hoá (từ vựng lớn ~1450 tên x 3 câu mẫu: chậm trên CPU) -> nạp lại lần sau."""
        """model: DensityYOLOWorld (đã encode_text ít nhất một lần để CLIP được nạp). vocab: list tên lớp đối chứng."""
        model.encode_text(["object"])
        self.clip = model.yolo.model.clip_model  # ultralytics.nn.text_model.CLIP
        # CLIP được tạo trên CPU lúc set_classes và không theo model.to(device): chuyển sang device của head đếm
        self.device = next(model.density_head.parameters()).device
        self.net = self.clip.model.to(self.device).eval()
        if self.device.type == "cuda" if half is None else half:
            # fp16 như clip.load trên GPU: convert_weights giữ LayerNorm fp32 (nn.Module.half() làm hỏng LayerNorm
            # của CLIP vốn ép đầu vào về fp32)
            from clip.model import convert_weights
            convert_weights(self.net)
        self.clip.device = self.device
        self.crop, self.context, self.min_side = crop, context, min_side
        self.vocab = list(dict.fromkeys(v.lower() for v in vocab))
        self.cache_path = cache_path
        self._txt = {}
        if cache_path and os.path.exists(cache_path):
            self._txt = {k: v.to(self.device) for k, v in torch.load(cache_path, map_location="cpu").items()}
        self.vocab_feats = self.text(self.vocab)
        self.vocabs = {"base": (self.vocab, self.vocab_feats)}
        self._targets = {}
        self.mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self.compile_batch = compile_batch
        if compile_batch:
            self._encode = torch.compile(self.net.encode_image, mode="reduce-overhead")

    @torch.no_grad()
    def text(self, names):
        miss = [n for n in dict.fromkeys(names) if n not in self._txt]
        for i in range(0, len(miss), 128):  # theo khúc: không đầy bộ nhớ GPU nhỏ
            chunk = miss[i:i + 128]
            tok = self.clip.tokenize([t.format(n) for n in chunk for t in TEMPLATES]).to(self.device)
            f = F.normalize(self.net.encode_text(tok).float(), dim=-1).view(len(chunk), len(TEMPLATES), -1)
            self._txt.update(zip(chunk, F.normalize(f.mean(1), dim=-1)))
        if miss and self.cache_path and len(miss) > 8:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            torch.save({k: v.cpu() for k, v in self._txt.items()}, self.cache_path)
        return torch.stack([self._txt[n] for n in names])

    @torch.no_grad()
    def image_feats(self, images):
        """images (N, 3, crop, crop) [0, 1] -> đặc trưng CLIP chuẩn hoá (N, 512)."""
        x = (images.to(self.device).float() - self.mean) / self.std
        n = len(x)
        if self.compile_batch:  # batch cố định -> một CUDA graph (chia khúc nếu nhiều vùng hơn)
            out = []
            for i in range(0, n, self.compile_batch):
                xb = x[i:i + self.compile_batch]
                pad = self.compile_batch - len(xb)
                if pad:
                    xb = torch.cat([xb, xb.new_zeros(pad, *xb.shape[1:])])
                out.append(self._encode(xb)[:len(xb) - pad].float().clone())
            return F.normalize(torch.cat(out), dim=-1)
        return F.normalize(self.net.encode_image(x).float(), dim=-1)  # encode_image tự ép về dtype của CLIP

    def crops(self, image, boxes):
        """image (3, H, W) [0, 1], boxes xyxy (N, 4) px của image -> vùng cắt (N, 3, crop, crop), nới context lần."""
        c = (boxes[:, :2] + boxes[:, 2:]) / 2
        side = ((boxes[:, 2:] - boxes[:, :2]).amax(1) * self.context).clamp(min=self.min_side)
        b = torch.cat([c - side[:, None] / 2, c + side[:, None] / 2], 1).float()
        return roi_align(image[None].float().to(self.device), [b.to(self.device)], self.crop, aligned=True)

    def add_vocab(self, key, names):
        """Thêm một bộ từ vựng đối chứng khác (vd LVIS) để so sánh điểm có mặt."""
        names = list(dict.fromkeys(n.lower() for n in names))
        self.vocabs[key] = (names, self.text(names))

    def probs(self, feats, prompt, vocab="base", syn_thr=None):
        """feats (N, 512) -> p(prompt) so với từ vựng (N,). Bỏ các tên trùng prompt (kể cả khác số ít / nhiều) và, nếu
        syn_thr, các tên có embedding text giống prompt >= syn_thr (đồng nghĩa / lớp cha: marbles ~ balls) — chúng mô tả
        đúng cùng vật nên không phải đối thủ."""
        key = (prompt.lower(), vocab, syn_thr)
        t = self._targets.get(key)
        if t is None:
            names, vf = self.vocabs[vocab]
            pt = self.text([prompt.lower()])
            keep = torch.tensor([not same_name(v, prompt) for v in names], device=vf.device)
            if syn_thr is not None:
                keep &= (vf @ pt[0]) < syn_thr
            t = self._targets[key] = torch.cat([pt, vf[keep]])
        return (100 * feats @ t.T).softmax(-1)[:, 0]

    @torch.no_grad()
    def crop_probs(self, image, boxes, prompt):
        if len(boxes) == 0:
            return torch.zeros(0, device=self.device)
        return self.probs(self.image_feats(self.crops(image, boxes)), prompt)

    @torch.no_grad()
    def global_prob(self, image, prompt):
        x = F.interpolate(image[None].float(), size=(self.crop, self.crop), mode="bilinear", align_corners=False)
        return float(self.probs(self.image_feats(x), prompt)[0])


def yolo_boxes(dets, k, iou=0.5):
    """dets (1, 4 + 1, N) xywh + score của YOLO-World cho đúng prompt -> top-k box xyxy sau NMS (trên device)."""
    d = dets[0].float()
    xy, wh, sc = d[:2].T, d[2:4].T, d[4]
    b = torch.cat([xy - wh / 2, xy + wh / 2], 1)
    top = sc.topk(min(200, len(sc))).indices
    keep = nms(b[top], sc[top], iou)[:k]
    return b[top][keep]


def coarse_peak_boxes(coarse, k, stride=8, size=640, min_side=32.0, max_side=160.0):
    """min(k, round(số đếm thô)) đỉnh cục bộ mạnh nhất của density thô (1, 1, H, W) -> box vuông quanh tâm ô, xếp theo
    giá trị giảm dần (5 đỉnh đầu ~ exemplar tự chọn của mô hình). Chưa có Scale Head ở bước 1: cạnh ~ khoảng cách TB
    giữa các vật = size / sqrt(số đếm), kẹp trong [min_side, max_side]."""
    d = coarse[0, 0].float()
    count = float(d.sum().clamp(min=1))
    peaks = torch.where(d == F.max_pool2d(d[None, None], 3, 1, 1)[0, 0], d, torch.zeros_like(d))
    vals, idx = peaks.flatten().topk(min(k, max(1, round(count))))
    idx = idx[vals > 0]
    W = d.shape[-1]
    c = torch.stack([(idx % W).float(), (idx // W).float()], 1) * stride + stride / 2
    side = min(max(size / count ** 0.5, min_side), max_side)
    return torch.cat([c - side / 2, c + side / 2], 1)


def select_exemplars(probs, order, n_ex=5, alpha=0.0):
    """Chỉ số exemplar CLIP: tối đa n_ex vùng điểm cao nhất, bỏ vùng có p < alpha x p cao nhất (giữ ít nhất 1)."""
    ex = order[:n_ex]
    if alpha > 0:
        ex = ex[probs[ex] >= alpha * probs[order[0]]]
    return ex


def negative_mask(density, vit_feats, boxes, probs, pos_idx, neg_p=0.05, margin=0.0, size=640):
    """Exemplar âm: vùng CLIP chắc chắn KHÔNG phải prompt (p < neg_p). Mỗi ô của bản đồ DINOv2 so cosine với exemplar
    dương (pos_idx) và âm; ô giống vật âm hơn vật dương quá margin -> xoá density ở đó (cảnh nhiều lớp: mô hình đếm cả
    vật lớp khác). density (1, 1, H, W), vit_feats (1, D, h, w), boxes (N, 4) px ảnh size, probs (N,).
    -> density đã lọc (hoặc nguyên vẹn nếu không có vùng âm)."""
    neg_idx = torch.nonzero(probs < neg_p).flatten()
    if vit_feats is None or len(neg_idx) == 0 or len(pos_idx) == 0:
        return density
    f = F.normalize(vit_feats.float(), dim=1)
    c = (boxes[:, :2] + boxes[:, 2:]) / 2 / size * 2 - 1  # tâm vùng -> [-1, 1]
    protos = F.grid_sample(f, c.to(f.device).view(1, -1, 1, 2), align_corners=False)[0, :, :, 0].T  # (N, D)
    protos = F.normalize(protos, dim=1)
    flat = f[0].flatten(1)  # (D, h*w)
    s_pos = (protos[pos_idx.to(f.device)] @ flat).amax(0)
    s_neg = (protos[neg_idx.to(f.device)] @ flat).amax(0)
    drop = (s_neg - s_pos > margin).float().view(1, 1, *f.shape[2:])
    drop = F.interpolate(drop, size=density.shape[2:], mode="nearest")
    return density * (1 - drop).to(density.dtype)


class VerifiedCounter:
    """CountYOLO + CLIP kiểm tra lớp trong một lượt.

    exemplars: "self" (đỉnh density, như cũ) | "clip" (tâm n_ex vùng giống prompt nhất) | "auto" (dùng exemplar CLIP khi
    các đỉnh mà mô hình tự chọn kém giống prompt hơn hẳn: p TB < switch x p TB của exemplar CLIP -> cảnh nhiều lớp).
    thr: ngưỡng điểm có mặt (score = 0.5 TB top-3 p vùng + 0.5 p cả ảnh); dưới ngưỡng -> đếm 0.
    presence_vocab: bộ từ vựng chấm điểm có mặt (mặc định = vocab, dùng để chọn exemplar). Từ vựng lớn ("all" = base +
    LVIS + Objects365, thêm bằng verifier.add_vocab) giảm lọt khi hỏi sai lớp: lớp thật của ảnh thường có trong đó.
    """

    def __init__(self, model, verifier, thr=0.0, k=12, n_ex=5, exemplars="auto", switch=0.5, vocab="base",
                 syn_thr=None, keep_feats=False, presence_vocab=None, neg_margin=None, neg_p=0.05, ex_alpha=0.0):
        self.m, self.v, self.thr, self.k, self.n_ex = model, verifier, thr, k, n_ex
        self.exemplars, self.switch, self.vocab, self.syn_thr = exemplars, switch, vocab, syn_thr
        self.presence_vocab = presence_vocab or vocab
        # exemplar âm (negative_mask) khi dùng exemplar CLIP (cảnh nhiều lớp); None = tắt
        self.neg_margin, self.neg_p = neg_margin, neg_p
        # chỉ giữ exemplar CLIP có p >= ex_alpha x p cao nhất (1..n_ex vùng): bớt exemplar lớp khác lọt vào cho đủ số
        self.ex_alpha = ex_alpha
        self.keep_feats = keep_feats  # info["feats"]: đặc trưng CLIP (vùng + cả ảnh) để chấm lại với từ vựng khác

    @torch.no_grad()
    def __call__(self, x, cls, exemplars=None, exemplar_points=None, return_aux=False):
        """x (1, 3, 640, 640) [0, 1]. exemplar_points: exemplar người dùng chọn (vẫn kiểm tra có mặt).
        -> density (đã về 0 nếu vắng mặt), info {count, present, score, use_clip, auto_use_clip, ...} [, aux]."""
        mode = exemplars or self.exemplars
        info = {}

        def fn(coarse, dets):
            img = x[0]
            pb = coarse_peak_boxes(coarse, self.k, size=img.shape[-1])
            boxes = torch.cat([pb, yolo_boxes(dets, self.k)])
            whole = F.interpolate(img[None].float(), size=(self.v.crop, self.v.crop), mode="bilinear",
                                  align_corners=False)
            feats = self.v.image_feats(torch.cat([self.v.crops(img, boxes), whole]))
            p = self.v.probs(feats, cls, self.vocab, self.syn_thr)
            if self.keep_feats:
                info["feats"] = feats
            pc, pg = p[:-1], p[-1]
            order = torch.argsort(pc, descending=True)
            ex = select_exemplars(pc, order, self.n_ex, self.ex_alpha)
            pts = (boxes[ex, :2] + boxes[ex, 2:]) / 2
            p_self = pc[:min(self.n_ex, len(pb))].mean() if len(pb) else pc.new_zeros(())
            auto = bool(p_self < self.switch * pc[ex].mean())
            use = exemplar_points is None and (mode == "clip" or (mode == "auto" and auto))
            if self.presence_vocab != self.vocab:  # điểm có mặt theo từ vựng khác, cùng đặc trưng ảnh
                p2 = self.v.probs(feats, cls, self.presence_vocab, self.syn_thr)
                pc2, pg2 = p2[:-1], p2[-1]
                info["score"] = float(0.5 * pc2.topk(min(3, len(pc2))).values.mean() + 0.5 * pg2)
            info.update(score=info.get("score", float(0.5 * pc[order[:3]].mean() + 0.5 * pg)), crop_max=float(pc[order[0]]),
                        crop_top3=float(pc[order[:3]].mean()), global_p=float(pg), use_clip=use, auto_use_clip=auto,
                        boxes=boxes.cpu(), probs=pc.cpu(), clip_points=pts.cpu(), ex_idx=ex.cpu(), n_peaks=len(pb))
            if exemplar_points is not None:
                return exemplar_points
            return [pts] if use else None

        need_aux = return_aux or self.neg_margin is not None
        out = self.m(x, [cls], exemplar_fn=fn, return_aux=need_aux)
        density = out[1]
        if self.neg_margin is not None and info["use_clip"]:
            density = negative_mask(density, out[2].get("vit_feats"), info["boxes"], info["probs"], info["ex_idx"],
                                    self.neg_p, self.neg_margin, x.shape[-1])
        info["present"] = info["score"] >= self.thr
        if not info["present"]:
            density = density * 0
        info["count"] = float(density.float().sum())
        return (density, info, out[2]) if return_aux else (density, info)
