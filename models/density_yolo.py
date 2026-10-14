import contextlib
import torch
import torch.nn as nn
from ultralytics import YOLOWorld
import torch.nn.functional as F

# Head dự đoán density đã nhân với hệ số này (giá trị ~O(1) thay vì ~1e-3/pixel),
# tránh việc ReLU đầu ra "chết" ở 0. forward() chia lại để trả về density theo đơn vị số lượng.
DENSITY_SCALE: float = 100.0

# P3 (128) + Text Similarity Map (1) + Edge Map (1)
HEAD_IN_CHANNELS: int = 130

# Self-Exemplar: số exemplar giả lấy từ đỉnh density thô, số chiều không gian so khớp,
# và 2 kênh similarity (mean + max qua K exemplar) mỗi mức feature
NUM_EXEMPLARS: int = 5
EXEMPLAR_DIM: int = 64
SIM_CHANNELS: int = 2
# Đỉnh density thô yếu hơn tỉ lệ này so với đỉnh mạnh nhất coi là nền, không dùng làm exemplar
PEAK_MIN_RATIO: float = 0.1

# Similarity-centric Refine Head: P3 bị nén + dropout mạnh để head không học thuộc ngoại hình lớp train,
# buộc dựa vào tín hiệu không phụ thuộc lớp (similarity P3+P4, density thô, text, edge)
P3_CHANNELS: int = 128
P4_CHANNELS: int = 256
P3_COMPRESS: int = 16
P3_DROPOUT: float = 0.5
# P3 nén (16) + text (1) + edge (1) + density thô (1) + similarity P3 (2) + similarity P4 (2)
SIMCENTRIC_IN_CHANNELS: int = P3_COMPRESS + 3 + 2 * SIM_CHANNELS

# Scale Head: dự đoán log2(kích thước vật, px trong ảnh 640) tại mỗi ô P3 từ P3/P4/P5 (mỗi mức nén 32 kênh)
# + similarity P3 + density thô. Sai số đếm gần như chỉ do vật nhỏ (<~40px) mà P3 stride 8 không phân giải
# được, nên biết kích thước vật cho phép zoom đúng vùng về kích thước chuẩn (xem utils/scale_zoom.py).
P5_CHANNELS: int = 512
SCALE_COMPRESS: int = 32
SCALE_IN_CHANNELS: int = 3 * SCALE_COMPRESS + SIM_CHANNELS + 1
SCALE_INIT_LOG2: float = 5.5  # ~45px, cỡ vật điển hình của FSC-147 ở đầu vào 640

# Nhánh ViT (arch "vit"): DINOv2 đóng băng cho feature thị giác không phụ thuộc lớp. Ảnh 640 được resize về 560
# -> lưới patch 40x40 (patch 14), đúng bằng lưới P4 nên vị trí exemplar trên P3 chỉ cần chia 2.
VIT_NAME: str = "facebook/dinov2-small"   # mặc định; checkpoint lưu vit_name / vit_input riêng
VIT_INPUT: int = 560
VIT_PROJ: int = 64          # kênh ViT chiếu xuống trước khi ghép vào Density Head
VIT_DROPOUT: float = 0.1
# Refine Head arch "vit": simcentric + similarity trên feature ViT (2)
VIT_REFINE_IN_CHANNELS: int = SIMCENTRIC_IN_CHANNELS + SIM_CHANNELS
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Bỏ đóng băng một phần: layer 12-22 của yolov8s-world là neck (C2fAttn có text-attention, ~6M tham số);
# backbone (0-9) và WorldDetect (23) giữ nguyên để region-text vẫn so khớp được với CLIP text embedding
NECK_FIRST_LAYER: int = 12


class SobelEdgeExtractor(nn.Module):
    """
    Trích xuất biên ảnh gốc (Edge Map) bằng bộ lọc Sobel cố định.
    Giúp Density Head bám sát ranh giới vật thể khi đếm đám đông.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        # Chuyển ảnh RGB sang Grayscale
        x_gray = x.mean(dim=1, keepdim=True)
        # Pad để giữ nguyên kích thước
        x_pad = F.pad(x_gray, (1, 1, 1, 1), mode='reflect')
        gx = F.conv2d(x_pad, self.sobel_x.to(dtype=x.dtype))
        gy = F.conv2d(x_pad, self.sobel_y.to(dtype=x.dtype))
        # Tính độ lớn Gradient (Edge Magnitude)
        edges = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        # Chuẩn hóa về [0, 1] trên mỗi ảnh trong batch
        batch_size = edges.size(0)
        edges_flat = edges.view(batch_size, -1)
        max_val = edges_flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        max_val = torch.clamp(max_val, min=1e-6)
        edges = edges / max_val
        return edges


class DensityHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Mạng CNN nhẹ để dự đoán Density Map (đã nhân DENSITY_SCALE) từ Feature Map
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)  # Trả về 1 kênh heatmap
        )
        # Chỉ khởi tạo nhỏ lớp cuối; các lớp ẩn giữ Kaiming mặc định để gradient không bị triệt tiêu
        last = self.conv_layers[-1]
        nn.init.normal_(last.weight, std=0.01)
        nn.init.constant_(last.bias, 0)

    def forward(self, x):
        # ReLU đảm bảo mật độ luôn >= 0
        return F.relu(self.conv_layers(x))


class SelfExemplarMatcher(nn.Module):
    """
    Self-Exemplar: lấy K vị trí làm exemplar giả, so khớp cosine giữa feature tại đó với toàn bộ feature map.

    Vị trí exemplar: đỉnh cục bộ của density thô (zero-shot), hoặc điểm cho trước (GT points khi train /
    tâm exemplar box khi đánh giá few-shot). Feature được chiếu qua Conv1x1 học được để không gian so khớp
    không bị giới hạn bởi P3 frozen.
    """
    def __init__(self, in_channels=128, dim=EXEMPLAR_DIM, k=NUM_EXEMPLARS):
        super().__init__()
        self.k = k
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=1)

    @torch.no_grad()
    def peak_locations(self, density):
        """Top-K đỉnh cục bộ (max-pool 3x3) của density (B, 1, H, W) -> chỉ số phẳng (B, K).
        Đỉnh yếu hơn PEAK_MIN_RATIO x đỉnh mạnh nhất (hoặc ô 0 khi ít hơn K đỉnh) là nền, không phải vật:
        thay bằng cách lặp lại các đỉnh thật."""
        pooled = F.max_pool2d(density, 3, stride=1, padding=1)
        peaks = torch.where(density == pooled, density, torch.zeros_like(density))
        vals, idx = peaks.flatten(1).topk(self.k, dim=1)
        n_real = (vals > PEAK_MIN_RATIO * vals[:, :1]).sum(1, keepdim=True).clamp(min=1)
        return idx.gather(1, torch.arange(self.k, device=idx.device).unsqueeze(0) % n_real)

    def point_locations(self, points, size, stride, device):
        """Điểm [x, y] (pixel ảnh vào) -> chỉ số phẳng trên lưới; lấy mẫu ngẫu nhiên K điểm, lặp lại nếu thiếu."""
        H, W = size
        idx = []
        for p in points:
            p = p.to(device)
            sel = torch.randperm(len(p), device=device)[:self.k]
            sel = sel.repeat((self.k + len(sel) - 1) // len(sel))[:self.k]
            gx = (p[sel, 0] / stride).long().clamp(0, W - 1)
            gy = (p[sel, 1] / stride).long().clamp(0, H - 1)
            idx.append(gy * W + gx)
        return torch.stack(idx)

    @staticmethod
    def downsample_locations(locations, size, factor):
        """Chỉ số phẳng trên lưới size=(H, W) -> chỉ số trên lưới thô hơn `factor` lần (vd P3 -> P4)."""
        W = size[1]
        y, x = locations // W, locations % W
        return (y // factor) * (W // factor) + x // factor

    @staticmethod
    def resize_locations(locations, size_from, size_to):
        """Chỉ số phẳng trên lưới size_from -> ô tương ứng trên lưới size_to (tỉ lệ bất kỳ, vd P3 80 -> ViT 60)."""
        (H, W), (h, w) = size_from, size_to
        y, x = locations // W, locations % W
        return (y * h // H).clamp(max=h - 1) * w + (x * w // W).clamp(max=w - 1)

    def forward(self, feats, locations):
        """feats (B, C, H, W), locations (B, K) -> similarity map (B, 2, H, W): [mean, max] qua K exemplar."""
        f = F.normalize(self.proj(feats), dim=1)
        B, D, H, W = f.shape
        flat = f.flatten(2)                                                       # (B, D, HW)
        protos = flat.gather(2, locations.unsqueeze(1).expand(B, D, -1))          # (B, D, K)
        sim = torch.einsum('bdk,bdn->bkn', protos, flat).reshape(B, -1, H, W)     # (B, K, H, W)
        return torch.cat([sim.mean(1, keepdim=True), sim.amax(1, keepdim=True)], dim=1)


class FrozenViT(nn.Module):
    """DINOv2 đóng băng: ảnh [0, 1] (B, 3, H, W) -> patch token (B, dim, size/14, size/14)."""
    def __init__(self, name=VIT_NAME, size=VIT_INPUT):
        super().__init__()
        from transformers import Dinov2Model
        self.vit = Dinov2Model.from_pretrained(name).eval()
        for p in self.vit.parameters():
            p.requires_grad = False
        self.name, self.size = name, size
        self.dim = self.vit.config.hidden_size
        self.grid = size // self.vit.config.patch_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        return super().train(False)  # luôn eval

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
        x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        tokens = self.vit(pixel_values=x).last_hidden_state
        tokens = tokens[:, -self.grid * self.grid:]  # bỏ CLS (và register token nếu có)
        return tokens.transpose(1, 2).reshape(x.shape[0], -1, self.grid, self.grid)


class ScaleHead(nn.Module):
    """Bản đồ log2(kích thước vật) trên lưới P3. P5 nhìn được vật lớn, P3 vật nhỏ -> dùng cả ba mức."""
    def __init__(self):
        super().__init__()
        self.compress = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, SCALE_COMPRESS, kernel_size=1), nn.BatchNorm2d(SCALE_COMPRESS),
                          nn.ReLU(inplace=True))
            for c in (P3_CHANNELS, P4_CHANNELS, P5_CHANNELS)])
        self.head = nn.Sequential(
            nn.Conv2d(SCALE_IN_CHANNELS, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1))
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.constant_(self.head[-1].bias, SCALE_INIT_LOG2)

    def forward(self, feats, sim, coarse):
        size = feats[0].shape[2:]
        levels = [F.interpolate(m(f), size=size, mode='bilinear', align_corners=False) if f.shape[2:] != size
                  else m(f) for m, f in zip(self.compress, feats)]
        return self.head(torch.cat(levels + [sim, coarse], dim=1))


class DensityYOLOWorld(nn.Module):
    """
    YOLO-World (frozen, hoặc mở neck bằng unfreeze_yolo) + Density Head trên P3 (stride 8), có điều kiện theo text prompt.

    Đầu vào Density Head = [P3 (128), Text Similarity Map (1), Edge Map (1)]:
    - Text Similarity Map: xác suất region-text của chính YOLO-World (max qua P3/P4/P5,
      upsample về lưới P3). Đây là kênh cho Head biết cần đếm LỚP NÀO.

    Self-Exemplar (2 bước): Density Head cho density thô -> K đỉnh làm exemplar giả -> Similarity Map
    thị giác -> Refine Head cho density cuối. Text của YOLO-World yếu với nhiều lớp FSC-147, còn vật cần
    đếm luôn lặp lại trong ảnh nên độ giống thị giác với chính các vật đó là tín hiệu mạnh hơn.

    arch (suy ra từ checkpoint khi load_head):
    - "coarse"     : chỉ Density Head (checkpoint trước Self-Exemplar)
    - "selfex"     : Refine Head trên [130 kênh + similarity P3]
    - "simcentric" : Refine Head trên [P3 nén 16 kênh + dropout, text, edge, density thô,
                     similarity P3 + P4] — lớp val/test không có trong train, nên head được buộc dựa vào
                     tín hiệu không phụ thuộc lớp thay vì học thuộc ngoại hình trong P3 (mặc định)
    - "vit"        : simcentric + DINOv2 đóng băng (FrozenViT): feature ViT chiếu 64 kênh ghép vào Density
                     Head, và thêm similarity Self-Exemplar trên feature ViT cho Refine Head

    Scale Head (tuỳ chọn, có khi checkpoint lưu "scale_head"): bản đồ log2 kích thước vật, dùng cho
    Canonical-Scale Zoom lúc suy luận.
    """
    def __init__(self, yolo_weights="yolov8s-world.pt", arch="simcentric", vit_name=VIT_NAME, vit_input=VIT_INPUT):
        super().__init__()
        print(f"Khởi tạo YOLO-World từ {yolo_weights}")
        self.yolo = YOLOWorld(yolo_weights)
        # Mỗi ảnh được đếm theo đúng 1 text prompt -> nc = 1
        self.yolo.set_classes(["object"])

        # Đóng băng (Freeze) toàn bộ YOLO-World
        for param in self.yolo.parameters():
            param.requires_grad = False

        # Hook để bắt Feature Map P3, P4, P5
        self.features = None
        self.features_p4 = None
        self.features_p5 = None
        self._register_hook()
        self._text_cache = {}
        self.yolo_unfreeze_from = None  # None: YOLO-World đóng băng hoàn toàn
        self._yolo_anchor = {}

        # Edge Extractor (không có tham số huấn luyện)
        self.edge_extractor = SobelEdgeExtractor()
        self.vit_extractor = None
        self.vit_name, self.vit_input = vit_name, vit_input
        self.exemplar_matcher = SelfExemplarMatcher(in_channels=P3_CHANNELS)
        self._build_refine(arch)
        self.scale_head = ScaleHead()

    def _build_refine(self, arch):
        self.arch = arch
        self.exemplar_matcher_p4 = self.p3_compress = self.vit_proj = self.exemplar_matcher_vit = None
        use_vit = arch == "vit"
        self.density_head = DensityHead(in_channels=HEAD_IN_CHANNELS + (VIT_PROJ if use_vit else 0))
        if use_vit and (self.vit_extractor is None or
                        (self.vit_extractor.name, self.vit_extractor.size) != (self.vit_name, self.vit_input)):
            self.vit_extractor = FrozenViT(self.vit_name, self.vit_input)
        elif not use_vit:
            self.vit_extractor = None  # checkpoint không có nhánh ViT
        if arch == "selfex":
            self.refine_head = DensityHead(in_channels=HEAD_IN_CHANNELS + SIM_CHANNELS)
        elif arch in ("simcentric", "vit"):
            self.exemplar_matcher_p4 = SelfExemplarMatcher(in_channels=P4_CHANNELS)
            self.p3_compress = nn.Sequential(nn.Conv2d(P3_CHANNELS, P3_COMPRESS, kernel_size=1),
                                             nn.ReLU(inplace=True), nn.Dropout2d(P3_DROPOUT))
            self.refine_head = DensityHead(in_channels=VIT_REFINE_IN_CHANNELS if use_vit
                                           else SIMCENTRIC_IN_CHANNELS)
            if use_vit:
                vit_dim = self.vit_extractor.dim
                self.vit_proj = nn.Sequential(nn.Conv2d(vit_dim, VIT_PROJ, kernel_size=1), nn.BatchNorm2d(VIT_PROJ),
                                              nn.ReLU(inplace=True), nn.Dropout2d(VIT_DROPOUT))
                self.exemplar_matcher_vit = SelfExemplarMatcher(in_channels=vit_dim)
        else:
            self.refine_head = None

    def unfrozen_yolo_layers(self):
        """{chỉ số: layer} của YOLO-World đang được train (rỗng khi đóng băng hoàn toàn)."""
        if self.yolo_unfreeze_from is None:
            return {}
        layers = self.yolo.model.model
        return {i: layers[i] for i in range(self.yolo_unfreeze_from, len(layers) - 1)}

    def unfreeze_yolo(self, from_layer=NECK_FIRST_LAYER):
        """Cho phép train layer from_layer..(Detect - 1). BN vẫn ở eval (thống kê không đổi).
        Lưu bản sao trọng số gốc làm mốc cho L2-SP (yolo_l2sp)."""
        self.yolo_unfreeze_from = from_layer
        self._yolo_anchor = {}
        for i, layer in self.unfrozen_yolo_layers().items():
            for name, p in layer.named_parameters():
                p.requires_grad = True
                self._yolo_anchor[f"{i}.{name}"] = p.detach().clone()

    def yolo_parameters(self):
        return [p for layer in self.unfrozen_yolo_layers().values() for p in layer.parameters()]

    def yolo_l2sp(self):
        """Tổng bình phương độ lệch so với trọng số YOLO-World gốc: giữ neck gần bản pretrained để không
        mất khả năng open-vocabulary (lớp val/test không có trong train)."""
        total = 0.0
        for i, layer in self.unfrozen_yolo_layers().items():
            for name, p in layer.named_parameters():
                total = total + (p.float() - self._yolo_anchor[f"{i}.{name}"].to(p.device)).pow(2).sum()
        return total

    def _register_hook(self):
        def hook_fn(module, input):
            # input[0] là list feature maps [P3, P4, P5]
            feats = input[0]
            if isinstance(feats, (list, tuple)):
                self.features, self.features_p4, self.features_p5 = feats[0], feats[1], feats[2]
            else:
                self.features = feats

        detect_layer = self.yolo.model.model[-1]
        detect_layer.register_forward_pre_hook(hook_fn)

    @torch.no_grad()
    def encode_text(self, class_names):
        """CLIP text embedding cho từng tên lớp (có cache). Trả về (B, 1, 512)."""
        missing = [n for n in dict.fromkeys(class_names) if n not in self._text_cache]
        if missing:
            # CLIP được tạo lúc set_classes (trên CPU); sau .to(device) tokenizer vẫn nhớ device cũ
            clip = getattr(self.yolo.model, "clip_model", None)
            if clip is not None:
                clip.device = next(clip.parameters()).device
            feats =self.yolo.model.get_text_pe(missing)[0]  # (len(missing), 512)
            for n, f in zip(missing, feats):
                self._text_cache[n] = f.float().cpu()  # predict() tự chuyển sang device/dtype của ảnh
        return torch.stack([self._text_cache[n] for n in class_names]).unsqueeze(1)

    def _text_similarity_map(self, preds, size):
        """Gom score map (sigmoid) của P3/P4/P5 về lưới P3, lấy max theo level và theo lớp."""
        feats, scores = preds["feats"], preds["scores"].sigmoid()  # scores: (B, nc, sum(H*W))
        maps, start = [], 0
        for f in feats:
            h, w = f.shape[2:]
            s = scores[:, :, start:start + h * w].reshape(scores.shape[0], -1, h, w)
            maps.append(F.interpolate(s, size=size, mode='nearest'))
            start += h * w
        return torch.stack(maps, 0).amax(dim=(0, 2)).unsqueeze(1)  # (B, 1, H3, W3)

    def forward(self, x, class_names=None, exemplar_points=None, return_aux=False):
        """
        Args:
            x: ảnh (B, 3, 640, 640), giá trị [0, 1]
            class_names: list[str] độ dài B (mỗi ảnh một prompt). None -> dùng prompt đã set_classes.
            exemplar_points: list độ dài B, mỗi phần tử None (dùng đỉnh density thô) hoặc tensor (N, 2)
                các điểm [x, y] pixel của x làm exemplar. None -> toàn bộ batch dùng đỉnh density thô.
            return_aux: trả thêm dict {"coarse": density thô bước 1, "log_size": bản đồ log2 kích thước vật
                (B, 1, H/8, W/8) hoặc None nếu không có Scale Head}.

        Returns:
            dets: (B, 4 + nc, N) box xywh (tọa độ pixel của x) + score, đã decode
            density_map: (B, 1, H/8, W/8) theo đơn vị số lượng (sum ≈ count)
            [aux]: khi return_aux=True
        """
        detect = self.yolo.model.model[-1]
        if class_names is not None:
            txt_feats = self.encode_text(class_names)
        else:
            txt_feats = self.yolo.model.txt_feats
        detect.nc = txt_feats.shape[1]

        # 1. YOLO-World: không cần lưu graph khi đóng băng hoàn toàn
        with torch.no_grad() if self.yolo_unfreeze_from is None else contextlib.nullcontext():
            dets, preds = self.yolo.model.predict(x, txt_feats=txt_feats)
            if self.features is None:
                raise RuntimeError("Không bắt được Feature Map từ Hook!")
            p3 = self.features
            text_map = self._text_similarity_map(preds, p3.shape[2:])
            edges = self.edge_extractor(x)
            edges_down = F.interpolate(edges, size=p3.shape[2:], mode='bilinear', align_corners=False)
            fused_features = torch.cat([p3, text_map.to(p3.dtype), edges_down.to(p3.dtype)], dim=1)

        # Nhánh ViT: feature DINOv2 (lưới 40x40) chiếu 64 kênh, upsample về P3, ghép vào đầu vào Density Head
        vit = None
        head_in = fused_features
        if self.vit_extractor is not None:
            vit = self.vit_extractor(x).to(fused_features.dtype)
            vit_up = F.interpolate(self.vit_proj(vit).float(), size=p3.shape[2:], mode='bilinear',
                                   align_corners=False)
            head_in = torch.cat([fused_features, vit_up.to(fused_features.dtype)], dim=1)

        # 2. Density Head: density thô
        coarse = self.density_head(head_in) / DENSITY_SCALE
        if self.arch == "coarse":
            return (dets, coarse, {"coarse": coarse, "log_size": None}) if return_aux else (dets, coarse)

        # 3. Self-Exemplar: vị trí exemplar -> similarity map -> Refine Head
        locations = self.exemplar_matcher.peak_locations(coarse.detach().float())
        if exemplar_points is not None:
            given = [i for i, p in enumerate(exemplar_points) if p is not None and len(p) > 0]
            if given:
                locations[given] = self.exemplar_matcher.point_locations(
                    [exemplar_points[i] for i in given], p3.shape[2:], x.shape[-1] / p3.shape[-1], x.device)
        sim = self.exemplar_matcher(p3, locations).to(fused_features.dtype)
        if self.arch == "selfex":
            refine_in = torch.cat([fused_features, sim], dim=1)
        else:
            p4 = self.features_p4
            loc4 = SelfExemplarMatcher.downsample_locations(locations, p3.shape[2:], p3.shape[-1] // p4.shape[-1])
            sim4 = F.interpolate(self.exemplar_matcher_p4(p4, loc4).float(), size=p3.shape[2:],
                                 mode='bilinear', align_corners=False).to(sim.dtype)
            parts = [self.p3_compress(p3), fused_features[:, P3_CHANNELS:],
                     (coarse.detach() * DENSITY_SCALE).to(sim.dtype), sim, sim4]
            if vit is not None:
                locv = SelfExemplarMatcher.resize_locations(locations, p3.shape[2:], vit.shape[2:])
                parts.append(F.interpolate(self.exemplar_matcher_vit(vit, locv).float(), size=p3.shape[2:],
                                           mode='bilinear', align_corners=False).to(sim.dtype))
            refine_in = torch.cat(parts, dim=1)
        density_map = self.refine_head(refine_in) / DENSITY_SCALE
        if not return_aux:
            return dets, density_map

        # 4. Scale Head: log2 kích thước vật (chỉ cần khi train hoặc Canonical-Scale Zoom)
        log_size = None
        if self.scale_head is not None:
            log_size = self.scale_head([p3, self.features_p4, self.features_p5], sim,
                                       (coarse.detach() * DENSITY_SCALE).to(sim.dtype))
        return dets, density_map, {"coarse": coarse, "log_size": log_size}

    def trainable_modules(self):
        mods = {"density_head": self.density_head, "exemplar_matcher": self.exemplar_matcher,
                "exemplar_matcher_p4": self.exemplar_matcher_p4, "p3_compress": self.p3_compress,
                "refine_head": self.refine_head, "scale_head": getattr(self, "scale_head", None),
                "vit_proj": self.vit_proj, "exemplar_matcher_vit": self.exemplar_matcher_vit}
        return nn.ModuleDict({k: v for k, v in mods.items() if v is not None})

    def train(self, mode=True):
        # Không gọi super().train(): nn.Module sẽ gọi YOLOWorld.train(), vốn là hàm
        # khởi động Ultralytics trainer. YOLO luôn ở eval (BN không cập nhật, Detect trả về box đã decode).
        self.training = mode
        self.trainable_modules().train(mode)
        self.edge_extractor.train(mode)
        self.yolo.model.eval()
        return self

    def train_mode(self):
        self.train(True)

    def eval_mode(self):
        self.train(False)

    def save_head(self, path, **meta):
        state = {k: m.state_dict() for k, m in self.trainable_modules().items()}
        if self.yolo_unfreeze_from is not None:
            state["yolo_unfreeze_from"] = self.yolo_unfreeze_from
            state["yolo_layers"] = {i: layer.state_dict() for i, layer in self.unfrozen_yolo_layers().items()}
        if self.vit_extractor is not None:
            state["vit_name"], state["vit_input"] = self.vit_extractor.name, self.vit_extractor.size
        torch.save({**state, "arch": self.arch, "in_channels": HEAD_IN_CHANNELS,
                    "density_scale": DENSITY_SCALE, **meta}, path)

    def load_head(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        if "density_head" not in ckpt or ckpt.get("in_channels") != HEAD_IN_CHANNELS:
            raise ValueError(f"Checkpoint {path} thuộc kiến trúc cũ (không có text map). Cần train lại.")
        # Checkpoint cũ không lưu arch: chỉ Density Head -> "coarse", có Refine Head -> "selfex"
        arch = ckpt.get("arch", "selfex" if "refine_head" in ckpt else "coarse")
        device = next(self.density_head.parameters()).device
        # Checkpoint "vit" đầu tiên không lưu cấu hình ViT -> mặc định dinov2-small @ 560
        self.vit_name, self.vit_input = ckpt.get("vit_name", VIT_NAME), ckpt.get("vit_input", VIT_INPUT)
        self._build_refine(arch)
        # Scale Head chỉ có ở checkpoint từ v6; arch "selfex"/"coarse" không có similarity cho nó
        self.scale_head = ScaleHead() if "scale_head" in ckpt else None
        for k, m in self.trainable_modules().items():
            if k in ckpt:
                m.load_state_dict(ckpt[k])
            elif arch != "coarse":  # "coarse" không dùng exemplar_matcher
                raise KeyError(f"Checkpoint {path} thiếu '{k}' cho arch={arch}")
        self.trainable_modules().to(device)
        if self.vit_extractor is not None:
            self.vit_extractor.to(device)
        # Checkpoint có neck YOLO-World đã fine-tune: nạp đè lên trọng số pretrained
        for i, sd in ckpt.get("yolo_layers", {}).items():
            self.yolo.model.model[int(i)].load_state_dict(sd)
        return ckpt
