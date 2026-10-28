"""
Giao diện CountYOLO (Gradio): đếm vật theo text trên ảnh và video.

    .venv/Scripts/python app/app.py            # mở http://127.0.0.1:7860 (hoặc cổng trống kế tiếp)
    python app/app.py --checkpoint checkpoints/<file>.pth --port 7861 --share
"""
import argparse
import glob
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

from models.density_yolo import DensityYOLOWorld  # noqa: E402
from utils.clip_verify import ClipVerifier, VerifiedCounter, default_vocab, lvis_names, o365_names  # noqa: E402
from utils.scale_zoom import _peak_boxes, density_tiled_count  # noqa: E402
from video.count_light import TRACKERS, count_video, frame_list  # noqa: E402
from engines.sam3_counter import DEFAULT_WEIGHTS as SAM3_WEIGHTS  # noqa: E402

# exnoise: exmix2 fine-tune thêm với exemplar nhiễu (FSC-147 test 17.89, chia ô 16.53), dùng cùng CLIP kiểm tra lớp
DEFAULT_CKPT = next((p for p in (os.path.join(ROOT, "checkpoints", f) for f in (
    "density_head_best_vit_exnoise.pth", "density_head_best_vit_exmix2_ep20.pth",
    "density_head_best_vit_seed3_ep19_valmae21.44.pth")) if os.path.exists(p)),
    os.path.join(ROOT, "checkpoints", "density_head_best_vit_seed3_ep19_valmae21.44.pth"))
# Ngưỡng điểm có mặt của CLIP (utils/clip_verify.VerifiedCounter), chọn trên FSC-147 VAL bằng tools/clip_verify_eval.py
# (nhỏ nhất sao cho đếm TB khi hỏi sai lớp <= 0.5) với checkpoint exmix2
# thiend27/countyolo-density-vit-exnoise (checkpoint exnoise, K=6, điểm có mặt theo từ vựng base + LVIS + Objects365,
# ngưỡng = nhỏ nhất giữ val đếm TB sai lớp <= 0.75): test hỏi sai lớp 0.77, chặn nhầm 3.0%, MAE 18.38
CLIP_THR = 0.0198
CLIP_K = 6
SIZE = 640
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = {"path": None, "model": None}
_SAM3 = {"counter": None}
_GDINO = {"counter": None}
_VERIFY = {"model": None, "vc": None}
ENGINES = ["SAM 3", "Grounding DINO", "CountYOLO"]


def get_sam3():
    """Engine SAM 3 (ổn định, nghe đúng text). Lỗi rõ ràng nếu chưa có weights/sam3.pt."""
    if _SAM3["counter"] is None:
        from engines.sam3_counter import SAM3Counter
        try:
            _SAM3["counter"] = SAM3Counter(SAM3_WEIGHTS)
        except FileNotFoundError as e:
            raise gr.Error(str(e))
    return _SAM3["counter"]


def get_grounding_dino():
    """Engine Grounding DINO (NMS-Free, Query-based Transformer)."""
    if _GDINO["counter"] is None:
        from engines.grounding_dino_counter import GroundingDinoCounter
        _GDINO["counter"] = GroundingDinoCounter()
    return _GDINO["counter"]


def draw_boxes(img, boxes, ids=None):
    """Vẽ box (màu theo id track nếu có)."""
    out = np.array(img).copy()
    t = max(1, int(min(out.shape[:2]) / 300))
    for k, b in enumerate(boxes):
        h = ((ids[k] if ids is not None else 0) * 47) % 180
        color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]) \
            if ids is not None else (0, 255, 0)
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, t)
    return out


def get_model(ckpt):
    if _MODEL["path"] != ckpt:
        m = DensityYOLOWorld(os.path.join(ROOT, "yolov8s-world.pt") if os.path.exists(
            os.path.join(ROOT, "yolov8s-world.pt")) else "yolov8s-world.pt")
        m.load_head(ckpt, map_location="cpu")
        m.to(DEVICE).eval_mode()
        _MODEL.update(path=ckpt, model=m)
    return _MODEL["model"]


def get_verifier(model):
    """CountYOLO + CLIP kiểm tra lớp trong một lượt (dùng lại CLIP mà YOLO-World đã nạp)."""
    if _VERIFY["model"] is not model:
        v = ClipVerifier(model, default_vocab(), cache_path=os.path.join(ROOT, "outputs", "clip_text_cache.pt"))
        v.add_vocab("all", default_vocab() + lvis_names() + o365_names())
        _VERIFY.update(model=model, vc=VerifiedCounter(model, v, thr=CLIP_THR, k=CLIP_K, presence_vocab="all"))
    return _VERIFY["vc"]


def colorize(m, size):
    """Bản đồ (H, W) -> ảnh RGB colormap jet, resize về size (w, h)."""
    m = np.asarray(m, dtype=np.float32)
    m = (m - m.min()) / max(float(m.max() - m.min()), 1e-6)
    rgb = (plt.get_cmap("jet")(m)[..., :3] * 255).astype(np.uint8)
    return cv2.resize(rgb, size, interpolation=cv2.INTER_NEAREST)


def draw_points(img, boxes, ids=None, radius=None):
    """Vẽ chấm tại tâm box; tô màu theo id track nếu có."""
    out = np.array(img).copy()
    r = radius or max(3, int(min(out.shape[:2]) / 120))
    for k, b in enumerate(boxes):
        cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
        if ids is None:
            color = (255, 230, 0)
        else:
            h = (ids[k] * 47) % 180
            color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2RGB)[0, 0])
        cv2.circle(out, (cx, cy), r, color, -1)
        cv2.circle(out, (cx, cy), r, (0, 0, 0), 1)
    return out


# ----------------------------------------------------------------- Ảnh
@torch.no_grad()
def count_image(engine, image, text, exemplars, verify, use_tiling, ttn_thr, min_mass, ckpt,
                gd_box_thr=0.10, gd_text_thr=0.10):
    if image is None or not text.strip():
        raise gr.Error("Cần ảnh và tên vật cần đếm (vd: apples, people, penguins).")
    if engine == "SAM 3":
        r = get_sam3().count_image(Image.fromarray(image), text.strip())
        g = r["grid"]
        tiled = f" · chia ảnh {g}×{g} (gần trần 200 vật/lượt)" if g > 1 else ""
        summary = (f"## Đếm được: **{r['count']}** \"{text.strip()}\"  (SAM 3)\n"
                   f"- {r['count']} đối tượng được phát hiện + tách mask{tiled}\n"
                   f"- Thời gian: {r['ms']:.0f} ms\n"
                   f"- SAM 3 trả về box từng vật, không có density / bản đồ trung gian (chọn engine CountYOLO để xem)")
        return summary, draw_boxes(image, r["boxes"]), gr.update(value=None, visible=False), \
            gr.update(value=[], visible=False)
    if engine == "Grounding DINO":
        gd = get_grounding_dino()
        r = gd.count_image(Image.fromarray(image), text.strip(), box_threshold=gd_box_thr, text_threshold=gd_text_thr)
        summary = (f"## Đếm được: **{r['count']}** \"{text.strip()}\"  (Grounding DINO)\n"
                   f"- {r['count']} đối tượng được phát hiện (NMS-Free Transformer Queries)\n"
                   f"- Thời gian: {r['ms']:.0f} ms trên {DEVICE.type.upper()}\n"
                   f"- Ngưỡng tự tin: box_threshold={gd_box_thr:.2f}, text_threshold={gd_text_thr:.2f}\n"
                   f"- Cơ chế NMS-Free: bắt trọn các vật dày đặc chồng lấn mà không bị NMS triệt tiêu")
        return summary, draw_boxes(image, r["boxes"]), gr.update(value=None, visible=False), \
            gr.update(value=[], visible=False)
    model = get_model(ckpt)
    img = Image.fromarray(image).convert("RGB")
    W, H = img.size
    t0 = time.perf_counter()
    x = TF.to_tensor(img.resize((SIZE, SIZE)))[None].to(DEVICE)
    ex = None
    if exemplars:
        ex = [torch.tensor([[px * SIZE / W, py * SIZE / H] for px, py in exemplars], dtype=torch.float32)]
    vinfo = None
    if verify:
        dens, vinfo, aux = get_verifier(model)(x, text.strip(), exemplar_points=ex, return_aux=True)
    else:
        _, dens, aux = model(x, [text.strip()], exemplar_points=ex, return_aux=True)
    d = dens[0, 0].float()
    count = float(d.sum())
    tiled = False
    if use_tiling and not exemplars and (vinfo is None or vinfo["present"]):
        # exemplar CLIP chọn ở lượt toàn ảnh -> dùng lại prototype cho các ô
        protos = aux["protos"] if vinfo is not None and vinfo["use_clip"] else None
        count, info = density_tiled_count(model, img, text.strip(), DEVICE, count_thr=ttn_thr, grid=2, protos=protos)
        tiled = info["tiled"]
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000

    ls = aux["log_size"][0, 0].float() if aux["log_size"] is not None else None
    boxes, _ = _peak_boxes(d, ls, float(d.sum()), min_mass=min_mass)
    boxes = (boxes * torch.tensor([W / SIZE, H / SIZE] * 2)).numpy()
    points = draw_points(img, boxes)
    for px, py in exemplars or []:
        cv2.drawMarker(points, (int(px), int(py)), (255, 0, 0), cv2.MARKER_STAR, 24, 3)
    if vinfo is not None and vinfo["use_clip"]:
        for px, py in (vinfo["clip_points"] * torch.tensor([W / SIZE, H / SIZE])).tolist():
            cv2.drawMarker(points, (int(px), int(py)), (0, 255, 255), cv2.MARKER_CROSS, 24, 3)
    heat = (0.45 * np.array(img) + 0.55 * colorize(d.cpu(), (W, H))).astype(np.uint8)
    maps = [(colorize(aux["coarse"][0, 0].float().cpu(), (320, 320)), "Density thô (bước 1)"),
            (colorize(aux["text_map"][0, 0].float().cpu(), (320, 320)), "Text Map (YOLO-World)"),
            (colorize(aux["sim_p3"][0, -1].float().cpu(), (320, 320)), "Similarity P3")]
    if aux.get("sim_vit") is not None:
        maps.append((colorize(aux["sim_vit"][0, -1].float().cpu(), (320, 320)), "Similarity DINOv2"))
    loc = aux["exemplar_locations"][0].cpu().numpy()
    g = d.shape[-1]
    note = "exemplar do bạn chọn" if exemplars else \
        "exemplar do CLIP chọn (dấu + xanh): cảnh có nhiều loại vật" if vinfo is not None and vinfo["use_clip"] else \
        f"exemplar tự chọn tại ô {[(int(i % g), int(i // g)) for i in loc]}"
    if vinfo is not None:
        verdict = f"có \"{text.strip()}\" trong ảnh" if vinfo["present"] else \
            f"**KHÔNG có \"{text.strip()}\"** trong ảnh → đếm 0"
        if not vinfo["present"]:
            note = "không đếm (lớp được hỏi vắng mặt)"
        note += (f"\n- Kiểm tra lớp (CLIP): điểm {vinfo['score']:.3f} {'≥' if vinfo['present'] else '<'} ngưỡng "
                 f"{CLIP_THR:.3f} → {verdict}")
    summary = (f"## Đếm được: **{count:.1f}** \"{text.strip()}\"\n"
               f"- {len(boxes)} điểm vật (đỉnh density, min_mass={min_mass:g})"
               f"{' · đã chia ảnh 2×2 (đếm ≥ ' + str(int(ttn_thr)) + ')' if tiled else ''}\n"
               f"- {note}\n- Thời gian: {ms:.0f} ms trên {DEVICE.type.upper()}")
    return summary, points, gr.update(value=heat, visible=True), gr.update(value=maps, visible=True)


def add_exemplar(exemplars, evt: gr.SelectData):
    exemplars = list(exemplars or [])
    if len(exemplars) < 3:
        exemplars.append(tuple(evt.index))
    return exemplars, f"Exemplar: {len(exemplars)}/3 — {exemplars}"


# ----------------------------------------------------------------- Video
def read_video(path, target_fps, max_frames):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps / target_fps)))
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    return frames


@torch.no_grad()
def count_in_video(engine, video, frame_dir, text, target_fps, max_frames, tracker, min_mass, ckpt,
                   progress=gr.Progress()):
    if not text.strip():
        raise gr.Error("Cần tên vật cần đếm.")
    if frame_dir and os.path.isdir(frame_dir):
        frames = frame_list(frame_dir)[:int(max_frames)]
    elif video:
        frames = read_video(video, target_fps, int(max_frames))
    else:
        raise gr.Error("Tải video lên hoặc nhập đường dẫn thư mục frame.")
    if not frames:
        raise gr.Error("Không đọc được frame nào.")
    drawn = []
    sam3 = engine == "SAM 3"

    def on_frame(i, img, boxes, ids, c):
        progress((i + 1) / len(frames), desc=f"frame {i + 1}/{len(frames)}")
        im = draw_boxes(img, boxes, ids) if sam3 else draw_points(img, boxes, ids)
        h, w = im.shape[:2]
        s = 640 / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        cv2.putText(im, f"frame {i + 1}: {c:.0f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(im, f"frame {i + 1}: {c:.0f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        drawn.append(Image.fromarray(im))

    if sam3:
        r = get_sam3().count_video(frames, text.strip(), fps=target_fps, on_frame=on_frame)
        tracker = "SAM 3 tracking"
    else:
        r = count_video(get_model(ckpt), frames, text.strip(), DEVICE, min_mass=min_mass, tracker=tracker,
                        on_frame=on_frame)
    gif = os.path.join(tempfile.mkdtemp(), "tracks.gif")
    drawn[0].save(gif, save_all=True, append_images=drawn[1:], duration=int(1000 / max(target_fps, 1)), loop=0)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(range(1, len(r["frame_counts"]) + 1), r["frame_counts"], color="#2b6cb0")
    ax.set_xlabel("frame")
    ax.set_ylabel("số đếm trong frame")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    summary = (f"## Số vật duy nhất: **{r['count']}** \"{text.strip()}\"\n"
               f"- {r['frames']} frame · {r['frames'] / r['sec']:.1f} FPS trên {DEVICE.type.upper()} · tracker "
               f"`{tracker}`\n- Đếm trung bình mỗi frame: {np.mean(r['frame_counts']):.1f}\n"
               f"- Lưu ý: tracker nhẹ còn hay tách một vật thành nhiều track (đếm thừa) — xem tab Mô hình.")
    return summary, gif, fig


ABOUT = """
### Kiến trúc (arch `vit`)
```
Ảnh 640 ─► YOLO-World v8s (đóng băng) ─► P3/P4/P5 + Text Map (khớp tên lớp)
       ├─► Sobel ─► Edge Map
       └─► DINOv2-small (đóng băng, 560px) ─► token 384ch 40×40
Bước 1  Density Head [P3 | Text | Edge | ViT 64ch] ─► density thô
Bước 2  5 đỉnh density thô = exemplar ─► similarity P3 / P4 / DINOv2
        Refine Head ─► density cuối  →  SỐ ĐẾM = tổng density
Bước 3  Scale Head ─► kích thước vật
```
Chỉ ~0.6M tham số được train. Ảnh rất đông (đếm ≥ 400) được chia 2×2 và đếm lại.

### Kết quả (Kaggle T4)
| | FSC-147 test MAE | Thời gian/ảnh |
|---|---|---|
| CountYOLO (exnoise) + chia ô | **16.53** | ~44 ms |
| CountYOLO (exnoise), fp16 + compile + CLIP kiểm tra lớp | 18.38 | ~40 ms |
| CountGD-Box (CountVid) | 15.01 | ~422 ms |
| YOLO-Count (ICCV 2025) | 14.80 | — |
| RT-Counter (preprint 2026) | 13.30 | — |

Chống nhầm lớp (CLIP kiểm tra, FSC-147 test): hỏi lớp không có trong ảnh → đếm TB 0.77 (không kiểm tra: ~60; SAM 3: 0.4).

Video (MAE số vật duy nhất): Crystals 49.9 (CountVid 69.1), Penguins 2.67 (CountVid 3.3) khi dùng SAM 2 + SAM 3 frame chính;
tracker nhẹ trong giao diện này nhanh (~10 FPS) nhưng còn đếm thừa.
"""


def build(ckpts, default):
    with gr.Blocks(title="CountYOLO") as demo:
        gr.Markdown("# CountYOLO — đếm vật theo text (YOLO-World + DINOv2)")
        with gr.Row():
            engine = gr.Radio(ENGINES, value="Grounding DINO",
                              label="Engine (SAM 3: tách mask · Grounding DINO: NMS-Free queries · CountYOLO: density map)")
            ckpt = gr.Dropdown(ckpts, value=default, label="Checkpoint CountYOLO")
        with gr.Tab("Đếm ảnh"):
            exemplars = gr.State([])
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(label="Ảnh (click vào 1–3 vật để làm exemplar, tuỳ chọn)", type="numpy")
                    text = gr.Textbox(label="Đếm cái gì?", placeholder="vd: apples")
                    ex_info = gr.Markdown("Exemplar: 0/3 (để trống = tự chọn)")
                    verify = gr.Checkbox(True, label="Kiểm tra lớp bằng CLIP (chống đếm nhầm lớp, CountYOLO)")
                    with gr.Row():
                        clear = gr.Button("Xoá exemplar")
                        run = gr.Button("Đếm", variant="primary")
                    with gr.Accordion("Tuỳ chọn CountYOLO", open=False):
                        tiling = gr.Checkbox(True, label="Chia ảnh 2×2 khi rất đông")
                        thr = gr.Slider(50, 1000, 400, step=50, label="Ngưỡng chia ô (số đếm)")
                        mm = gr.Slider(0, 0.7, 0.2, step=0.05, label="min_mass khi vẽ điểm vật")
                    with gr.Accordion("Tuỳ chọn Grounding DINO", open=False):
                        gd_box_thr = gr.Slider(0.01, 0.40, 0.10, step=0.01, label="Ngưỡng tự tin box (box_threshold)")
                        gd_text_thr = gr.Slider(0.01, 0.40, 0.10, step=0.01, label="Ngưỡng tự tin text (text_threshold)")
                with gr.Column(scale=1):
                    summary = gr.Markdown()
                    pts = gr.Image(label="Điểm vật")
                    heat = gr.Image(label="Density")
            maps = gr.Gallery(label="Bản đồ trung gian", columns=4, height=300)
            img.select(add_exemplar, exemplars, [exemplars, ex_info])
            img.upload(lambda: ([], "Exemplar: 0/3 (để trống = tự chọn)"), None, [exemplars, ex_info])
            clear.click(lambda: ([], "Exemplar: 0/3 (để trống = tự chọn)"), None, [exemplars, ex_info])
            inputs = [engine, img, text, exemplars, verify, tiling, thr, mm, ckpt, gd_box_thr, gd_text_thr]
            run.click(count_image, inputs, [summary, pts, heat, maps])
            text.submit(count_image, inputs, [summary, pts, heat, maps])
        with gr.Tab("Đếm video"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid = gr.Video(label="Video")
                    fdir = gr.Textbox(label="…hoặc thư mục frame (vd data/VideoCount/Penguins/frames/penguins_1)")
                    vtext = gr.Textbox(label="Đếm cái gì?", placeholder="vd: penguin")
                    with gr.Row():
                        vfps = gr.Slider(1, 10, 3, step=1, label="FPS lấy mẫu (video)")
                        vmax = gr.Slider(10, 400, 120, step=10, label="Số frame tối đa")
                    trk = gr.Radio(list(TRACKERS), value="appearance", label="Tracker")
                    vmm = gr.Slider(0, 0.7, 0.2, step=0.05, label="min_mass")
                    vrun = gr.Button("Đếm video", variant="primary")
                with gr.Column(scale=1):
                    vsum = gr.Markdown()
                    vgif = gr.Image(label="Track (màu = id)")
                    vplot = gr.Plot(label="Số đếm theo frame")
            vrun.click(count_in_video, [engine, vid, fdir, vtext, vfps, vmax, trk, vmm, ckpt], [vsum, vgif, vplot])
        with gr.Tab("Mô hình"):
            gr.Markdown(ABOUT)
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--port", type=int, default=None,
                   help="Mặc định: cổng trống đầu tiên từ 7860 (Gradio tự tìm nếu 7860 đang bận)")
    p.add_argument("--share", action="store_true")
    a = p.parse_args()
    ckpts = sorted(glob.glob(os.path.join(ROOT, "checkpoints", "*.pth")))
    ckpts = [c for c in ckpts if "vit" in os.path.basename(c)] + [c for c in ckpts if "vit" not in os.path.basename(c)]
    default = a.checkpoint if os.path.exists(a.checkpoint) else (ckpts[0] if ckpts else a.checkpoint)
    print(f"Thiết bị: {DEVICE} | checkpoint mặc định: {default}")
    # Nạp model và chạy thử một lượt để lần đếm đầu không phải chờ khởi tạo CUDA (~15 s)
    with torch.no_grad():
        get_model(default)(torch.zeros(1, 3, SIZE, SIZE, device=DEVICE), ["object"])
        get_verifier(get_model(default))(torch.zeros(1, 3, SIZE, SIZE, device=DEVICE), "object")  # nạp CLIP + từ vựng
    build(ckpts or [default], default).launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
