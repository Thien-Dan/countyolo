"""
Chống nhầm lớp bằng CLIP cho CountYOLO trong MỘT lượt (utils/clip_verify.VerifiedCounter), không cần SAM 3:
  1. bước 1 của CountYOLO (density thô) -> vùng ứng viên: K đỉnh density thô + K box YOLO-World cho đúng prompt
  2. CLIP: p(prompt) từng vùng (+ cả ảnh) so với từ vựng chung (lớp FSC-147 train + COCO)
  3. có mặt nếu điểm >= ngưỡng (chọn trên VAL), ngược lại đếm 0
  4. exemplar cho Refine Head: tự chọn (self) / tâm các vùng giống prompt nhất (clip) / tự động (auto: dùng exemplar CLIP
     khi đỉnh mô hình tự chọn kém giống prompt hơn hẳn -> cảnh nhiều lớp)
Báo cáo TEST theo giao thức tools/prompt_eval.py: hỏi đúng lớp, hỏi sai lớp, mosaic 2 lớp; thời gian của đúng lượt
chạy thật (chế độ auto).

    python tools/clip_verify_eval.py --checkpoint <ckpt> --data-dir data/FSC147 --fast --out clip_verify.json
"""
# Chạy được từ thư mục gốc dự án: python tools/<script>.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from datasets.fsc147_dataset import FSC147Dataset  # noqa: E402
from models.density_yolo import DensityYOLOWorld  # noqa: E402
from utils.clip_verify import (ClipVerifier, VerifiedCounter, default_vocab, lvis_names, negative_mask,  # noqa: E402
                               o365_names, select_exemplars)

SCORES = ("score", "global_p", "crop_max", "crop_top3")
# Biến thể điểm có mặt, chấm lại từ cùng đặc trưng CLIP (không chạy lại mô hình): (bộ từ vựng đối chứng, ngưỡng loại
# tên đồng nghĩa / lớp cha theo embedding text). base = FSC-147 train + COCO; lvis = 1203 lớp LVIS; all = base + LVIS +
# Objects365. Chọn biến thể theo VAL.
VARIANTS = [(v, s) for v in ("base", "lvis", "all") for s in (None, 0.9, 0.95)]


def vkey(vocab, syn):
    return f"v_{vocab}_{syn or 'none'}"


KEYS = SCORES + tuple(vkey(*x) for x in VARIANTS)
MODES = ("self", "clip", "auto")
# Exemplar âm (utils/clip_verify.negative_mask) áp lên lượt exemplar CLIP: (ngưỡng p vùng âm, margin)
# Vòng C3: exemplar âm không giúp (mosaic 27–34%, MAE tệ đi) -> tắt; để lại để tái lập: [(0.05, 0.0), (0.05, 0.05), ...]
NEG = []
# Số exemplar CLIP thích ứng (select_exemplars): giữ vùng p >= alpha x p cao nhất; alpha chọn trên mosaic VAL
# Vòng C4: alpha không giúp (mosaic VAL tốt nhất ở alpha 0 = đủ 5 exemplar) -> tắt; để lại để tái lập: (0.3, 0.5, 0.7)
ALPHAS = ()
# Vòng C5: nguồn exemplar CLIP — chỉ box YOLO-World cho prompt / chỉ đỉnh density (mặc định: cả hai)
EX_SRC = ("yolo", "peaks")


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(s)) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Runner:
    def __init__(self, vc, device):
        self.vc, self.device, self.ms = vc, device, []

    def _timed(self, x, cls, mode):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.vc(x, cls, exemplars=mode, return_aux=mode == "clip")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        info = out[1]
        if mode == "clip":  # đếm lại với exemplar âm từ cùng density / đặc trưng (không chạy lại mô hình)
            for neg_p, margin in NEG:
                d = negative_mask(out[0], out[2].get("vit_feats"), info["boxes"], info["probs"], info["ex_idx"],
                                  neg_p, margin, x.shape[-1])
                info[f"neg_{neg_p}_{margin}"] = float(d.float().sum())
        return info, (time.perf_counter() - t0) * 1000

    @torch.no_grad()
    def __call__(self, image, cls, oracle=None, alphas=False):
        x = image[None].to(self.device)
        info, ms_self = self._timed(x, cls, "self")
        clip, ms_clip = self._timed(x, cls, "clip")
        # thời gian lượt chạy thật ở chế độ auto = lượt self hoặc clip tuỳ quyết định (cùng điểm, cùng quyết định)
        self.ms.append(ms_clip if info["auto_use_clip"] else ms_self)
        r = {k: info[k] for k in SCORES}
        feats = info.pop("feats")
        for voc, syn in VARIANTS:
            pv = self.vc.v.probs(feats, cls, voc, syn)
            pc, pg = pv[:-1], pv[-1]
            top3 = pc.topk(min(3, len(pc))).values.mean() if len(pc) else pg
            r[vkey(voc, syn)] = float(0.5 * top3 + 0.5 * pg)
        r.update(count_self=info["count"], count_clip=clip["count"], auto=info["auto_use_clip"])
        r["count_auto"] = r["count_clip"] if r["auto"] else r["count_self"]
        for neg_p, margin in NEG:  # auto + exemplar âm (chỉ khi auto chọn exemplar CLIP)
            r[f"count_autoneg_{neg_p}_{margin}"] = clip[f"neg_{neg_p}_{margin}"] if r["auto"] else r["count_self"]
        if alphas:  # đếm lại với số exemplar thích ứng (cùng điểm CLIP, chỉ đổi tập exemplar)
            order = torch.argsort(info["probs"], descending=True)
            for al in ALPHAS:
                ex = select_exemplars(info["probs"], order, self.vc.n_ex, al)
                b = info["boxes"][ex]
                c = float(self.vc.m(x, [cls], exemplar_points=[(b[:, :2] + b[:, 2:]) / 2])[1].float().sum())
                r[f"count_clipa_{al}"] = c
                r[f"count_autoa_{al}"] = c if r["auto"] else r["count_self"]
            npk, pr = info["n_peaks"], info["probs"]
            for src in EX_SRC:
                cand = torch.arange(npk, len(pr)) if src == "yolo" else torch.arange(0, npk)
                if len(cand) == 0:
                    cand = torch.arange(len(pr))
                ex = cand[torch.argsort(pr[cand], descending=True)[:self.vc.n_ex]]
                b = info["boxes"][ex]
                c = float(self.vc.m(x, [cls], exemplar_points=[(b[:, :2] + b[:, 2:]) / 2])[1].float().sum())
                r[f"count_clips_{src}"] = c
                r[f"count_autos_{src}"] = c if r["auto"] else r["count_self"]
        if oracle is not None:
            r["count_oracle"] = float(self.vc.m(x, [cls], exemplar_points=[oracle])[1].float().sum())
        return r


def per_image(run, ds, n, rng, classes, alphas=False):
    pos, neg = [], []
    for i in range(n):
        s = ds[i]
        wrong = rng.choice([c for c in classes if c != s['class_name']])
        pos.append({**run(s['image'], s['class_name'], alphas=alphas), "gt": s['count']})
        neg.append({**run(s['image'], wrong), "gt": s['count']})
    return pos, neg


def col(rows, k):
    return np.array([float(r[k]) for r in rows])


def gated(rows, key, t, count):
    return col(rows, count) * (col(rows, key) >= t)


def mosaics(run, ds, rng, classes, n, alphas=True):
    """Cùng cách dựng với tools/prompt_eval.py (2 ô mỗi lớp, vị trí ngẫu nhiên)."""
    rows = []
    for _ in range(n):
        ca, cb = rng.sample([c for c in classes if len(ds.class_to_idx[c]) >= 2], 2)
        tiles = [(ca, j) for j in rng.sample(ds.class_to_idx[ca], 2)] + \
                [(cb, j) for j in rng.sample(ds.class_to_idx[cb], 2)]
        rng.shuffle(tiles)
        half = ds.image_size // 2
        image = torch.zeros(3, ds.image_size, ds.image_size)
        gt, pts = {ca: 0.0, cb: 0.0}, {ca: [], cb: []}
        for k, (c, j) in enumerate(tiles):
            s = ds._load(j)
            oy, ox = divmod(k, 2)
            image[:, oy * half:(oy + 1) * half, ox * half:(ox + 1) * half] = \
                F.interpolate(s['image'][None], size=(half, half), mode='area')[0]
            gt[c] += s['count']
            pts[c].append(s['points'] / 2 + torch.tensor([ox * half, oy * half], dtype=torch.float32))
        for c in (ca, cb):
            rows.append({**run(image, c, oracle=torch.cat(pts[c]), alphas=alphas), "gt": gt[c]})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data/FSC147")
    p.add_argument("--mosaics", type=int, default=300)
    p.add_argument("--val-mosaics", type=int, default=150, help="Mosaic VAL để chọn alpha (số exemplar thích ứng)")
    p.add_argument("--limit", type=int, default=None, help="Chỉ dùng N ảnh mỗi split (chạy thử)")
    p.add_argument("--k", type=int, default=12, help="Số đỉnh density thô và số box YOLO-World đem kiểm tra")
    p.add_argument("--wrong-target", type=float, default=0.5,
                   help="Ngưỡng có mặt = nhỏ nhất sao cho đếm TB khi hỏi sai lớp trên VAL <= giá trị này")
    p.add_argument("--fast", action="store_true", help="CountYOLO fp16 + torch.compile (đo thời gian như T3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dump", default=None, help="CSV từng câu hỏi")
    p.add_argument("--out", default="clip_verify.json")
    p.add_argument("--vocabs", nargs="+", default=["base", "lvis", "all"], help="Bộ từ vựng chấm điểm có mặt (chạy thử: base)")
    a = p.parse_args()
    global VARIANTS, KEYS
    VARIANTS = [(v, s) for v, s in VARIANTS if v in a.vocabs]
    KEYS = SCORES + tuple(vkey(*x) for x in VARIANTS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DensityYOLOWorld("yolov8s-world.pt")
    ckpt = model.load_head(a.checkpoint, map_location="cpu")
    model.to(device).eval_mode()
    if a.fast:
        model.optimize_inference(half_vit=device.type == "cuda", compile_vit=device.type == "cuda")
    # --fast: CLIP compile batch cố định 2K + 1 (K đỉnh + K box YOLO-World + cả ảnh)
    verifier = ClipVerifier(model, default_vocab(), compile_batch=2 * a.k + 1 if a.fast and device.type == "cuda" else 0)
    if "lvis" in a.vocabs:
        verifier.add_vocab("lvis", lvis_names())
    if "all" in a.vocabs:
        verifier.add_vocab("all", default_vocab() + lvis_names() + o365_names())
    run = Runner(VerifiedCounter(model, verifier, thr=0.0, k=a.k, keep_feats=True), device)
    print(f"Từ vựng đối chứng: {len(verifier.vocab)} tên (FSC-147 train + COCO), K={a.k}, fast={a.fast} | biến thể: "
          + ", ".join(f"{k} {len(v[0])}" for k, v in verifier.vocabs.items()), flush=True)
    rng = random.Random(a.seed)

    # 1. VAL: chọn ngưỡng có mặt cho từng loại điểm (đếm theo chế độ auto)
    val = FSC147Dataset(a.data_dir, split="val")
    vp, vn = per_image(run, val, min(a.limit or len(val), len(val)), rng, sorted(val.class_to_idx))
    thr = {}
    for key in KEYS:
        sp, sn = col(vp, key), col(vn, key)
        cands = np.unique(np.concatenate([[0.0], sp, sn]))
        ok = [t for t in cands if gated(vn, key, t, "count_auto").mean() <= a.wrong_target]
        thr[key] = float(min(ok)) if ok else float(cands.max())
        print(f"VAL {key:15s}: AUC {auc(sp, sn):.3f} | ngưỡng {thr[key]:.4f} | MAE đúng lớp "
              f"{np.abs(gated(vp, key, thr[key], 'count_auto') - col(vp, 'gt')).mean():.2f} | đếm TB sai lớp "
              f"{gated(vn, key, thr[key], 'count_auto').mean():.2f}", flush=True)
    # biến thể tốt nhất trên VAL: MAE hỏi đúng lớp nhỏ nhất (ở ngưỡng đã giữ đếm sai lớp <= wrong_target)
    best = min(KEYS, key=lambda k: np.abs(gated(vp, k, thr[k], "count_auto") - col(vp, "gt")).mean())
    print(f"Biến thể tốt nhất theo VAL: {best}", flush=True)

    # 2. TEST: hỏi đúng / sai lớp
    test = FSC147Dataset(a.data_dir, split="test")
    classes = sorted(test.class_to_idx)
    tp, tn = per_image(run, test, min(a.limit or len(test), len(test)), rng, classes, alphas=True)
    gt = col(tp, "gt")
    res = {"checkpoint": os.path.basename(a.checkpoint), "arch": ckpt.get("arch"), "k": a.k, "fast": a.fast,
           "wrong_target_val": a.wrong_target, "thresholds": thr, "best_on_val": best,
           "no_gate": {m: {"mae": float(np.abs(col(tp, f"count_{m}") - gt).mean()),
                           "wrong_mean": float(col(tn, f"count_{m}").mean())} for m in MODES},
           "auto_use_clip_pct": {"test_correct": float(col(tp, "auto").mean() * 100)}, "test": {}}
    print("\n=== TEST không cổng: " + " | ".join(f"{m}: MAE {v['mae']:.2f}, sai lớp {v['wrong_mean']:.1f}"
                                            for m, v in res["no_gate"].items())
          + f" | auto dùng exemplar CLIP ở {res['auto_use_clip_pct']['test_correct']:.1f}% ảnh đúng lớp", flush=True)
    for key in KEYS:
        t = thr[key]
        sp, sn = col(tp, key), col(tn, key)
        r = res["test"][key] = {"auc": auc(sp, sn), "correct_suppressed_pct": float((sp < t).mean() * 100),
                                "wrong_passed_pct": float((sn >= t).mean() * 100)}
        for m in (*MODES, *[f"autoneg_{a}_{b}" for a, b in NEG]):
            r[m] = {"mae": float(np.abs(gated(tp, key, t, f"count_{m}") - gt).mean()),
                    "wrong_mean": float(gated(tn, key, t, f"count_{m}").mean())}
        print(f"  {key:15s}: AUC {r['auc']:.3f} | chặn nhầm {r['correct_suppressed_pct']:.1f}% | lọt "
              f"{r['wrong_passed_pct']:.1f}% | " + " | ".join(
                  f"{m}: MAE {r[m]['mae']:.2f} sai lớp {r[m]['wrong_mean']:.2f}"
                  for m in (r if key in ("score", "v_all_none") else MODES) if isinstance(r[m], dict)), flush=True)

    # 3. mosaic 2 lớp (cả hai lớp đều có mặt)
    mos = mosaics(run, test, rng, classes, a.mosaics)
    # alpha chọn trên mosaic VAL (rng riêng: không đổi chuỗi mosaic TEST so với các lần trước)
    vmos = mosaics(run, val, random.Random(a.seed + 1), sorted(val.class_to_idx), a.val_mosaics)
    vmg = col(vmos, "gt")
    vrel = {al: float(np.median(np.abs(col(vmos, f"count_autoa_{al}") - vmg) / np.maximum(vmg, 1))) for al in ALPHAS}
    vrel[0.0] = float(np.median(np.abs(col(vmos, "count_auto") - vmg) / np.maximum(vmg, 1)))
    vsrc = {"both": vrel[0.0], **{src: float(np.median(np.abs(col(vmos, f"count_autos_{src}") - vmg) / np.maximum(vmg, 1)))
                                  for src in EX_SRC}}
    best_src = min(vsrc, key=vsrc.get)
    print("=== Nguồn exemplar CLIP: mosaic VAL (auto) " + " | ".join(f"{k}: {v:.3f}" for k, v in vsrc.items())
          + f" -> chọn {best_src}", flush=True)
    for src in EX_SRC:
        print(f"  TEST nguồn {src}: ảnh 1 lớp MAE (auto, không cổng) "
              f"{np.abs(col(tp, f'count_autos_{src}') - gt).mean():.2f}", flush=True)
    best_alpha = min(vrel, key=vrel.get)
    print("\n=== Số exemplar thích ứng: mosaic VAL (auto) " + " | ".join(f"alpha {k}: {v:.3f}" for k, v in sorted(vrel.items()))
          + f" -> chọn alpha {best_alpha}", flush=True)
    pk = "v_all_none" if "v_all_none" in KEYS else "score"
    for al in ALPHAS:
        print(f"  TEST alpha {al}: ảnh 1 lớp MAE (auto, không cổng) {np.abs(col(tp, f'count_autoa_{al}') - gt).mean():.2f} | "
              f"có cổng score {np.abs(gated(tp, 'score', thr['score'], f'count_autoa_{al}') - gt).mean():.2f} / "
              f"{pk} {np.abs(gated(tp, pk, thr[pk], f'count_autoa_{al}') - gt).mean():.2f}",
              flush=True)
    mg = col(mos, "gt")
    rel = lambda c: float(np.median(np.abs(c - mg) / np.maximum(mg, 1)))  # noqa: E731
    res["mosaic"] = {m: {"mae": float(np.abs(col(mos, f"count_{m}") - mg).mean()), "median_rel_err": rel(col(mos, f"count_{m}"))}
                     for m in (*MODES, "oracle", *[f"autoneg_{a}_{b}" for a, b in NEG],
                               *[f"{k}a_{al}" for k in ("clip", "auto") for al in ALPHAS],
                               *[f"{k}s_{src}" for k in ("clip", "auto") for src in EX_SRC])}
    for name, key in (("auto_gated", "score"), ("auto_gated_best", best)):
        c = gated(mos, key, thr[key], "count_auto")
        res["mosaic"][name] = {"mae": float(np.abs(c - mg).mean()), "median_rel_err": rel(c),
                               "suppressed_pct": float((col(mos, key) < thr[key]).mean() * 100)}
    res["auto_use_clip_pct"]["mosaic"] = float(col(mos, "auto").mean() * 100)
    res["alpha_val_mosaic_rel"], res["alpha_best_on_val"] = vrel, best_alpha
    res["exsrc_val_mosaic_rel"], res["exsrc_best_on_val"] = vsrc, best_src
    print("\n=== Mosaic 2 lớp ===", flush=True)
    for name, r in res["mosaic"].items():
        print(f"  {name:12s}: MAE {r['mae']:7.2f} | sai số tương đối trung vị {r['median_rel_err']:.2f}", flush=True)
    print(f"  auto dùng exemplar CLIP ở {res['auto_use_clip_pct']['mosaic']:.1f}% câu hỏi mosaic", flush=True)

    ms = run.ms[20:] or run.ms  # bỏ các lượt khởi động / torch.compile
    res["ms_auto"] = {"median": float(np.median(ms)), "p90": float(np.percentile(ms, 90))}
    print(f"\nThời gian một lượt (CountYOLO + CLIP, chế độ auto): trung vị {res['ms_auto']['median']:.1f} ms, "
          f"p90 {res['ms_auto']['p90']:.1f} ms", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    if a.dump:
        keys = ["gt", *KEYS, "count_self", "count_clip", "count_auto", "auto"]
        with open(a.dump, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "kind", *keys, "count_oracle"])
            for split, kind, rows in [("val", "pos", vp), ("val", "neg", vn), ("test", "pos", tp), ("test", "neg", tn),
                                      ("test", "mosaic", mos)]:
                for r in rows:
                    w.writerow([split, kind, *[round(float(r[k]), 5) for k in keys], round(r.get("count_oracle", -1), 3)])


if __name__ == "__main__":
    main()
