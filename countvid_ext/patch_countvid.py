"""
Vá count_in_videos.py của CountVid (https://github.com/niki-amini-naieni/CountVid) để thêm Canonical-Scale Tiling
+ NMS vào Stage 1. Không truyền cờ mới -> hành vi giữ nguyên như bản gốc.

    python countvid_ext/patch_countvid.py <thư mục CountVid>

Cờ mới của count_in_videos.py:
    --tile_target PX     bật tiling tự chọn lưới để vật đạt ~PX px sau resize của CountGD (0 = tắt)
    --tile_grid G        lưới cố định G x G (ghi đè --tile_target)
    --tile_max_grid G    lưới tối đa khi tự chọn (mặc định 3)
    --tile_overlap R     tỉ lệ chồng lấn giữa các ô (mặc định 0.25)
    --nms_iou T          NMS trên box CountGD (mặc định 1.0 = tắt; tiling bật thì nên 0.5)
    --stage1_countyolo DIR   thay CountGD-Box bằng CountYOLO (code ở DIR) làm Stage 1 (countyolo_stage1.py)
    --countyolo_ckpt P       checkpoint head CountYOLO
    --countyolo_ttn_thr T    test-time tiling theo số đếm cho CountYOLO (0 = tắt)
    --countyolo_ttn_grid G   lưới khi chia (mặc định 2)
    --countyolo_sam3 W       sam3.pt: frame chính SAM 3 (có mặt + exemplar đúng lớp) cho Stage 1 CountYOLO
    --countyolo_kf_every N   chạy SAM 3 mỗi N frame (0 = tắt)
"""
import os
import shutil
import sys

ARGS_ANCHOR = '    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")\n'
ARGS_PATCH = ARGS_ANCHOR + '''    # countvid_ext: Canonical-Scale Tiling + NMS cho Stage 1
    parser.add_argument("--tile_target", type=float, default=0)
    parser.add_argument("--tile_grid", type=int, default=0)
    parser.add_argument("--tile_max_grid", type=int, default=3)
    parser.add_argument("--tile_overlap", type=float, default=0.25)
    parser.add_argument("--nms_iou", type=float, default=1.0)
    parser.add_argument("--stage1_countyolo", type=str, default="")
    parser.add_argument("--countyolo_ckpt", type=str, default="")
    parser.add_argument("--countyolo_ttn_thr", type=float, default=0.0)
    parser.add_argument("--countyolo_ttn_grid", type=int, default=2)
    parser.add_argument("--countyolo_sam3", type=str, default="")
    parser.add_argument("--countyolo_kf_every", type=int, default=0)
'''

# Stage 1 bằng CountYOLO: không gọi CountGD (logits rỗng -> không box nào qua CONF_THRESH), box do CountYOLO cấp
COUNT_ANCHOR = ('    countgd_output = count(input_frame, input_image_exemplars, exemplars, label, input_text, '
                'device=args.device)\n')
COUNT_PATCH = '''    if STAGE1 is not None:
        countgd_output = {"pred_logits": torch.full((input_frame.shape[0], 1, 1), -10.0),
                          "pred_boxes": torch.zeros(input_frame.shape[0], 1, 4)}
    else:
        countgd_output = count(input_frame, input_image_exemplars, exemplars, label, input_text, device=args.device)
'''

MODEL_ANCHOR = "model, transform = build_model_and_transforms(args)\n"
MODEL_PATCH = MODEL_ANCHOR + '''STAGE1 = CountYOLOStage1(args.stage1_countyolo, args.countyolo_ckpt, device=args.device,
                         ttn_thr=args.countyolo_ttn_thr, ttn_grid=args.countyolo_ttn_grid,
                         sam3_weights=args.countyolo_sam3, kf_every=args.countyolo_kf_every) \\
    if args.stage1_countyolo else None
'''

BOX_ANCHOR = '''        box_prompts = box_prompts.cpu()
        countgd_pred_boxes[frame_id] = box_prompts
'''
BOX_PATCH = '''        box_prompts = box_prompts.cpu()
        frame_id = idx * (args.img_batch_size) + intra_batch_ind
        if STAGE1 is not None:
            box_prompts, _, _ = STAGE1.boxes(Image.open(video_dir + "/" + frame_names[frame_id]).convert("RGB"),
                                             input_text)
            TILE_LOG.append((frame_id, 0, int(len(box_prompts))))
        elif args.tile_target > 0 or args.tile_grid > 0 or args.nms_iou < 1.0:
            box_prompts, _, grid = tiled_boxes(
                lambda *a: count(*a, device=args.device), transform,
                Image.open(video_dir + "/" + frame_names[frame_id]).convert("RGB"), box_prompts,
                logits.max(dim=-1).values[intra_batch_ind, box_mask_frame].float().cpu(), input_text,
                input_image_exemplars[intra_batch_ind], exemplars[intra_batch_ind], label[intra_batch_ind],
                CONF_THRESH, target=args.tile_target if args.tile_target > 0 else 32,
                max_grid=args.tile_max_grid, overlap=args.tile_overlap,
                nms_iou=args.nms_iou if args.nms_iou < 1.0 else 0.5, batch_size=args.img_batch_size,
                fixed_grid=args.tile_grid if args.tile_grid > 0 else (1 if args.tile_target <= 0 else 0))
            TILE_LOG.append((frame_id, int(grid), int(len(box_prompts))))
        countgd_pred_boxes[frame_id] = box_prompts
'''

IMPORT_ANCHOR = "import scipy.ndimage as ndimage\n"
IMPORT_PATCH = IMPORT_ANCHOR + ("from tiled_countgd import tiled_boxes\n"
                                "from countyolo_stage1 import CountYOLOStage1\nTILE_LOG = []\n")

STAGE1_ANCHOR = 'print("time stage 1: " + str(time_stage_1))\n'
STAGE1_PATCH = STAGE1_ANCHOR + '''if STAGE1 is not None and STAGE1.kf_log:
    print("sam3 keyframe: %d frame chính, không thấy vật: %d, số vật SAM 3 TB %.1f" % (
        len(STAGE1.kf_log), sum(n == 0 for _, n in STAGE1.kf_log),
        sum(n for _, n in STAGE1.kf_log) / len(STAGE1.kf_log)))
if TILE_LOG:
    print("tiling: grid trung bình %.2f, box/frame trung bình %.1f" % (
        sum(g for _, g, _ in TILE_LOG) / len(TILE_LOG), sum(n for _, _, n in TILE_LOG) / len(TILE_LOG)))
'''


def main(countvid_dir):
    path = os.path.join(countvid_dir, "count_in_videos.py")
    src = open(path, encoding="utf-8").read()
    if "countvid_ext" in src:
        print("Đã vá từ trước:", path)
        return
    for anchor, patch in [(IMPORT_ANCHOR, IMPORT_PATCH), (ARGS_ANCHOR, ARGS_PATCH), (BOX_ANCHOR, BOX_PATCH),
                          (STAGE1_ANCHOR, STAGE1_PATCH), (COUNT_ANCHOR, COUNT_PATCH), (MODEL_ANCHOR, MODEL_PATCH)]:
        if src.count(anchor) != 1:
            sys.exit(f"Không tìm thấy đúng 1 chỗ vá trong {path}:\n{anchor}")
        src = src.replace(anchor, patch)
    open(path, "w", encoding="utf-8").write(src)
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ("tiled_countgd.py", "countyolo_stage1.py"):
        shutil.copy(os.path.join(here, f), countvid_dir)
    print("Đã vá:", path)


if __name__ == "__main__":
    main(sys.argv[1])
