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
'''

BOX_ANCHOR = '''        box_prompts = box_prompts.cpu()
        countgd_pred_boxes[frame_id] = box_prompts
'''
BOX_PATCH = '''        box_prompts = box_prompts.cpu()
        frame_id = idx * (args.img_batch_size) + intra_batch_ind
        if args.tile_target > 0 or args.tile_grid > 0 or args.nms_iou < 1.0:
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
IMPORT_PATCH = IMPORT_ANCHOR + "from tiled_countgd import tiled_boxes\nTILE_LOG = []\n"

STAGE1_ANCHOR = 'print("time stage 1: " + str(time_stage_1))\n'
STAGE1_PATCH = STAGE1_ANCHOR + '''if TILE_LOG:
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
                          (STAGE1_ANCHOR, STAGE1_PATCH)]:
        if src.count(anchor) != 1:
            sys.exit(f"Không tìm thấy đúng 1 chỗ vá trong {path}:\n{anchor}")
        src = src.replace(anchor, patch)
    open(path, "w", encoding="utf-8").write(src)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiled_countgd.py"), countvid_dir)
    print("Đã vá:", path)


if __name__ == "__main__":
    main(sys.argv[1])
