"""
Chọn cấu hình test-time tiling (ngưỡng số đếm, lưới) trên VAL, rồi báo cáo TEST với cấu hình đó (và không tiling
để so sánh). Không nhìn test khi chọn.

    python tools/ttn_sweep.py --checkpoint <ckpt> --data-dir data/FSC147 --out ttn_sweep.json
"""
# Chạy được từ thư mục gốc dự án: python tools/<script>.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json

from eval_fsc147 import eval_fsc147

THRESHOLDS = (100, 200, 400)
GRIDS = (2, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data/FSC147")
    p.add_argument("--out", default="ttn_sweep.json")
    p.add_argument("--dump-dir", default=".")
    a = p.parse_args()

    run = lambda split, thr=0.0, grid=2, dump=None: eval_fsc147(
        split=split, mode="density-count", checkpoint=a.checkpoint, data_dir=a.data_dir,
        ttn_thr=thr, ttn_grid=grid, dump=dump)

    val = {"off": run("val")}
    for thr in THRESHOLDS:
        for grid in GRIDS:
            val[f"thr{thr}_g{grid}"] = run("val", thr, grid)
    best = min(val, key=lambda k: val[k]["mae"])
    print("\n=== VAL ===")
    for k, v in val.items():
        print(f"  {k:12s} MAE {v['mae']:.2f} RMSE {v['rmse']:.2f}" + ("  <- chọn" if k == best else ""))

    test = {"off": run("test", dump=f"{a.dump_dir}/test_off.csv")}
    if best != "off":
        thr, grid = int(best.split("_")[0][3:]), int(best.split("_g")[1])
        test[best] = run("test", thr, grid, dump=f"{a.dump_dir}/test_{best}.csv")
    print("\n=== TEST ===")
    for k, v in test.items():
        print(f"  {k:12s} MAE {v['mae']:.2f} RMSE {v['rmse']:.2f}")
    json.dump({"val": val, "best_on_val": best, "test": test}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
