#!/usr/bin/env python3
"""Image-level evaluation of a trained detector, with all plots saved for the report.

Runs Ultralytics validation on one split of dataset/data.yaml and stores, in --out:
    metrics.json          precision, recall, mAP50, mAP50-95 overall and per class, speed, settings
    confusion_matrix.png, confusion_matrix_normalized.png
    BoxPR_curve.png, BoxF1_curve.png, BoxP_curve.png, BoxR_curve.png
    val_batch*_labels.jpg / val_batch*_pred.jpg   (ground truth vs prediction examples)

Usage (from the project folder):
    python tools/evaluate_images.py --model models/y11n_dfire.pt --split test --device mps --out docs/image_eval/y11n_test
    python tools/evaluate_images.py --model models/y11s_dfire.pt --split test --device mps --out docs/image_eval/y11s_test
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_OFFLINE", "true")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT / "dataset" / "data.yaml"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.001, help="Ultralytics default for mAP")
    ap.add_argument("--iou", type=float, default=0.7, help="Ultralytics default NMS IoU for validation")
    ap.add_argument("--device", default="cpu", help="cpu, mps, 0 ...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from ultralytics import YOLO
    from train import portable_data_yaml

    data = portable_data_yaml(Path(args.data))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        model = YOLO(args.model)
        m = model.val(data=str(data), split=args.split, imgsz=args.imgsz, batch=args.batch,
                      conf=args.conf, iou=args.iou, device=args.device, plots=True,
                      project=tmp, name="eval", exist_ok=True, verbose=False)
        save_dir = Path(m.save_dir)
        for f in sorted(save_dir.iterdir()):
            if f.suffix.lower() in (".png", ".jpg"):
                shutil.copy2(f, out / f.name)

    names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                    "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)}
    result = {
        "model": str(Path(args.model).name),
        "split": args.split,
        "data": str(data),
        "settings": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "device": args.device},
        "all": {"precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
                "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4)},
        "per_class": per_class,
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in m.speed.items()},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"plots and metrics saved in {out}")


if __name__ == "__main__":
    main()
