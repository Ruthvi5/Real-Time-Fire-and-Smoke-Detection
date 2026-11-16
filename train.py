#!/usr/bin/env python3
"""Fine-tune a YOLO11 detector for fire and smoke.

Example (Colab / Kaggle GPU):
    python train.py --data dataset/data.yaml --model yolo11s.pt --epochs 60 --batch 16 --device 0 --name y11s_dfire

The best validation checkpoint is copied to models/<name>.pt. Copy the one you
finally choose to models/best.pt (the path in config.yaml).

Surveillance-style augmentation: besides the Ultralytics defaults we add
grayscale (night IR cameras), blur, JPEG compression and sensor noise, so a model
trained on clean photos copes better with CCTV footage. Needs `albumentations`;
without it training still runs with the default augmentation only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cctv_augmentations(strength: float = 1.0):
    try:
        import albumentations as A
    except ImportError:
        print("albumentations not installed: using default Ultralytics augmentation only", file=sys.stderr)
        return None
    s = strength
    aug = [
        A.ToGray(p=0.15 * s),
        A.OneOf([A.MotionBlur(blur_limit=5), A.GaussianBlur(blur_limit=(3, 5))], p=0.15 * s),
        A.ImageCompression(quality_range=(30, 75), p=0.25 * s),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.10 * s),  # mild sensor noise (library default is very strong)
        A.CLAHE(p=0.05 * s),
    ]
    return aug


def portable_data_yaml(data_yaml: Path) -> Path:
    """Point data.yaml's `path` at the folder it lives in, so the dataset can be moved (Mac -> Colab)."""
    import yaml
    data_yaml = data_yaml.resolve()
    d = yaml.safe_load(data_yaml.read_text())
    if Path(str(d.get("path", ""))) == data_yaml.parent:
        return data_yaml
    d["path"] = str(data_yaml.parent)
    fixed = data_yaml.parent / "data_resolved.yaml"
    fixed.write_text(yaml.safe_dump(d, sort_keys=False))
    return fixed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--model", default="yolo11s.pt", help="pretrained starting weights, e.g. yolo11n.pt / yolo11s.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="0 for first GPU, cpu, mps")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="fire_smoke")
    p.add_argument("--project", default=str(ROOT / "runs"))
    p.add_argument("--aug_strength", type=float, default=1.0, help="0 disables the extra CCTV augmentation")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    args.data = str(portable_data_yaml(Path(args.data)))
    model = YOLO(args.model)
    extra = cctv_augmentations(args.aug_strength) if args.aug_strength > 0 else None
    train_kwargs = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=args.workers, patience=args.patience, seed=args.seed, deterministic=True,
        project=args.project, name=args.name, exist_ok=False, resume=args.resume,
        # Fire and smoke have no left/right meaning, so horizontal flip is safe; vertical flip is not.
        fliplr=0.5, flipud=0.0,
        # Moderate colour jitter: too much hue shift turns orange lamps into "fire" and vice versa.
        hsv_h=0.010, hsv_s=0.6, hsv_v=0.4,
        mosaic=1.0, close_mosaic=10, plots=True,
    )
    if extra:
        train_kwargs["augmentations"] = extra

    model.train(**train_kwargs)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    dest = ROOT / "models" / f"{args.name}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest)

    # Record exactly how this model was trained, for the report.
    record = {k: v for k, v in vars(args).items()}
    record["extra_augmentations"] = [type(a).__name__ for a in (extra or [])]
    record["save_dir"] = str(save_dir)
    metrics = YOLO(str(dest)).val(data=args.data, imgsz=args.imgsz, device=args.device, split="val", plots=False)
    record["val"] = {"mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
                     "precision": float(metrics.box.mp), "recall": float(metrics.box.mr)}
    (ROOT / "models" / f"{args.name}.json").write_text(json.dumps(record, indent=2))
    print(f"best weights copied to {dest}")
    print(json.dumps(record["val"], indent=2))


if __name__ == "__main__":
    main()
