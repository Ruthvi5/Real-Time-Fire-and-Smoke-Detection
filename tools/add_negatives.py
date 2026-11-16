#!/usr/bin/env python3
"""Add hard-negative frames (no fire, no smoke) from normal videos to the training set.

Each extracted frame gets an empty label file, which teaches YOLO that headlights,
street lamps, sunsets, orange clothing, steam and fog are NOT fire or smoke.

Usage
    python tools/add_negatives.py --videos clips/normal_train --out dataset --split train --every 2.0

Only use videos that are not in your validation or test clips (whole videos, never
individual frames, go to one split). The list of extracted frames is appended to
dataset/negatives_manifest.json for the report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import VideoReader, list_videos  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="video file or folder of NORMAL videos")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--every", type=float, default=2.0, help="seconds between extracted frames")
    ap.add_argument("--max_per_video", type=int, default=200)
    ap.add_argument("--max_side", type=int, default=1280, help="downscale large frames to save disk")
    args = ap.parse_args()

    out = Path(args.out)
    img_dir, lbl_dir = out / "images" / args.split, out / "labels" / args.split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "negatives_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    total = 0
    for source_id, path in list_videos(args.videos):
        tag = re.sub(r"[^A-Za-z0-9]+", "_", Path(source_id).with_suffix("").as_posix())
        n = 0
        for frame in VideoReader(path, sample_fps=1.0 / args.every):
            img = frame.image
            h, w = img.shape[:2]
            s = args.max_side / max(h, w)
            if s < 1:
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            name = f"neg_{tag}_{frame.frame_index:06d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (lbl_dir / f"{name}.txt").write_text("")
            n += 1
            if n >= args.max_per_video:
                break
        manifest[source_id] = {"split": args.split, "frames": n, "every_s": args.every}
        total += n
        print(f"{source_id}: {n} frames")
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"added {total} negative frames to {img_dir}")


if __name__ == "__main__":
    main()
