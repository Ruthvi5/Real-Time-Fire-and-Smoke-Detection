#!/usr/bin/env python3
"""Build dataset/ (YOLO format) from D-Fire, check class IDs and make a leakage-aware val split.

D-Fire (https://github.com/gaiasd/DFireDataset) unpacks to something like
    D-Fire/train/images  D-Fire/train/labels  D-Fire/test/images  D-Fire/test/labels

Usage
    python tools/prepare_dataset.py --src /path/to/D-Fire --out dataset --names smoke,fire

What it does
  1. keeps the official test split untouched as our test set
  2. carves a validation set out of train, grouping images whose names are
     consecutive numbers (likely frames of the same video) so near-duplicate
     frames never land on both sides of the split
  3. counts boxes per class ID and writes preview images with the boxes drawn,
     so you can confirm by eye which ID is fire and which is smoke
  4. writes dataset/data.yaml and dataset/split_manifest.json (for the report)

--names gives the class names in ID order *as they are in the source labels*.
infer.py maps classes by name, so we never need to rewrite label files.
D-Fire publishes 14,692 fire boxes and 11,865 smoke boxes; the script compares
its counts with those numbers and warns if --names looks reversed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DFIRE_BOX_TOTALS = {"fire": 14692, "smoke": 11865}


def find_split(root: Path, split: str):
    for cand in (root / split / "images", root / "images" / split, root / split):
        if cand.is_dir() and any(p.suffix.lower() in IMG_EXT for p in cand.iterdir()):
            return cand
    return None


def label_for(img: Path) -> Path:
    # .../images/x.jpg -> .../labels/x.txt  (works for both layouts)
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def group_key(name: str, block: int) -> str:
    m = re.match(r"^(.*?)(\d+)$", Path(name).stem)
    if not m:
        return Path(name).stem
    return f"{m.group(1)}#{int(m.group(2)) // block}"


def read_labels(path: Path):
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                rows.append((int(float(parts[0])), *map(float, parts[1:])))
    return rows


def place(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def draw_previews(items, names, out_dir: Path, n=24, seed=0):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = [(0, 0, 255), (200, 200, 200), (0, 255, 0), (255, 0, 0)]
    pool = [it for it in items if it[1]]
    random.Random(seed).shuffle(pool)
    for img_path, rows in pool[:n]:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for cls, cx, cy, bw, bh in rows:
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            c = colors[cls % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            name = names[cls] if cls < len(names) else str(cls)
            cv2.putText(img, f"{cls}:{name}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        cv2.imwrite(str(out_dir / img_path.name), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="D-Fire root folder")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--names", default="smoke,fire", help="class names in source ID order")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--group_block", type=int, default=50,
                    help="images whose trailing numbers fall in the same block of this size stay together")
    ap.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    names = [n.strip() for n in args.names.split(",")]

    train_dir, test_dir = find_split(src, "train"), find_split(src, "test")
    val_dir = find_split(src, "val") or find_split(src, "valid")
    if train_dir is None:
        raise SystemExit(f"could not find train images under {src}")

    def collect(d):
        return [(p, read_labels(label_for(p))) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT]

    train_items = collect(train_dir)
    test_items = collect(test_dir) if test_dir else []
    if val_dir:
        val_items = collect(val_dir)
        val_note = "official validation split"
    else:
        groups = defaultdict(list)
        for it in train_items:
            groups[group_key(it[0].name, args.group_block)].append(it)
        keys = sorted(groups)
        random.Random(args.seed).shuffle(keys)
        target = args.val_frac * len(train_items)
        val_keys, count = set(), 0
        for k in keys:
            if count >= target:
                break
            val_keys.add(k)
            count += len(groups[k])
        val_items = [it for k in sorted(val_keys) for it in groups[k]]
        train_items = [it for k in keys if k not in val_keys for it in groups[k]]
        val_note = f"grouped split from train (block={args.group_block}, seed={args.seed})"

    splits = {"train": train_items, "val": val_items, "test": test_items}
    manifest = {"source": str(src.resolve()), "names": names, "val_split": val_note, "splits": {}}
    all_counts = Counter()
    for split, items in splits.items():
        boxes, empty = Counter(), 0
        for img, rows in items:
            place(img, out / "images" / split / img.name, args.mode)
            lbl_src = label_for(img)
            lbl_dst = out / "labels" / split / (img.stem + ".txt")
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            if lbl_src.is_file():
                shutil.copy2(lbl_src, lbl_dst)
            else:
                lbl_dst.write_text("")  # negative image
            if not rows:
                empty += 1
            boxes.update(r[0] for r in rows)
        all_counts.update(boxes)
        manifest["splits"][split] = {
            "images": len(items), "negative_images": empty,
            "boxes_per_class": {names[c] if c < len(names) else str(c): n for c, n in sorted(boxes.items())},
            "files": sorted(img.name for img, _ in items),
        }
        print(f"{split:5s}: {len(items):6d} images, {empty:5d} negatives, boxes {dict(sorted(boxes.items()))}")

    # Sanity check of the class order against D-Fire's published totals.
    by_name = {names[c]: n for c, n in all_counts.items() if c < len(names)}
    if set(DFIRE_BOX_TOTALS) <= set(by_name):
        # D-Fire has about 1.24 fire boxes per smoke box; reversed names flip that ratio.
        swapped = by_name["fire"] < by_name["smoke"]
        print(f"box totals by name: {by_name}  (full D-Fire publishes {DFIRE_BOX_TOTALS})")
        if swapped:
            print("WARNING: counts suggest --names is REVERSED. Check the previews before training.")
    unknown = [c for c in all_counts if c >= len(names)]
    if unknown:
        print(f"WARNING: label files use class IDs {unknown} not covered by --names")

    draw_previews(splits["train"], names, out / "previews")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out/'data.yaml'}; open {out/'previews'} and confirm the box labels look right")


if __name__ == "__main__":
    main()
