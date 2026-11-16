#!/usr/bin/env python3
"""Event-level evaluation and threshold tuning on annotated validation videos.

1. Write annotations.csv for your validation clips, one row per event:
       source_id,label,onset_s,end_s
       kitchen_fire.mp4,fire,4.0,30.0
       kitchen_fire.mp4,smoke,2.5,30.0
       street_night.mp4,,,
   A row with an empty label marks a normal video (no events). Every video must
   appear at least once. end_s may be empty (= until the end of the video).

2. Run inference ONCE with low thresholds so every candidate box is cached, e.g.
   a copy of config.yaml with detection.confidence fire/smoke = 0.10:
       python infer.py --source clips/val --config tools/dev_lowconf.yaml --output-dir outputs/val_raw --device cpu --no-display

3a. Evaluate one setting (the frozen config):
       python tools/evaluate_videos.py --predictions outputs/val_raw/predictions.jsonl \
           --annotations clips/val/annotations.csv --config config.yaml

3b. Sweep settings (the temporal logic is replayed from the cache, so this is fast):
       python tools/evaluate_videos.py ... --config config.yaml \
           --set detection.confidence.fire=0.3,0.4,0.5 \
           --set temporal.fire.window_seconds=1,2,3 \
           --set temporal.fire.min_positive_ratio=0.5,0.6,0.8 \
           --set motion.enabled=false,true --csv outputs/sweep.csv

Only ever tune on validation clips. Freeze config.yaml before looking at test clips.

Metrics
  event recall      share of annotated events that got an alarm of the same label
                    no earlier than onset_s - tolerance and before end_s + grace
  false alarms      alarms that match no annotated event (incl. any alarm on a normal video)
  FA per hour       false alarms per hour of processed video
  delay             decision_timestamp_s - onset_s for detected events
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.alarms import AlarmManager  # noqa: E402
from src.config import LABELS, load_config, validate  # noqa: E402
from src.postprocessing import box_passes  # noqa: E402


def load_annotations(path):
    events, videos = [], set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = row["source_id"].strip()
            videos.add(sid)
            label = (row.get("label") or "").strip().lower()
            if label in ("", "none", "normal"):
                continue
            onset = float(row["onset_s"])
            end = float(row["end_s"]) if (row.get("end_s") or "").strip() else float("inf")
            events.append({"source_id": sid, "label": label, "onset_s": onset, "end_s": end})
    return events, videos


def load_predictions(path):
    by_src = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                by_src[r["source_id"]].append(r)
    for rows in by_src.values():
        rows.sort(key=lambda r: r["timestamp_s"])
    return by_src


def replay(cfg, by_src):
    """Re-run postprocessing filters and the alarm logic on cached predictions."""
    det = cfg["detection"]
    mot = cfg["motion"]
    alarms = []
    for sid, rows in by_src.items():
        mgr = AlarmManager(cfg, sid)
        for r in rows:
            votes = {}
            for p in r["predictions"]:
                lab = p["label"]
                if not box_passes(lab, p["confidence"], p.get("area_frac", 0.0), det):
                    continue
                if mot["enabled"]:
                    s = p.get("motion_score")
                    if s is not None and s < mot["min_mean_abs_diff"].get(lab, 0.0):
                        continue
                if p["confidence"] > (votes.get(lab) or 0.0):
                    votes[lab] = p["confidence"]
            alarms += mgr.update(r["frame_index"], r["timestamp_s"], votes)
    return alarms


def score(alarms, events, by_src, tolerance=0.5, grace=2.0):
    hours = sum((rows[-1]["timestamp_s"] - rows[0]["timestamp_s"]) for rows in by_src.values() if rows) / 3600
    used, detected, delays = set(), defaultdict(int), defaultdict(list)
    totals = defaultdict(int)
    for ev in events:
        totals[ev["label"]] += 1
        cands = [i for i, a in enumerate(alarms)
                 if i not in used and a["source_id"] == ev["source_id"] and a["label"] == ev["label"]
                 and ev["onset_s"] - tolerance <= a["decision_timestamp_s"] <= ev["end_s"] + grace]
        if cands:
            i = min(cands, key=lambda j: alarms[j]["decision_timestamp_s"])
            used.add(i)
            detected[ev["label"]] += 1
            delays[ev["label"]].append(max(0.0, alarms[i]["decision_timestamp_s"] - ev["onset_s"]))
    fa = defaultdict(int)
    for i, a in enumerate(alarms):
        if i not in used:
            fa[a["label"]] += 1
    out = {"video_hours": round(hours, 4), "alarms": len(alarms)}
    all_delays = []
    for lab in LABELS:
        out[f"{lab}_recall"] = round(detected[lab] / totals[lab], 3) if totals[lab] else None
        out[f"{lab}_events"] = totals[lab]
        out[f"{lab}_false_alarms"] = fa[lab]
        out[f"{lab}_median_delay_s"] = round(statistics.median(delays[lab]), 2) if delays[lab] else None
        all_delays += delays[lab]
    n_ev = sum(totals.values())
    out["event_recall"] = round(sum(detected.values()) / n_ev, 3) if n_ev else None
    out["false_alarms"] = sum(fa.values())
    out["fa_per_hour"] = round(out["false_alarms"] / hours, 2) if hours > 0 else None
    out["median_delay_s"] = round(statistics.median(all_delays), 2) if all_delays else None
    return out


def set_key(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def parse_value(s):
    s = s.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        return float(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    ap.add_argument("--alarms", help="score this alarms.jsonl directly instead of replaying")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=V1,V2", help="grid values to sweep")
    ap.add_argument("--tolerance", type=float, default=0.5, help="seconds an alarm may precede onset_s")
    ap.add_argument("--grace", type=float, default=2.0, help="seconds after end_s an alarm still counts")
    ap.add_argument("--max_fa_per_hour", type=float, default=None, help="when sweeping, drop settings above this")
    ap.add_argument("--csv", help="write all sweep results here")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    events, ann_videos = load_annotations(args.annotations)
    by_src = load_predictions(args.predictions)
    missing = ann_videos - set(by_src)
    if missing:
        print(f"WARNING: annotated videos with no predictions: {sorted(missing)}")
    unannotated = set(by_src) - ann_videos
    if unannotated:
        print(f"WARNING: videos without annotations are ignored: {sorted(unannotated)}")
        by_src = {k: v for k, v in by_src.items() if k in ann_videos}

    base = load_config(args.config)

    if args.alarms:
        alarms = [json.loads(line) for line in open(args.alarms) if line.strip()]
        print(json.dumps(score(alarms, events, by_src, args.tolerance, args.grace), indent=2))
        return

    grid_keys, grid_vals = [], []
    for item in args.set:
        k, v = item.split("=", 1)
        grid_keys.append(k)
        grid_vals.append([parse_value(x) for x in v.split(",")])

    results = []
    for combo in itertools.product(*grid_vals) if grid_keys else [()]:
        cfg = copy.deepcopy(base)
        for k, v in zip(grid_keys, combo):
            set_key(cfg, k, v)
        try:
            validate(cfg)
        except ValueError:
            continue
        res = score(replay(cfg, by_src), events, by_src, args.tolerance, args.grace)
        res.update({k: v for k, v in zip(grid_keys, combo)})
        results.append(res)

    if not grid_keys:
        print(json.dumps(results[0], indent=2))
        return

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)
    kept = [r for r in results if args.max_fa_per_hour is None or (r["fa_per_hour"] or 0) <= args.max_fa_per_hour]
    kept.sort(key=lambda r: (-(r["event_recall"] or 0), r["false_alarms"], r["median_delay_s"] or 1e9))
    cols = grid_keys + ["event_recall", "false_alarms", "fa_per_hour", "median_delay_s"]
    print(f"{len(results)} settings evaluated, {len(kept)} kept; best first:")
    print("  ".join(cols))
    for r in kept[:args.top]:
        print("  ".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
