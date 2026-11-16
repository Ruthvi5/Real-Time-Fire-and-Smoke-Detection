#!/usr/bin/env python3
"""Video-level comparison table for the report (frame-wise vs temporal vs final config).

Replays the alarm logic on a cached low-threshold inference run, so every variant is scored on
exactly the same detections. Annotation format is the one used by tools/evaluate_videos.py; a row
with an empty label marks a normal video.

1. Cache detections once per model (low thresholds, nothing is filtered yet):
    python infer.py --source clips/dfire_videos/Videos --config tools/dev_lowconf.yaml \
        --output-dir results/cache_y11n --model models/y11n_dfire.pt --device mps --no-display
2. Score the variants on the official split you want (test only for final reporting):
    python tools/video_ablation.py --predictions results/cache_y11n/predictions.jsonl \
        --annotations clips/dfire_videos/annotations_test.csv --out docs/video_eval/y11n_test.md

Definitions
  caught          an event video with at least one alarm of the annotated label
  false alarms    every alarm on a normal video, every alarm of a label the video is not
                  annotated with, and every repeated alarm of the same label in one video
  delay           first matching alarm time minus the annotated onset (0 = start of clip)
"""

from __future__ import annotations

import argparse
import collections
import copy
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_videos import load_annotations, load_predictions, replay  # noqa: E402
from src.config import load_config  # noqa: E402

VARIANTS = [
    "frame-wise (alarm on any detection)",
    "temporal, no large-box rule",
    "final config",
]


def make_variant(final: dict, name: str) -> dict:
    c = copy.deepcopy(final)
    if name == "frame-wise (alarm on any detection)":
        c["detection"]["large_box"]["area_frac"] = 0.0
        for lab in ("fire", "smoke"):
            c["temporal"][lab].update(window_seconds=0.001, min_positive_ratio=1.0, min_positive_frames=1)
    elif name == "temporal, no large-box rule":
        c["detection"]["large_box"]["area_frac"] = 0.0
    return c


def score(alarms, events, by_src):
    hours = sum(r[-1]["timestamp_s"] - r[0]["timestamp_s"] for r in by_src.values() if r) / 3600
    ev_by = collections.defaultdict(list)
    for e in events:
        ev_by[e["source_id"]].append(e)
    al_by = collections.defaultdict(list)
    for a in alarms:
        al_by[a["source_id"]].append(a)
    event_videos = [s for s in by_src if ev_by[s]]
    normal_videos = [s for s in by_src if not ev_by[s]]
    caught, delays, false_alarms = 0, [], []
    for s in by_src:
        labels = {e["label"]: e for e in ev_by[s]}
        seen = set()
        for a in sorted(al_by[s], key=lambda x: x["decision_timestamp_s"]):
            if a["label"] in labels and a["label"] not in seen:
                seen.add(a["label"])
                delays.append(max(0.0, a["decision_timestamp_s"] - labels[a["label"]]["onset_s"]))
            else:
                false_alarms.append(a)
        if labels and seen == set(labels):
            caught += 1
    return {
        "caught": f"{caught}/{len(event_videos)}",
        "normal_with_alarm": f"{sum(1 for s in normal_videos if al_by[s])}/{len(normal_videos)}",
        "false_alarms": len(false_alarms),
        "fire_fa": sum(a["label"] == "fire" for a in false_alarms),
        "smoke_fa": sum(a["label"] == "smoke" for a in false_alarms),
        "fa_per_hour": round(len(false_alarms) / hours, 1) if hours else None,
        "median_delay_s": round(statistics.median(delays), 2) if delays else None,
        "minutes": round(hours * 60, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", help="also write the table as Markdown here")
    args = ap.parse_args()

    events, videos = load_annotations(args.annotations)
    by_src = {k: v for k, v in load_predictions(args.predictions).items() if k in videos}
    missing = videos - set(by_src)
    if missing:
        print(f"WARNING: {len(missing)} annotated videos have no predictions: {sorted(missing)[:5]} ...")
    final = load_config(args.config)

    lines = [f"Source: `{args.predictions}`, annotations `{args.annotations}`, config `{args.config}`",
             "",
             "| Variant | Event videos caught | Normal videos with an alarm | False alarms (fire/smoke) | False alarms per hour | Median delay (s) |",
             "|---|---|---|---|---|---|"]
    for name in VARIANTS:
        r = score(replay(make_variant(final, name), by_src), events, by_src)
        lines.append(f"| {name} | {r['caught']} | {r['normal_with_alarm']} | {r['false_alarms']} "
                     f"({r['fire_fa']}/{r['smoke_fa']}) | {r['fa_per_hour']} | {r['median_delay_s']} |")
    lines.append(f"\n{len(by_src)} videos, {r['minutes']} minutes.")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
