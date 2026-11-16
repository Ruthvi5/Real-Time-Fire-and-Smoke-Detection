"""Make pictures from the outputs of infer.py (optional).

infer.py already writes one summary picture per video to <output-dir>/images/ (see config.yaml,
output.summary_image). Use this tool to re-make those pictures from an existing output folder, or
for the extra pictures below. By default it writes one picture per video into <results>/images/:

    <video name>_summary.jpg   title, list of alarms (time, onset, confidence) and 6 captioned frames:
                               an early frame, each onset and alarm frame, and frames spread over the video

With --details it also writes, into <results>/images/<video name>/:
    alarm_<NNN>_<label>.jpg    the frame at decision_timestamp_s (the frame that raised the alarm)
    onset_<NNN>_<label>.jpg    the frame at estimated_onset_s
    overview.jpg               12 frames spread over the whole video
    timeline.png               detection confidence over time, thresholds and alarm times
and with --video:
    annotated.mp4              the sampled frames with boxes and an alarm banner while the alarm is on

Boxes are exactly the ones in predictions.jsonl. The alarm state is found by replaying the same
temporal logic (src/alarms.py) over predictions.jsonl, and is checked against alarms.jsonl.
Needs only the packages already installed for infer.py (not even PyTorch) and works offline.

Usage (same --source and --config as the infer.py run):
    python tools/visualize_results.py --results results --source INPUT.mp4 --config config.yaml
    python tools/visualize_results.py --results results --source videos_folder --config config.yaml --details --video
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.alarms import AlarmManager            # noqa: E402
from src.config import load_config             # noqa: E402
from src.postprocessing import MotionChecker   # noqa: E402
from src.preprocessing import list_videos      # noqa: E402
from src.visualize import (COLORS, FONT, OTHER, ascii_text, caption_for, draw_boxes,  # noqa: E402
                           imwrite, pick_panels, put_label, resize_to_width, safe_name,
                           summary_sheet)

RED = (30, 30, 220)


# ---------------------------------------------------------------- small helpers
def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------- drawing
def draw_frame(image, record, states, alarms_so_far, source_id):
    """Boxes, a time stamp and the alarm banner on a copy of one frame."""
    img = image.copy()
    H, W = img.shape[:2]
    s = max(0.5, min(1.4, W / 1280))
    thick = max(2, int(round(3 * s)))
    bar = 0
    active = [lab for lab, st in states.items() if st in ("ACTIVE", "CLEARING")]
    if active:
        bar = int(46 * s)
        cv2.rectangle(img, (0, 0), (W, bar), RED, -1)
        parts = []
        for lab in active:
            since = alarms_so_far.get(lab)
            parts.append(f"{lab.upper()} ALARM" + (f" (raised at {since:.1f} s)" if since is not None else ""))
        cv2.putText(img, ascii_text("   ".join(parts)), (int(12 * s), int(32 * s)), FONT, 0.9 * s,
                    (255, 255, 255), max(2, int(2 * s)), cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), RED, thick * 2)

    label_h = int(26 * s)
    for p in record["predictions"]:
        color = COLORS.get(p["label"], OTHER)
        bb = p.get("bbox_xyxy")
        if not bb:
            continue
        x1, y1, x2, y2 = (int(v) for v in bb)
        cv2.rectangle(img, (x1, max(y1, bar)), (x2, y2), color, thick)
        ly = y1 - 2 if y1 - 2 >= bar + label_h else max(y1, bar) + label_h + thick
        put_label(img, f"{p['label']} {p['confidence']:.2f}", (x1, ly), 0.6 * s,
                  bg=color, thick=max(1, int(s)))

    put_label(img, f"{source_id}   t = {record['timestamp_s']:.2f} s   frame {record['frame_index']}",
              (8, H - 8), 0.55 * s, bg=(40, 40, 40))
    return img


def alarm_card(image, record, alarm, kind, source_id, states):
    """Decision or onset frame with a caption explaining it."""
    img = draw_frame(image, record, states if kind == "alarm" else {}, {}, source_id)
    H, W = img.shape[:2]
    s = max(0.5, min(1.4, W / 1280))
    if kind == "alarm":
        title = (f"{alarm['label'].upper()} ALARM {alarm['event_id']}: raised at "
                 f"{alarm['decision_timestamp_s']:.2f} s (latest frame used), alarm confidence "
                 f"{alarm['confidence']:.2f}")
        sub = (f"estimated onset {alarm.get('estimated_onset_s', float('nan')):.2f} s, "
               f"{alarm.get('positive_frames', '?')} of {alarm.get('window_frames', '?')} "
               f"frames in the window had a detection (alarm confidence = their mean)")
    else:
        title = (f"Estimated onset of {alarm['event_id']}: {alarm.get('estimated_onset_s', 0):.2f} s "
                 f"(first positive frame in the confirming window)")
        sub = f"alarm raised later, at {alarm['decision_timestamp_s']:.2f} s"
    cap_h = int(70 * s)
    canvas = np.full((H + cap_h, W, 3), 255, np.uint8)
    canvas[cap_h:] = img
    cv2.putText(canvas, ascii_text(title), (int(10 * s), int(28 * s)), FONT, 0.7 * s, (0, 0, 0),
                max(1, int(2 * s)), cv2.LINE_AA)
    cv2.putText(canvas, ascii_text(sub), (int(10 * s), int(58 * s)), FONT, 0.6 * s, (60, 60, 60),
                max(1, int(s)), cv2.LINE_AA)
    return canvas


def timeline(records, alarms, cfg, source_id, width=1400, height=460):
    img = np.full((height, width, 3), 255, np.uint8)
    L, R, T, B = 70, 30, 50, 60
    pw, ph = width - L - R, height - T - B
    duration = max([r["timestamp_s"] for r in records] + [1.0])

    def X(t):
        return int(L + pw * t / duration)

    def Y(c):
        return int(T + ph * (1 - c))

    # grid and axes
    for c in (0, 0.25, 0.5, 0.75, 1.0):
        cv2.line(img, (L, Y(c)), (L + pw, Y(c)), (230, 230, 230), 1)
        cv2.putText(img, f"{c:.2f}", (L - 50, Y(c) + 5), FONT, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    step = next(s for s in (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600)
                if duration / s <= 14)
    t = 0.0
    while t <= duration + 1e-9:
        cv2.line(img, (X(t), T + ph), (X(t), T + ph + 5), (80, 80, 80), 1)
        cv2.putText(img, f"{t:g}", (X(t) - 10, T + ph + 22), FONT, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
        t += step
    cv2.rectangle(img, (L, T), (L + pw, T + ph), (120, 120, 120), 1)
    cv2.putText(img, "time (s)", (L + pw // 2 - 30, height - 12), FONT, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, ascii_text(f"{source_id}: best box confidence per sampled frame"), (L, 30),
                FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    labels = list(cfg["detection"]["confidence"].keys())
    for lab in labels:
        color = COLORS.get(lab, OTHER)
        thr = float(cfg["detection"]["confidence"][lab])
        y = Y(thr)
        for x in range(L, L + pw, 14):
            cv2.line(img, (x, y), (min(x + 7, L + pw), y), color, 1)
        pts = []
        for r in records:
            c = max([p["confidence"] for p in r["predictions"] if p["label"] == lab] or [0.0])
            pts.append((X(r["timestamp_s"]), Y(c)))
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)
        for (x, y), r in zip(pts, records):
            if any(p["label"] == lab for p in r["predictions"]):
                cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)

    for a in alarms:
        color = COLORS.get(a["label"], OTHER)
        xd = X(a["decision_timestamp_s"])
        cv2.line(img, (xd, T), (xd, T + ph), RED, 2)
        on = a.get("estimated_onset_s")
        if on is not None:
            xo = X(on)
            for y in range(T, T + ph, 10):
                cv2.line(img, (xo, y), (xo, min(y + 5, T + ph)), color, 1)
        put_label(img, f"{a['label']} alarm {a['decision_timestamp_s']:.1f}s", (xd + 4, T + 18), 0.45,
                  bg=RED)

    # legend
    x0 = width - R - 420
    for i, lab in enumerate(labels):
        color = COLORS.get(lab, OTHER)
        cv2.line(img, (x0 + i * 120, 30), (x0 + i * 120 + 25, 30), color, 3)
        cv2.putText(img, lab, (x0 + i * 120 + 30, 35), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(img, (x0 + 240, 30), (x0 + 265, 30), RED, 2)
    cv2.putText(img, "alarm", (x0 + 270, 35), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "dashed: threshold / onset", (x0 + 240, 20), FONT, 0.38, (90, 90, 90), 1,
                cv2.LINE_AA)
    return img


def contact_sheet(tiles, cols=4):
    if not tiles:
        return None
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * (th + 6) + 6, cols * (tw + 6) + 6, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = 6 + r * (th + 6), 6 + c * (tw + 6)
        sheet[y:y + t.shape[0], x:x + t.shape[1]] = t
    return sheet


# ---------------------------------------------------------------- main work
def replay_states(records, cfg, source_id):
    """Re-run the alarm logic over predictions.jsonl to know the alarm state at each frame."""
    mgr = AlarmManager(cfg, source_id)
    motion = MotionChecker(cfg)
    states, raised, replayed = {}, {}, []
    for r in records:
        votes = {}
        for p in r["predictions"]:
            if motion.passes(p) and p["confidence"] > (votes.get(p["label"]) or 0.0):
                votes[p["label"]] = p["confidence"]
        for a in mgr.update(r["frame_index"], r["timestamp_s"], votes):
            replayed.append(a)
            raised[a["label"]] = a["decision_timestamp_s"]
        states[r["frame_index"]] = (dict(mgr.states()), dict(raised))
    return states, replayed


def open_writer(path: Path, fps: float, size):
    # mp4v plays in VLC, Windows Media Player and QuickTime, and every OpenCV build can write it.
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if w.isOpened():
        return w
    w.release()
    return None


def process(source_id, video_path, records, alarms, cfg, out_dir, opts):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda r: r["frame_index"])
    states, replayed = replay_states(records, cfg, source_id)
    if [(a["label"], a["decision_timestamp_s"]) for a in replayed] != \
            [(a["label"], a["decision_timestamp_s"]) for a in alarms]:
        print(f"  note: replayed alarms differ from alarms.jsonl for {source_id} "
              f"(was the same config used?)")

    by_index = {r["frame_index"]: r for r in records}
    ts_list = [r["timestamp_s"] for r in records]
    stem = safe_name(Path(source_id).with_suffix("").as_posix())
    details_dir = out_dir / stem

    def nearest(t):
        return records[int(np.argmin([abs(x - t) for x in ts_list]))]["frame_index"]

    wanted = {}   # frame_index -> list of (kind, key, payload)
    ts_arr = np.array(ts_list)
    pinned = []
    if records:
        pinned.append(int(np.argmin(np.abs(ts_arr - min(2.0, ts_arr[-1])))))
        for a in alarms:
            for t in (a.get("estimated_onset_s"), a["decision_timestamp_s"]):
                if t is not None:
                    pinned.append(int(np.argmin(np.abs(ts_arr - t))))
    for k in pick_panels(ts_list, pinned, opts.panels) if records else []:
        wanted.setdefault(records[k]["frame_index"], []).append(("summary", k, None))
    if opts.details:
        for i, a in enumerate(alarms, 1):
            di = a.get("decision_frame_index")
            di = di if di in by_index else nearest(a["decision_timestamp_s"])
            wanted.setdefault(di, []).append(("alarm", i, a))
            if a.get("estimated_onset_s") is not None:
                wanted.setdefault(nearest(a["estimated_onset_s"]), []).append(("onset", i, a))
        if records:
            picks = np.linspace(0, len(records) - 1, min(12, len(records))).round().astype(int)
            for k in sorted(set(picks.tolist())):
                wanted.setdefault(records[k]["frame_index"], []).append(("tile", k, None))

    written, tiles, panels = [], [], []
    writer = None
    make_video = opts.video
    sample_fps = float(cfg["video"].get("sample_fps") or 0) or 5.0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  cannot open video {video_path}")
    else:
        idx = -1
        last = max(by_index) if by_index else -1
        while idx < last:
            if not cap.grab():
                break
            idx += 1
            if idx not in by_index or not (idx in wanted or make_video):
                continue
            ok, image = cap.retrieve()
            if not ok or image is None:
                continue
            rec = by_index[idx]
            st, raised = states.get(idx, ({}, {}))
            if make_video:
                frame = resize_to_width(draw_frame(image, rec, st, raised, source_id), opts.max_width)
                if writer is None:
                    details_dir.mkdir(parents=True, exist_ok=True)
                    writer = open_writer(details_dir / "annotated.mp4", sample_fps,
                                         (frame.shape[1], frame.shape[0]))
                    if writer is None:
                        print("  could not create annotated.mp4 (no video encoder); skipping video")
                        make_video = False
                if writer is not None:
                    writer.write(frame)
            for kind, i, a in wanted.get(idx, []):
                if kind == "summary":
                    small = resize_to_width(image, 960)
                    k = small.shape[1] / image.shape[1]
                    panels.append((draw_boxes(small, rec["predictions"], k),
                                   caption_for(rec["timestamp_s"], alarms)))
                elif kind in ("alarm", "onset"):
                    details_dir.mkdir(parents=True, exist_ok=True)
                    card = resize_to_width(alarm_card(image, rec, a, kind, source_id, st), opts.max_width)
                    pth = details_dir / f"{kind}_{i:03d}_{a['label']}.jpg"
                    imwrite(pth, card)
                    written.append(pth)
                else:
                    tiles.append(resize_to_width(draw_frame(image, rec, st, raised, source_id), 400))
        cap.release()
    if writer is not None:
        writer.release()
        written.append(details_dir / "annotated.mp4")

    if panels:
        pth = out_dir / f"{stem}_summary.jpg"
        imwrite(pth, summary_sheet(source_id, panels, alarms, opts.note), 90)
        written.insert(0, pth)
    if opts.details and tiles:
        th = min(t.shape[0] for t in tiles)
        pth = details_dir / "overview.jpg"
        imwrite(pth, contact_sheet([t[:th] for t in tiles]), 88)
        written.append(pth)
    if opts.details and records:
        details_dir.mkdir(parents=True, exist_ok=True)
        pth = details_dir / "timeline.png"
        imwrite(pth, timeline(records, alarms, cfg, source_id))
        written.append(pth)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", "--output-dir", dest="results", required=True,
                    help="output folder of infer.py (contains predictions.jsonl, alarms.jsonl)")
    ap.add_argument("--source", required=True, help="the same --source given to infer.py (file or folder)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"), help="the same config used by infer.py")
    ap.add_argument("--details", action="store_true",
                    help="also write alarm/onset frames, overview.jpg and timeline.png per video")
    ap.add_argument("--video", action="store_true", help="also write annotated.mp4 per video")
    ap.add_argument("--panels", type=int, default=6, help="frames in each summary picture (default 6)")
    ap.add_argument("--note", default=None, help="text after the video name in the title "
                    "(default: output.summary_note from the config)")
    ap.add_argument("--max-width", type=int, default=1280, help="max width of detail images (default 1280)")
    opts = ap.parse_args(argv)

    results = Path(opts.results).expanduser()
    preds = read_jsonl(results / "predictions.jsonl")
    alarms = read_jsonl(results / "alarms.jsonl")
    if not preds:
        sys.exit(f"no predictions found in {results / 'predictions.jsonl'}; run infer.py first")
    cfg = load_config(opts.config)
    if opts.note is None:
        opts.note = str(cfg.get("output", {}).get("summary_note", ""))
    videos = dict(list_videos(opts.source))

    p_by, a_by = defaultdict(list), defaultdict(list)
    for r in preds:
        p_by[r["source_id"]].append(r)
    for a in alarms:
        a_by[a["source_id"]].append(a)

    out_dir = results / "images"
    total = []
    for sid in sorted(p_by):
        if sid not in videos:
            print(f"skip {sid}: not found under --source {opts.source}")
            continue
        print(f"{sid}: {len(p_by[sid])} frames, {len(a_by.get(sid, []))} alarm(s)")
        files = process(sid, videos[sid], p_by[sid], a_by.get(sid, []), cfg, out_dir, opts)
        for f in files:
            print(f"   {f}")
        total += files
    print(f"done: {len(total)} files in {out_dir}")


if __name__ == "__main__":
    main()
