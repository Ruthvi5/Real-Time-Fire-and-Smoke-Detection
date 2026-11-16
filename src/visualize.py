"""Summary pictures of a run (one per video), written next to the required output files.

This module never influences detection or alarms: it only looks at frames, predictions and alarm
records that the pipeline has already produced. SummaryCollector works inside the same single pass
over the video, keeping a small, bounded set of downscaled frames, so no second read is needed.
"""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path

import cv2
import numpy as np

COLORS = {"fire": (0, 90, 255), "smoke": (235, 170, 40)}   # BGR: orange-red, blue
OTHER = (160, 160, 160)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- helpers
def ascii_text(s) -> str:
    return str(s).encode("ascii", "replace").decode("ascii")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "video"


def imwrite(path: Path, image: np.ndarray, quality: int = 90) -> None:
    """cv2.imwrite fails on non-ASCII paths on Windows; encode in memory instead."""
    ext = Path(path).suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    Path(path).write_bytes(buf.tobytes())


def put_label(img, text, org, scale=0.6, color=(255, 255, 255), bg=(0, 0, 0), thick=1, pad=4):
    text = ascii_text(text)
    (w, h), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    x = max(0, min(x, img.shape[1] - w - 2 * pad))
    y = max(h + 2 * pad, y)
    cv2.rectangle(img, (x, y - h - 2 * pad), (x + w + 2 * pad, y + base), bg, -1)
    cv2.putText(img, text, (x + pad, y - pad), FONT, scale, color, thick, cv2.LINE_AA)


def resize_to_width(img, width):
    if width <= 0 or img.shape[1] <= width:
        return img
    h = int(round(img.shape[0] * width / img.shape[1]))
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


def shrink(img, long_side):
    """Downscale so the longer side is at most long_side. Returns (image, scale)."""
    h, w = img.shape[:2]
    k = min(1.0, long_side / max(h, w))
    if k >= 1.0:
        return img.copy(), 1.0
    return cv2.resize(img, (int(round(w * k)), int(round(h * k))), interpolation=cv2.INTER_AREA), k


def draw_boxes(image, predictions, scale=1.0):
    """Copy of the frame with its detection boxes; labels sized for a ~640 px wide tile."""
    img = image.copy()
    H, W = img.shape[:2]
    s = max(0.5, min(4.0, max(W, H) / 640))
    thick = max(2, int(round(2 * s)))
    label_h = int(22 * s)
    for p in predictions:
        bb = p.get("bbox_xyxy")
        if not bb:
            continue
        color = COLORS.get(p["label"], OTHER)
        x1, y1, x2, y2 = (int(round(v * scale)) for v in bb)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        ly = y1 - 2 if y1 - 2 >= label_h else y1 + label_h + thick
        put_label(img, f"{p['label']} {p['confidence']:.2f}", (x1, ly), 0.55 * s, bg=color,
                  thick=max(1, int(round(1.2 * s))), pad=max(3, int(3 * s)))
    return img


def fit_tile(img, tw, th):
    """Resize to fit tw x th, keeping the aspect ratio, padded with dark grey."""
    h, w = img.shape[:2]
    k = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * k))), max(1, int(round(h * k)))
    out = np.full((th, tw, 3), 30, np.uint8)
    x, y = (tw - nw) // 2, (th - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def wrap(text, max_chars):
    lines, cur = [], ""
    for w in text.split(" "):
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + ([cur] if cur else [])


def caption_for(timestamp_s, alarms):
    text = f"t = {timestamp_s:.1f} s"
    done = sorted({a["label"] for a in alarms if a["decision_timestamp_s"] <= timestamp_s + 1e-6})
    if done:
        text += "   alarm raised: " + ", ".join(done)
    return text


# ---------------------------------------------------------------- the summary picture
def summary_sheet(source_id, panels, alarms, note=""):
    """Title, list of alarms and captioned frames. panels: list of (image, caption)."""
    h0, w0 = panels[0][0].shape[:2]
    if h0 > w0:   # portrait video: narrow, tall tiles, more per row
        tw, th, cols = 360, min(640, int(round(360 * h0 / w0))), min(6, len(panels))
    else:
        tw, th, cols = 640, min(480, max(270, int(round(640 * h0 / w0)))), 3
    cap_h = 34
    rows = (len(panels) + cols - 1) // cols
    W = max(cols * tw, 1280)
    title = f"{source_id}  ({note})" if note else source_id
    if alarms:
        sub = "Alarms: " + "; ".join(
            f"{a['label']} at {a['decision_timestamp_s']:.1f} s (onset est. "
            f"{a.get('estimated_onset_s', float('nan')):.1f} s, conf {a['confidence']:.2f})" for a in alarms)
    else:
        sub = "No alarms raised"
    sub_lines = wrap(ascii_text(sub), int(W / 14.5))
    head_h = 40 + 26 * len(sub_lines) + 6
    sheet = np.full((head_h + rows * (th + cap_h), W, 3), 255, np.uint8)
    cv2.putText(sheet, ascii_text(title), (10, 30), FONT, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    for i, line in enumerate(sub_lines):
        cv2.putText(sheet, line, (10, 60 + 26 * i), FONT, 0.62, (0, 0, 170) if alarms else (0, 120, 0),
                    1, cv2.LINE_AA)
    for k, (img, caption) in enumerate(panels):
        r, c = divmod(k, cols)
        x, y = c * tw, head_h + r * (th + cap_h)
        sheet[y:y + th, x:x + tw] = fit_tile(img, tw, th)
        cv2.rectangle(sheet, (x, y + th), (x + tw, y + th + cap_h), (28, 28, 28), -1)
        cv2.putText(sheet, ascii_text(caption), (x + 8, y + th + 24), FONT, 0.6, (110, 220, 255), 1,
                    cv2.LINE_AA)
    return sheet


def pick_panels(times, pinned, n_panels, max_panels=9):
    """Indices into `times`: all pinned ones (up to max_panels), then farthest-point fill so the
    picture covers before, during and after the events."""
    picks = sorted(set(pinned))[:max_panels]
    target = max(n_panels, len(picks))
    while len(picks) < min(target, len(times)):
        gaps = [min((abs(t - times[p]) for p in picks), default=1e9) for t in times]
        k = int(np.argmax(gaps))
        if gaps[k] < 0.5:
            break
        picks.append(k)
    return sorted(set(picks))


class SummaryCollector:
    """Collects a few frames during the single pass and writes <name>_summary.jpg at the end.

    Memory is bounded: a reservoir of at most `keep` downscaled frames spread over the video
    (the stride doubles when it fills up), the frames of the last few seconds (for the onset
    frame), and the onset and decision frame of every alarm.
    """

    def __init__(self, source_id, out_path, history_seconds, n_panels=6, note="",
                 long_side=960, keep=40):
        self.source_id = source_id
        self.out_path = Path(out_path)
        self.n_panels = n_panels
        self.note = note
        self.long_side = long_side
        self.keep = keep
        self.history_seconds = history_seconds
        self.recent = deque()
        self.reservoir = []
        self.stride = 1
        self.count = 0
        self.early = None
        self.pinned = {}        # frame_index -> entry (onset and decision frames)
        self.alarms = []

    def add(self, frame_index, timestamp_s, image, predictions):
        small, k = shrink(image, self.long_side)
        entry = (frame_index, timestamp_s, small, k, predictions)
        self.recent.append(entry)
        while self.recent and self.recent[0][1] < timestamp_s - self.history_seconds:
            self.recent.popleft()
        if timestamp_s <= 2.0 or self.early is None:
            self.early = entry
        if self.count % self.stride == 0:
            self.reservoir.append(entry)
            if len(self.reservoir) > self.keep:
                self.reservoir = self.reservoir[::2]
                self.stride *= 2
        self.count += 1

    def alarm(self, alarm):
        self.alarms.append(alarm)
        if self.recent:
            current = self.recent[-1]
            self.pinned[current[0]] = current
            onset = alarm.get("estimated_onset_s")
            if onset is not None:
                e = min(self.recent, key=lambda x: abs(x[1] - onset))
                self.pinned[e[0]] = e

    def finish(self):
        entries = {e[0]: e for e in self.reservoir}
        if self.early is not None:
            entries[self.early[0]] = self.early
        entries.update(self.pinned)
        if not entries:
            return None
        ordered = sorted(entries.values(), key=lambda e: e[0])
        times = [e[1] for e in ordered]
        pin_idx = [i for i, e in enumerate(ordered)
                   if e[0] in self.pinned or (self.early is not None and e[0] == self.early[0])]
        panels = []
        for i in pick_panels(times, pin_idx, self.n_panels):
            fi, ts, small, k, preds = ordered[i]
            panels.append((draw_boxes(small, preds, k), caption_for(ts, self.alarms)))
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        imwrite(self.out_path, summary_sheet(self.source_id, panels, self.alarms, self.note))
        return self.out_path
