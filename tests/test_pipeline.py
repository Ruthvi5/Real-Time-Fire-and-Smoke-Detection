"""Checks for the temporal logic and the output contract. Run: python -m pytest tests -q

No trained model is needed: a fake colour-threshold detector stands in for YOLO.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.pipeline import run  # noqa: E402
from src.temporal import ACTIVE, CLEAR, ClassTracker  # noqa: E402


# ---------------------------------------------------------------- temporal logic

def feed(tracker, pattern, fps=5.0):
    """pattern: string of 1/0 per sampled frame. Returns list of (index, Confirmation)."""
    fired = []
    for i, ch in enumerate(pattern):
        r = tracker.update(i / fps, i, ch == "1", 0.8 if ch == "1" else 0.0)
        if r:
            fired.append((i, r))
    return fired


def make_tracker(**kw):
    args = dict(window_seconds=0.8, min_positive_ratio=0.6, min_positive_frames=3,
                clear_after_seconds=1.0, cooldown_seconds=0.0)
    args.update(kw)
    return ClassTracker(**args)


def test_three_of_five_fires_once_at_latest_frame():
    # 0.8 s window at 5 fps = current + previous 4 frames.
    fired = feed(make_tracker(), "1011111111")
    assert len(fired) == 1
    idx, c = fired[0]
    assert idx == 3                      # frames 0, 2, 3 positive -> third positive confirms
    assert c.decision_timestamp_s == pytest.approx(0.6)
    assert c.estimated_onset_s == pytest.approx(0.0)
    assert c.estimated_onset_s <= c.decision_timestamp_s


def test_isolated_blips_never_alarm():
    assert feed(make_tracker(), "1000100010001000100") == []


def test_new_event_after_clearing_raises_second_alarm():
    t = make_tracker()
    # event, 10 empty frames (2 s > clear_after 1 s), second event
    fired = feed(t, "11111" + "0" * 10 + "11111")
    assert [i for i, _ in fired] == [2, 17]


def test_short_gap_does_not_duplicate_alarm():
    fired = feed(make_tracker(), "11111" + "00" + "11111")
    assert len(fired) == 1


def test_cooldown_blocks_quick_realarm():
    t = make_tracker(cooldown_seconds=10.0)
    fired = feed(t, "11111" + "0" * 10 + "11111")
    assert len(fired) == 1


def test_state_reset():
    t = make_tracker()
    feed(t, "11111")
    assert t.state == ACTIVE
    t.reset()
    assert t.state == CLEAR and len(t.window) == 0


# ---------------------------------------------------------------- end to end

class OrangeBlobDetector:
    """Stand-in for YOLO: reports 'fire' wherever strongly orange pixels exist."""
    warnings = []

    def __call__(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (5, 150, 150), (25, 255, 255))
        if cv2.countNonZero(mask) < 50:
            return []
        x, y, w, h = cv2.boundingRect(mask)
        return [{"label": "fire", "confidence": 0.9, "bbox_xyxy": [x, y, x + w, y + h]}]


def write_video(path, fps, seconds, fire_from=None, fire_to=None, flicker=True):
    w, h = 320, 240
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    for i in range(int(fps * seconds)):
        t = i / fps
        img = np.full((h, w, 3), 60, np.uint8)
        cv2.rectangle(img, (20, 150), (300, 230), (90, 90, 90), -1)
        if fire_from is not None and fire_from <= t < fire_to:
            r = int(25 + (rng.integers(0, 12) if flicker else 0))
            cv2.circle(img, (160, 120), r, (0, 140, 255), -1)
        vw.write(img)
    vw.release()


@pytest.fixture()
def cfg():
    return load_config(ROOT / "config.yaml")


def read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def test_end_to_end_folder(tmp_path, cfg):
    vids = tmp_path / "videos"
    vids.mkdir()
    write_video(vids / "event25.mp4", 25, 10, fire_from=3.0, fire_to=8.0)
    write_video(vids / "event10.mp4", 10, 10, fire_from=3.0, fire_to=8.0)
    write_video(vids / "normal.mp4", 25, 6)
    (vids / "broken.mp4").write_bytes(b"not a video at all")

    out = tmp_path / "out"
    summary = run(str(vids), str(out), cfg, OrangeBlobDetector(), device="test")

    preds = read_jsonl(out / "predictions.jsonl")
    alarms = read_jsonl(out / "alarms.jsonl")
    on_disk = json.loads((out / "run_summary.json").read_text())

    # summary contract
    assert on_disk["status"] == "completed_with_errors"          # broken.mp4 is reported, not fatal
    assert any(e["source_id"] == "broken.mp4" for e in on_disk["errors"])
    assert on_disk["frames_read"] == on_disk["frames_processed"] + on_disk["frames_skipped"]
    assert on_disk["frames_processed"] == len(preds)
    assert summary["videos_completed"] == 3

    # predictions contract: one per processed frame, increasing time per video
    for r in preds:
        assert set(r) >= {"source_id", "frame_index", "timestamp_s", "predictions"}
        for p in r["predictions"]:
            assert p["label"] in ("fire", "smoke") and 0 <= p["confidence"] <= 1
    for sid in ("event25.mp4", "event10.mp4", "normal.mp4"):
        ts = [r["timestamp_s"] for r in preds if r["source_id"] == sid]
        assert ts == sorted(ts) and len(set(ts)) == len(ts)
        # ~5 sampled frames per second regardless of source fps
        dur = 10 if sid.startswith("event") else 6
        assert abs(len(ts) - 5 * dur) <= 1, (sid, len(ts))

    # alarms: exactly one per event video, none on the normal clip, decided after onset
    by_src = {a["source_id"]: a for a in alarms}
    assert len(alarms) == 2 and set(by_src) == {"event25.mp4", "event10.mp4"}
    for a in alarms:
        assert set(a) >= {"source_id", "event_id", "label", "decision_timestamp_s", "confidence"}
        assert a["label"] == "fire"
        assert 3.0 <= a["estimated_onset_s"] <= a["decision_timestamp_s"] <= 5.0
        # decision time is a real processed frame of that video
        assert any(r["source_id"] == a["source_id"] and r["timestamp_s"] == a["decision_timestamp_s"]
                   for r in preds)

    # one summary picture per readable video, listed in run_summary.json
    images = sorted(p.name for p in (out / "images").glob("*.jpg"))
    assert images == ["event10_summary.jpg", "event25_summary.jpg", "normal_summary.jpg"]
    assert all(v["summary_image"] == f"images/{Path(v['source_id']).stem}_summary.jpg"
               for v in on_disk["videos"])

    # a failure while making pictures must not change any decision or stop the run
    import src.visualize as viz

    def broken(*args, **kwargs):
        raise RuntimeError("simulated drawing failure")

    original = viz.SummaryCollector.add
    viz.SummaryCollector.add = broken
    try:
        out2 = tmp_path / "out2"
        run(str(vids), str(out2), cfg, OrangeBlobDetector(), device="test")
    finally:
        viz.SummaryCollector.add = original
    assert (out2 / "alarms.jsonl").read_text() == (out / "alarms.jsonl").read_text()
    assert (out2 / "predictions.jsonl").read_text() == (out / "predictions.jsonl").read_text()
    s2 = json.loads((out2 / "run_summary.json").read_text())
    assert s2["status"] == "completed_with_errors" and s2["videos_completed"] == 3
    assert all(v["summary_image"] is None and v["warnings"] for v in s2["videos"])


def test_motion_check_rejects_static_lookalike(tmp_path, cfg):
    write_video(tmp_path / "lamp.mp4", 25, 8, fire_from=0.0, fire_to=8.0, flicker=False)
    write_video(tmp_path / "flame.mp4", 25, 8, fire_from=2.0, fire_to=8.0, flicker=True)
    cfg["motion"]["enabled"] = True
    out = tmp_path / "out"
    run(str(tmp_path), str(out), cfg, OrangeBlobDetector())
    alarms = read_jsonl(out / "alarms.jsonl")
    assert [a["source_id"] for a in alarms] == ["flame.mp4"]


def test_missing_input_still_writes_outputs(tmp_path, cfg):
    out = tmp_path / "out"
    s = run(str(tmp_path / "nope.mp4"), str(out), cfg, OrangeBlobDetector())
    assert s["status"] == "failed"
    assert (out / "predictions.jsonl").exists() and (out / "alarms.jsonl").exists()


# ---------------------------------------------------------------- course command

def test_official_command_line_parses():
    import infer
    a = infer.parse_args(["--source", "INPUT.mp4", "--config", "config.yaml",
                          "--output-dir", "results", "--device", "cuda:0", "--no-display"])
    assert (a.input, a.config, a.output_dir, a.device, a.no_display) == \
        ("INPUT.mp4", "config.yaml", "results", "cuda:0", True)


# ---------------------------------------------------------------- box rules

def test_large_box_needs_higher_confidence():
    from src.postprocessing import box_passes
    det = {"confidence": {"fire": 0.35, "smoke": 0.3}, "min_box_area_frac": 0.0,
           "large_box": {"area_frac": 0.25, "min_confidence": {"fire": 0.0, "smoke": 0.65}}}
    assert box_passes("smoke", 0.40, 0.05, det)        # small box, normal threshold
    assert not box_passes("smoke", 0.40, 0.40, det)    # large hazy box, weak evidence
    assert box_passes("smoke", 0.70, 0.40, det)        # large box, strong evidence
    assert box_passes("fire", 0.40, 0.40, det)         # rule disabled for fire (0.0)
    assert not box_passes("smoke", 0.20, 0.05, det)    # below base threshold
    det["large_box"]["area_frac"] = 0.0                # 0 disables the rule
    assert box_passes("smoke", 0.40, 0.90, det)
