"""Alarm generation for one video: per-label trackers, event IDs and alarm records."""

from __future__ import annotations

import re

from .config import LABELS
from .temporal import ClassTracker


def _safe_id(source_id: str) -> str:
    stem = source_id.rsplit(".", 1)[0] if "." in source_id.rsplit("/", 1)[-1] else source_id
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "video"


class AlarmManager:
    """Create a new one for every video, so temporal state never leaks between videos."""

    def __init__(self, cfg: dict, source_id: str):
        self.source_id = source_id
        self.prefix = _safe_id(source_id)
        cooldown = float(cfg["alarm"]["cooldown_seconds"])
        self.trackers = {
            label: ClassTracker(cooldown_seconds=cooldown, **cfg["temporal"][label])
            for label in LABELS
        }
        self.counts = {label: 0 for label in LABELS}

    def update(self, frame_index: int, timestamp_s: float, votes: dict[str, float | None]) -> list[dict]:
        """votes maps label -> best confidence among boxes that count as evidence, or None.

        Returns the alarm records raised at this frame (usually empty).
        """
        alarms = []
        for label, tracker in self.trackers.items():
            conf = votes.get(label)
            result = tracker.update(timestamp_s, frame_index, conf is not None, conf or 0.0)
            if result is None:
                continue
            self.counts[label] += 1
            alarms.append({
                "source_id": self.source_id,
                "event_id": f"{self.prefix}_{label}_{self.counts[label]:03d}",
                "label": label,
                "decision_timestamp_s": round(result.decision_timestamp_s, 3),
                "confidence": round(result.confidence, 4),
                # Extra fields: the onset estimate is kept separate from the decision time,
                # as the submission instructions require.
                "estimated_onset_s": round(result.estimated_onset_s, 3),
                "decision_frame_index": result.decision_frame_index,
                "positive_frames": result.positive_frames,
                "window_frames": result.window_frames,
            })
        return alarms

    def states(self) -> dict[str, str]:
        return {label: t.state for label, t in self.trackers.items()}
