"""Causal temporal verification: one state machine per label.

States
  CLEAR      no ongoing event
  CANDIDATE  some detections inside the window, not yet confirmed
  ACTIVE     confirmation rule met; the alarm has been raised exactly once
  CLEARING   detections stopped; waiting clear_after_seconds before returning to CLEAR

Confirmation rule (evaluated on every sampled frame, using only the current
and earlier frames inside the last window_seconds):
  positives >= min_positive_frames  and  positives / frames_in_window >= min_positive_ratio
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

CLEAR, CANDIDATE, ACTIVE, CLEARING = "CLEAR", "CANDIDATE", "ACTIVE", "CLEARING"


@dataclass
class Confirmation:
    decision_timestamp_s: float
    decision_frame_index: int
    estimated_onset_s: float
    confidence: float
    positive_frames: int
    window_frames: int


class ClassTracker:
    def __init__(self, window_seconds: float, min_positive_ratio: float,
                 min_positive_frames: int, clear_after_seconds: float,
                 cooldown_seconds: float = 0.0):
        self.window_seconds = float(window_seconds)
        self.min_ratio = float(min_positive_ratio)
        self.min_frames = int(min_positive_frames)
        self.clear_after = float(clear_after_seconds)
        self.cooldown = float(cooldown_seconds)
        self.reset()

    def reset(self):
        self.state = CLEAR
        self.window = deque()          # (timestamp_s, frame_index, positive, confidence)
        self.last_positive_ts = None
        self.event_ended_ts = None     # when the previous event returned to CLEAR

    def update(self, timestamp_s: float, frame_index: int, positive: bool,
               confidence: float = 0.0) -> Confirmation | None:
        """Feed one sampled frame. Returns a Confirmation only when a new alarm must be raised."""
        self.window.append((timestamp_s, frame_index, bool(positive), float(confidence)))
        cutoff = timestamp_s - self.window_seconds - 1e-6
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()
        if positive:
            self.last_positive_ts = timestamp_s

        n = len(self.window)
        positives = [e for e in self.window if e[2]]
        k = len(positives)
        confirmed = k >= self.min_frames and k / n >= self.min_ratio

        if self.state in (CLEAR, CANDIDATE):
            in_cooldown = (self.event_ended_ts is not None
                           and timestamp_s - self.event_ended_ts < self.cooldown)
            if confirmed and not in_cooldown:
                self.state = ACTIVE
                return Confirmation(
                    decision_timestamp_s=timestamp_s,
                    decision_frame_index=frame_index,
                    estimated_onset_s=positives[0][0],
                    confidence=sum(e[3] for e in positives) / k,
                    positive_frames=k,
                    window_frames=n,
                )
            self.state = CANDIDATE if k > 0 else CLEAR
            return None

        # ACTIVE or CLEARING: the alarm was already raised, just track when the event ends.
        if positive:
            self.state = ACTIVE
        elif self.last_positive_ts is not None and timestamp_s - self.last_positive_ts >= self.clear_after:
            self.state = CLEAR
            self.event_ended_ts = timestamp_s
            self.window.clear()  # a new event must be confirmed from fresh evidence
        else:
            self.state = CLEARING
        return None
