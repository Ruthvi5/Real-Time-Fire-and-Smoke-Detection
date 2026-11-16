"""Turn raw YOLO output into prediction records, and the optional motion check."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np


def box_passes(label: str, confidence: float, area_frac: float, det_cfg: dict) -> bool:
    """Decide whether one box is kept. Shared by inference and by tools/evaluate_videos.py.

    Rules: per-class confidence threshold, optional minimum box size, and an optional
    stricter threshold for very large boxes. Large hazy regions (fog, low cloud, steam)
    are the most common smoke false alarm, while real large smoke is usually detected
    with high confidence, so large boxes must clear a higher bar.
    """
    if confidence < det_cfg["confidence"].get(label, 1.01):
        return False
    min_area = float(det_cfg.get("min_box_area_frac", 0.0) or 0.0)
    if min_area > 0 and area_frac < min_area:
        return False
    large = det_cfg.get("large_box") or {}
    large_area = float(large.get("area_frac", 0.0) or 0.0)
    if large_area > 0 and area_frac >= large_area:
        if confidence < float((large.get("min_confidence") or {}).get(label, 0.0)):
            return False
    return True


def boxes_to_predictions(result, id_to_label: dict, det_cfg: dict, image_shape) -> list[dict]:
    """Convert YOLO boxes to prediction records: clip to the frame, then apply box_passes."""
    h, w = image_shape[:2]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    preds = []
    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
        label = id_to_label.get(int(cls))
        if label is None:
            continue
        x1, x2 = float(np.clip(x1, 0, w)), float(np.clip(x2, 0, w))
        y1, y2 = float(np.clip(y1, 0, h)), float(np.clip(y2, 0, h))
        if x2 <= x1 or y2 <= y1:
            continue
        area_frac = (x2 - x1) * (y2 - y1) / float(w * h)
        if not box_passes(label, float(conf), area_frac, det_cfg):
            continue
        preds.append({
            "label": label,
            "confidence": round(float(conf), 4),
            "bbox_xyxy": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "area_frac": round(area_frac, 5),
        })
    preds.sort(key=lambda p: -p["confidence"])
    return preds


class MotionChecker:
    """Measures how much the image changed inside each box compared with a past frame.

    Real flames flicker and real smoke drifts; a lamp, a painting of fire or a
    sunset barely changes. Only current and past frames are used.
    """

    def __init__(self, cfg: dict):
        mcfg = cfg["motion"]
        self.enabled = bool(mcfg.get("enabled", False))
        self.compare_seconds = float(mcfg.get("compare_seconds", 1.0))
        self.width = int(mcfg.get("analysis_width", 320))
        self.min_diff = {k: float(v) for k, v in mcfg.get("min_mean_abs_diff", {}).items()}
        self.history = deque()  # (timestamp_s, small gray image, scale)

    def _small_gray(self, image):
        h, w = image.shape[:2]
        scale = self.width / float(w) if w > self.width else 1.0
        small = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA) if scale != 1.0 else image
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        return cv2.GaussianBlur(gray, (5, 5), 0), scale

    def score(self, image, timestamp_s: float, predictions: list[dict]) -> None:
        """Add 'motion_score' (mean absolute grey-level change, 0-255) to each prediction in place.

        Score is None when there is no past frame old enough to compare with.
        """
        gray, scale = self._small_gray(image)
        reference = None
        for ts, g, _ in self.history:  # oldest first
            if timestamp_s - ts <= self.compare_seconds + 1e-6:
                reference = g
                break
        for p in predictions:
            if reference is None or reference.shape != gray.shape:
                p["motion_score"] = None
                continue
            x1, y1, x2, y2 = [int(round(v * scale)) for v in p["bbox_xyxy"]]
            x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
            a = gray[y1:y2, x1:x2].astype(np.int16)
            b = reference[y1:y2, x1:x2].astype(np.int16)
            p["motion_score"] = round(float(np.abs(a - b).mean()), 3) if a.size else None

        self.history.append((timestamp_s, gray, scale))
        while self.history and timestamp_s - self.history[0][0] > self.compare_seconds + 1e-6:
            self.history.popleft()

    def passes(self, prediction: dict) -> bool:
        """Should this detection count as a vote for the temporal logic?"""
        if not self.enabled:
            return True
        s = prediction.get("motion_score")
        if s is None:  # nothing to compare with yet: do not veto
            return True
        return s >= self.min_diff.get(prediction["label"], 0.0)
