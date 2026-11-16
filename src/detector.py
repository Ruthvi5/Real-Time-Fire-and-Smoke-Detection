"""YOLO detector wrapper: device selection, offline loading and class-name mapping."""

from __future__ import annotations

import os
from pathlib import Path

# Never let Ultralytics try to reach the internet during inference.
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_VERBOSE", "false")

import numpy as np  # noqa: E402

SUPPORTED_DEVICES = "auto, cpu, cuda, cuda:N (e.g. cuda:0), N (GPU index), mps"


def resolve_device(requested: str | None) -> tuple[str, str | None]:
    """Turn a requested device into one that actually exists here.

    Returns (device, warning). If the request cannot be satisfied we fall back
    to CPU and return a warning instead of crashing, so evaluation always finishes.
    """
    import torch

    req = str(requested or "auto").strip().lower()
    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if req == "auto":
        if cuda_ok:
            return "cuda:0", None
        if mps_ok:
            return "mps", None
        return "cpu", None
    if req == "cpu":
        return "cpu", None
    if req == "mps":
        return ("mps", None) if mps_ok else ("cpu", "device 'mps' requested but not available; using cpu")
    if req.isdigit():
        req = f"cuda:{req}"
    if req == "cuda":
        req = "cuda:0"
    if req.startswith("cuda:") and req[5:].isdigit():
        idx = int(req[5:])
        if cuda_ok and idx < torch.cuda.device_count():
            return req, None
        return "cpu", f"device '{requested}' requested but not available; using cpu"
    return "cpu", f"unknown device '{requested}' (supported: {SUPPORTED_DEVICES}); using cpu"


class YoloDetector:
    """Runs the fine-tuned YOLO model on one frame and returns fire/smoke boxes."""

    def __init__(self, cfg: dict, device: str):
        from ultralytics import YOLO

        mcfg = cfg["model"]
        model_path = Path(mcfg["path"])
        # Check ourselves: given a missing file, Ultralytics would try to download it.
        if not model_path.is_file():
            raise FileNotFoundError(f"model weights not found: {model_path}")

        self.model = YOLO(str(model_path), task="detect")
        self.device = device
        self.imgsz = int(mcfg["input_size"])
        self.iou = float(mcfg["iou_threshold"])
        self.max_det = int(mcfg["max_detections"])
        self.half = bool(cfg["runtime"].get("half", False)) and device.startswith("cuda")
        self.det_cfg = cfg["detection"]
        self.conf_by_label = {k: float(v) for k, v in cfg["detection"]["confidence"].items()}

        # Map model class IDs to our output labels by name.
        wanted = {str(v).strip().lower(): label for label, v in mcfg["class_names"].items()}
        names = self.model.names
        names = names if isinstance(names, dict) else dict(enumerate(names))
        self.id_to_label = {int(i): wanted[str(n).strip().lower()]
                            for i, n in names.items() if str(n).strip().lower() in wanted}
        self.warnings = []
        missing = set(mcfg["class_names"]) - set(self.id_to_label.values())
        if missing:
            self.warnings.append(
                f"model classes {sorted(names.values())[:10]} do not include {sorted(missing)}; "
                "those labels will never be predicted")
        self.class_ids = sorted(self.id_to_label) or None
        self.min_conf = min(self.conf_by_label.values()) if self.conf_by_label else 0.25

    def __call__(self, image: np.ndarray) -> list[dict]:
        if not self.id_to_label:
            return []
        results = self.model.predict(
            image, imgsz=self.imgsz, conf=self.min_conf, iou=self.iou, device=self.device,
            half=self.half, max_det=self.max_det, classes=self.class_ids, verbose=False)
        from .postprocessing import boxes_to_predictions
        return boxes_to_predictions(results[0], self.id_to_label, self.det_cfg, image.shape)
