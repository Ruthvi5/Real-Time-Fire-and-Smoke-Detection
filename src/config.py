"""Load and validate config.yaml."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

LABELS = ("fire", "smoke")

DEFAULTS = {
    "model": {
        "path": "models/best.pt",
        "input_size": 640,
        "iou_threshold": 0.5,
        "max_detections": 50,
        "class_names": {"fire": "fire", "smoke": "smoke"},
    },
    "runtime": {"device": "auto", "half": False},
    "video": {"sample_fps": 5, "fallback_fps": 25},
    "detection": {
        "confidence": {"fire": 0.35, "smoke": 0.30},
        "min_box_area_frac": 0.0,
        "large_box": {"area_frac": 0.0, "min_confidence": {"fire": 0.0, "smoke": 0.0}},
    },
    "motion": {
        "enabled": False,
        "compare_seconds": 1.0,
        "analysis_width": 320,
        "min_mean_abs_diff": {"fire": 4.0, "smoke": 2.0},
    },
    "temporal": {
        "fire": {"window_seconds": 2.0, "min_positive_ratio": 0.6,
                 "min_positive_frames": 4, "clear_after_seconds": 3.0},
        "smoke": {"window_seconds": 3.0, "min_positive_ratio": 0.6,
                  "min_positive_frames": 6, "clear_after_seconds": 4.0},
    },
    "alarm": {"cooldown_seconds": 5.0},
    "output": {"summary_image": True, "summary_frames": 6, "summary_note": ""},
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml must contain a mapping at the top level")

    cfg = _merge(DEFAULTS, raw)
    cfg["_config_dir"] = str(path.parent)

    # Resolve the model path relative to the config file, not the working directory.
    model_path = Path(cfg["model"]["path"]).expanduser()
    if not model_path.is_absolute():
        model_path = path.parent / model_path
    cfg["model"]["path"] = str(model_path)

    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    def positive(name, value, allow_zero=False):
        ok = value >= 0 if allow_zero else value > 0
        if not isinstance(value, (int, float)) or not ok:
            raise ConfigError(f"{name} must be a {'non-negative' if allow_zero else 'positive'} number, got {value!r}")

    positive("model.input_size", cfg["model"]["input_size"])
    positive("video.sample_fps", cfg["video"]["sample_fps"], allow_zero=True)
    positive("video.fallback_fps", cfg["video"]["fallback_fps"])
    positive("alarm.cooldown_seconds", cfg["alarm"]["cooldown_seconds"], allow_zero=True)
    large = cfg["detection"].get("large_box") or {}
    if not 0 <= float(large.get("area_frac", 0) or 0) <= 1:
        raise ConfigError("detection.large_box.area_frac must be in [0, 1] (0 disables)")
    for label, v in (large.get("min_confidence") or {}).items():
        if not isinstance(v, (int, float)) or not 0 <= v <= 1:
            raise ConfigError(f"detection.large_box.min_confidence.{label} must be in [0, 1]")
    for label in LABELS:
        conf = cfg["detection"]["confidence"].get(label)
        if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            raise ConfigError(f"detection.confidence.{label} must be in [0, 1]")
        t = cfg["temporal"].get(label)
        if not isinstance(t, dict):
            raise ConfigError(f"temporal.{label} block is missing")
        positive(f"temporal.{label}.window_seconds", t["window_seconds"])
        positive(f"temporal.{label}.min_positive_frames", t["min_positive_frames"])
        positive(f"temporal.{label}.clear_after_seconds", t["clear_after_seconds"])
        if not 0 < t["min_positive_ratio"] <= 1:
            raise ConfigError(f"temporal.{label}.min_positive_ratio must be in (0, 1]")
        if label not in cfg["model"]["class_names"]:
            raise ConfigError(f"model.class_names.{label} is missing")
