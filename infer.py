#!/usr/bin/env python3
"""Headless fire and smoke detection on video files.

Required course command:
    python infer.py --source INPUT.mp4 --config config.yaml --output-dir results --device cuda:0 --no-display

--source may be a single video or a folder (searched recursively). Writes
predictions.jsonl, alarms.jsonl and run_summary.json into --output_dir.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("MPLBACKEND", "Agg")  # never open a GUI

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import ConfigError, load_config  # noqa: E402
from src.detector import SUPPORTED_DEVICES  # noqa: E402
from src.pipeline import run, write_summary  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fire and smoke detection on surveillance videos (offline, headless).")
    p.add_argument("--source", "--input", "--video", "-i", dest="input", required=True,
                   help="video file or folder of videos")
    p.add_argument("--output-dir", "--output_dir", "--output", "--out", "-o", dest="output_dir", required=True,
                   help="folder for predictions.jsonl, alarms.jsonl and run_summary.json")
    p.add_argument("--no-display", "--no_display", dest="no_display", action="store_true",
                   help="accepted for compatibility; inference never opens a window")
    p.add_argument("--config", "-c", default=str(ROOT / "config.yaml"), help="path to config.yaml")
    p.add_argument("--device", "-d", default=None,
                   help=f"one of: {SUPPORTED_DEVICES}. Default: value in config (auto)")
    p.add_argument("--model", default=None, help="optional override of model.path (development only)")
    p.add_argument("--quiet", action="store_true", help="only print warnings and errors")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    log = logging.getLogger("fire_smoke")
    t0 = time.perf_counter()
    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    try:
        cfg = load_config(args.config)
        if args.model:
            cfg["model"]["path"] = str(Path(args.model).expanduser().resolve())

        from src.detector import YoloDetector, resolve_device
        device, dev_warning = resolve_device(args.device or cfg["runtime"]["device"])
        warnings = [dev_warning] if dev_warning else []
        if dev_warning:
            log.warning(dev_warning)
        detector = YoloDetector(cfg, device)
        for w in detector.warnings:
            log.warning(w)
        warnings += detector.warnings
        log.info("model %s on %s", cfg["model"]["path"], device)
    except (ConfigError, FileNotFoundError, ImportError, RuntimeError) as exc:
        # Fatal setup problem: still leave valid (empty) outputs and a summary explaining why.
        log.error("%s", exc)
        for name in ("predictions.jsonl", "alarms.jsonl"):
            (out / name).touch()
        write_summary(out, {
            "status": "failed", "frames_read": 0, "frames_processed": 0, "frames_skipped": 0,
            "total_runtime_s": round(time.perf_counter() - t0, 3),
            "errors": [{"source_id": None, "message": f"{type(exc).__name__}: {exc}"}],
        })
        return 1

    summary = run(args.input, str(out), cfg, detector, device=device, startup_warnings=warnings)
    log.info("status=%s videos=%d/%d frames read=%d processed=%d alarms=%d time=%.1fs",
             summary["status"], summary["videos_completed"], summary["videos_found"],
             summary["frames_read"], summary["frames_processed"], summary["alarms_raised"],
             summary["total_runtime_s"])
    return 0 if summary["status"] in ("completed", "completed_with_errors") else 1


if __name__ == "__main__":
    sys.exit(main())
