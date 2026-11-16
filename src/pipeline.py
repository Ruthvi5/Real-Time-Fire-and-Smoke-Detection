"""End-to-end processing of one or more videos into the three required output files."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .alarms import AlarmManager
from .postprocessing import MotionChecker
from .preprocessing import VideoOpenError, VideoReader, list_videos
from .visualize import SummaryCollector, safe_name

log = logging.getLogger("fire_smoke")


def _dump(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def _summary_collector(cfg: dict, source_id: str, image_path):
    if image_path is None:
        return None
    history = max(float(t["window_seconds"]) for t in cfg["temporal"].values()) + 1.0
    return SummaryCollector(source_id, image_path, history,
                            n_panels=int(cfg["output"].get("summary_frames", 6)),
                            note=str(cfg["output"].get("summary_note", "")))


def process_video(source_id: str, path: Path, cfg: dict, detector, pred_fh, alarm_fh,
                  image_path: Path | None = None) -> dict:
    """Process one video sequentially. All temporal state is created fresh here.

    If image_path is given, a summary picture is also written there at the end. That step only
    reads what has already been decided; if it fails, a warning is recorded and nothing else changes.
    """
    started = time.perf_counter()
    alarms_mgr = AlarmManager(cfg, source_id)
    motion = MotionChecker(cfg)
    reader = VideoReader(path, cfg["video"]["sample_fps"], cfg["video"]["fallback_fps"])
    n_alarms = 0
    image_warnings = []
    try:
        collector = _summary_collector(cfg, source_id, image_path)
    except Exception as exc:
        collector = None
        image_warnings.append(f"summary image disabled: {type(exc).__name__}: {exc}")

    for frame in reader:
        preds = detector(frame.image)
        motion.score(frame.image, frame.timestamp_s, preds)

        # Best confidence per label among boxes that count as evidence.
        votes = {}
        for p in preds:
            if motion.passes(p) and p["confidence"] > (votes.get(p["label"]) or 0.0):
                votes[p["label"]] = p["confidence"]

        pred_fh.write(_dump({
            "source_id": source_id,
            "frame_index": frame.frame_index,
            "timestamp_s": round(frame.timestamp_s, 3),
            "predictions": preds,
        }) + "\n")

        if collector is not None:   # pictures only; never affects the decisions below
            try:
                collector.add(frame.frame_index, frame.timestamp_s, frame.image, preds)
            except Exception as exc:
                collector = None
                image_warnings.append(f"summary image skipped: {type(exc).__name__}: {exc}")

        for alarm in alarms_mgr.update(frame.frame_index, frame.timestamp_s, votes):
            alarm_fh.write(_dump(alarm) + "\n")
            alarm_fh.flush()  # alarms are emitted as they occur
            n_alarms += 1
            log.info("ALARM %s %s at %.2fs (conf %.2f)", source_id, alarm["label"],
                     alarm["decision_timestamp_s"], alarm["confidence"])
            if collector is not None:
                try:
                    collector.alarm(alarm)
                except Exception as exc:
                    collector = None
                    image_warnings.append(f"summary image skipped: {type(exc).__name__}: {exc}")

    pred_fh.flush()
    summary_image = None
    if collector is not None:
        try:
            written = collector.finish()
            if written is not None:
                summary_image = Path(written).relative_to(Path(image_path).parent.parent).as_posix()
        except Exception as exc:
            image_warnings.append(f"summary image not written: {type(exc).__name__}: {exc}")
    for w in image_warnings:
        log.warning("%s: %s", source_id, w)
    st = reader.stats
    st.warnings.extend(image_warnings)
    if st.frames_read == 0:
        st.warnings.append("no frames could be read")
    return {
        "source_id": source_id,
        "frames_read": st.frames_read,
        "frames_processed": st.frames_processed,
        "frames_skipped": st.frames_skipped,
        "fps": round(st.fps, 3),
        "fps_source": st.fps_source,
        "timestamp_source": st.timestamp_source,
        "alarms": n_alarms,
        "summary_image": summary_image,
        "runtime_s": round(time.perf_counter() - started, 3),
        "warnings": st.warnings,
    }


def run(input_path: str, output_dir: str, cfg: dict, detector, device: str = "",
        startup_warnings: list | None = None) -> dict:
    t0 = time.perf_counter()
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "running",
        "frames_read": 0,
        "frames_processed": 0,
        "frames_skipped": 0,
        "total_runtime_s": 0.0,
        "errors": [],
        "warnings": list(startup_warnings or []),
        "videos_found": 0,
        "videos_completed": 0,
        "alarms_raised": 0,
        "device": device,
        "videos": [],
    }

    try:
        videos = list_videos(input_path)
    except FileNotFoundError as exc:
        videos = []
        summary["errors"].append({"source_id": None, "message": str(exc)})
    summary["videos_found"] = len(videos)
    if not videos and not summary["errors"]:
        summary["errors"].append({"source_id": None, "message": f"no video files found in {input_path}"})

    save_images = bool(cfg.get("output", {}).get("summary_image", False))
    used_names: set[str] = set()

    def image_path_for(source_id: str):
        if not save_images:
            return None
        stem = safe_name(Path(source_id).with_suffix("").as_posix())
        name, n = stem, 2
        while name in used_names:
            name, n = f"{stem}_{n}", n + 1
        used_names.add(name)
        return out / "images" / f"{name}_summary.jpg"

    with open(out / "predictions.jsonl", "w", encoding="utf-8") as pred_fh, \
         open(out / "alarms.jsonl", "w", encoding="utf-8") as alarm_fh:
        try:
            for i, (source_id, path) in enumerate(videos, 1):
                log.info("[%d/%d] %s", i, len(videos), source_id)
                try:
                    v = process_video(source_id, path, cfg, detector, pred_fh, alarm_fh,
                                      image_path_for(source_id))
                except VideoOpenError as exc:
                    summary["errors"].append({"source_id": source_id, "message": str(exc)})
                    continue
                except Exception as exc:  # keep going with the next video
                    log.exception("failed on %s", source_id)
                    summary["errors"].append({"source_id": source_id,
                                              "message": f"{type(exc).__name__}: {exc}"})
                    continue
                summary["videos"].append(v)
                summary["videos_completed"] += 1
                summary["alarms_raised"] += v["alarms"]
                for key in ("frames_read", "frames_processed", "frames_skipped"):
                    summary[key] += v[key]
                if v["frames_read"] == 0:
                    summary["errors"].append({"source_id": source_id, "message": "no frames could be read"})
            summary["status"] = "completed" if not summary["errors"] else (
                "completed_with_errors" if summary["videos_completed"] else "failed")
        except KeyboardInterrupt:
            summary["status"] = "interrupted"
            raise
        finally:
            if summary["status"] == "running":
                summary["status"] = "failed"
            summary["total_runtime_s"] = round(time.perf_counter() - t0, 3)
            write_summary(out, summary)
    return summary


def write_summary(out: Path, summary: dict) -> None:
    tmp = Path(out) / "run_summary.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    tmp.replace(Path(out) / "run_summary.json")
