# Real-Time Fire and Smoke Detection in Surveillance Videos

**Selected project:** Smoke/fire alarm detection (fire/smoke detection)

| Roll number | Name |
|---|---|
| 142301021 | M. Ruthvi |
| 142301034 | Eswar Sai Rama Krishna |
| 142201001 | R. Durga Sri |

A fine-tuned YOLO11 detector finds fire and smoke boxes in each sampled frame. A causal
temporal module, with a separate state machine for each class, then decides when the
evidence is strong enough to raise an alarm. Videos are processed strictly in order,
using only the current and earlier frames, and the temporal state is reset for every video.

## 1. Requirements

| Item | Tested |
|---|---|
| Python | 3.12 (conda environment on macOS); 3.13 on Linux |
| OS | macOS 26.6.2 on Apple Silicon (MacBook Air, `--device mps` and `cpu`); Ubuntu 24.04 x86_64 (`--device cpu`) |
| CUDA | Optional. Training and image-level evaluation ran on CUDA (Google Colab, Tesla T4, torch 2.14.1+cu130). CPU-only inference is fully supported |
| Hardware | Any 64-bit CPU with 4 GB RAM is enough. Measured end-to-end speed (decoding, YOLO11n, temporal logic, writing outputs): about 55 processed frames/s on the MacBook Air with `mps`, about 17 processed frames/s on a 2-core cloud CPU. With `video.sample_fps: 5` both are several times faster than real time |

No internet connection, GUI or interactive input is needed at inference time. All weights are in `models/`.

## 2. Installation

With conda (as tested):

```bash
conda create -n firecv python=3.12 -y
conda activate firecv
pip install --upgrade pip
pip install -r requirements.txt
```

Or with a plain virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Inference

Run inference only with the command given in the course instructions, changing only the
input path, the output folder and the device:

```bash
python infer.py --source INPUT.mp4 --config config.yaml --output-dir results --device cuda:0 --no-display
```

`--source` takes a video file (a folder of videos also works; it is searched recursively for
`.mp4 .avi .mkv .mov .m4v .mpg .mpeg .wmv .flv .webm .ts .3gp .dav .h264`).
`--no-display` is required by the instructions; inference never opens a window in any case.

**Supported `--device` values:** `auto` (default: cuda:0, then mps, then cpu), `cpu`,
`cuda`, `cuda:N`, `N` (GPU index) and `mps` (Apple Silicon). If the requested device is not
available, the run falls back to `cpu` and records a warning in `run_summary.json` instead of failing.

Exit code is 0 when processing finished (even if some individual videos were unreadable;
these are listed in `errors`), and 1 when setup failed, for example missing weights.

## 4. Outputs

The three required files are written to `--output-dir`, plus one summary picture per video in
`--output-dir/images/` (see the end of this section).

**predictions.jsonl**: one line per processed frame.
```json
{"source_id":"event.mp4","frame_index":105,"timestamp_s":4.2,"predictions":[{"label":"fire","confidence":0.9684,"bbox_xyxy":[121,81,199,159],"area_frac":0.0066,"motion_score":7.81}]}
```
`predictions` is `[]` when nothing is detected. `bbox_xyxy` is in pixels of the original frame.
`area_frac` is the box area as a fraction of the frame. `motion_score` is the mean grey-level change inside the box over
the last second (null when there is no earlier frame). Only `label` and `confidence` are required by the course; the
other fields are extra.

**alarms.jsonl**: one line per newly raised alarm, written as soon as it is raised.
```json
{"source_id":"event.mp4","event_id":"event_fire_001","label":"fire","decision_timestamp_s":4.2,"confidence":0.9684,"estimated_onset_s":3.0,"decision_frame_index":105,"positive_frames":7,"window_frames":11}
```
`decision_timestamp_s` is the timestamp of the latest frame used to raise the alarm.
The estimated earlier onset (first positive frame in the confirming window) is recorded
separately in `estimated_onset_s`. `confidence` is the mean detector confidence over the
positive frames in that window.

**run_summary.json**: `status` (`completed`, `completed_with_errors` or `failed`),
`frames_read`, `frames_processed`, `frames_skipped`, `total_runtime_s` and `errors`, plus
warnings, device and per-video details. `frames_read` counts every decoded frame;
`frames_skipped` counts frames that were read but not sent to the detector because of frame sampling.

**images/<video name>_summary.jpg**: one picture per video, written by the same command. It
shows the video name, the list of alarms (time, estimated onset, confidence) and six captioned
frames with the detection boxes from `predictions.jsonl`: an early frame, the estimated onset frame
and the frame that raised each alarm, and frames spread over the rest of the video. The frames are
kept during the same single pass over the video (a bounded set of small copies, about 80 MB at most
even for hour-long videos), so no second read is needed; this adds about 5% to the runtime. Making
the picture never changes detections or alarms, and if it fails the run still completes and a
warning is recorded. It can be switched off with `output.summary_image: false` in `config.yaml`.
The path of each picture is listed as `summary_image` in `run_summary.json`.

More pictures (optional): `tools/visualize_results.py` re-makes the summary from an existing output
folder and, with `--details`, adds the separate alarm and onset frames, an overview grid and a
confidence timeline; `--video` adds an annotated video.

```bash
python tools/visualize_results.py --results results --source INPUT.mp4 --config config.yaml --details --video
```

## 5. Method summary

1. **Frame sampling** (`src/preprocessing.py`): frames are read sequentially and sampled
   by timestamp at `video.sample_fps`, so a 10 fps and a 30 fps camera behave the same.
   Container timestamps are used when valid, otherwise `frame_index / fps`.
2. **Detection** (`src/detector.py`, `src/postprocessing.py`): YOLO11n fine-tuned on D-Fire
   (`models/best.pt` is `models/y11n_dfire.pt`), per-class confidence thresholds (fire 0.35,
   smoke 0.25). Classes are mapped by name, so the class-ID order of the training data does not matter.
3. **Large-box rule**: a box covering at least 25% of the frame counts as smoke only with
   confidence 0.65 or more. Fog, low cloud and steam produce large, weak smoke boxes, while real
   large smoke is usually detected with high confidence. Not applied to fire.
   An optional motion check (`motion.enabled`) was also implemented and evaluated, but gave no
   benefit on validation videos, so it is disabled.
4. **Temporal verification** (`src/temporal.py`): per class, an alarm is raised when, within
   the last `window_seconds`, at least `min_positive_frames` sampled frames and at least
   `min_positive_ratio` of them contain evidence. States: CLEAR, CANDIDATE, ACTIVE, CLEARING.
   One alarm per event; the event ends after `clear_after_seconds` without evidence, and a
   new alarm of the same class needs `cooldown_seconds` after that.
5. **Alarm records** (`src/alarms.py`) and output files (`src/pipeline.py`).

All settings live in the single `config.yaml`, which stays fixed during testing.

## 6. Verification example

Two short clips recorded by the team (shareable, no third-party content):

| Clip | Content | Expected result |
|---|---|---|
| `examples/event_lamp_fire.mp4` | 16 s, 1280x720, 30 fps. An oil lamp (diya) is lit at about 5 s | exactly one `fire` alarm, at about 6.6 s |
| `examples/normal_mountain_clouds.mp4` | 10.5 s, 1280x720, 30 fps. Hills with low cloud, no fire or smoke | no alarm |

Exact commands (the course command, run once per clip):

```bash
python infer.py --source examples/event_lamp_fire.mp4 --config config.yaml --output-dir results/event --device cuda:0 --no-display
python infer.py --source examples/normal_mountain_clouds.mp4 --config config.yaml --output-dir results/normal --device cuda:0 --no-display
```

Use `--device cpu` or `--device mps` on machines without CUDA (with `cuda:0` and no GPU the run
falls back to CPU and notes this in `run_summary.json`).

Expected outputs are in `expected_outputs/event/` and `expected_outputs/normal/`, produced with
`--device cpu`. `results/event/alarms.jsonl` should contain one record with `"label":"fire"` and
`decision_timestamp_s` close to 6.6 (estimated onset about 5.2 s); `results/normal/alarms.jsonl`
should be empty. Each folder also has the summary picture in `images/`. Frame counts must match exactly (event: 480 read, 80 processed, 400 skipped;
normal: 316 read, 53 processed, 263 skipped). Confidences can differ in the third decimal between
CPU, CUDA and MPS, and runtimes depend on the machine.

### Test on new videos with the course command

We also tested a fresh copy of the project, in a new virtual environment, on two fire videos
the system had never seen. They are third-party clips, so they are not included in the
submission. Both were put in a folder `test_videos/` and run with only the course command:

```bash
python infer.py --source test_videos --config config.yaml --output-dir results/real_all --device cpu --no-display
```

This single command produced all outputs:

```
results/real_all/
  predictions.jsonl
  alarms.jsonl
  run_summary.json
  images/car_fire_summary.jpg
  images/home_fire_summary.jpg
```

`run_summary.json` reported `"status": "completed"` with no errors (1,024 frames read, 187 processed,
17 s in total on a MacBook Air CPU).

| Video | Content | Alarms raised (decision time, estimated onset) |
|---|---|---|
| `car_fire.mp4` (15 s, 720x1280, 24 fps) | street CCTV, a car on a highway starts smoking and catches fire | smoke at 5.8 s (onset 4.6 s), fire at 6.6 s (onset 5.4 s) |
| `home_fire.mp4` (22 s, 1080x1920, 30 fps) | indoor home camera, smoke on a carpet, then a flash fire | smoke at 2.8 s (onset 1.2 s), fire at 5.0 s (onset 3.6 s) |

Both events were caught, smoke before fire, each alarm 1.2 to 1.6 s after its estimated onset.
In the summary pictures the boxes sit on the burning car and its smoke, and on the smoke and
flames on the carpet, with no boxes elsewhere in the scene. Running the
same command on Linux gave the same four alarms within 0.2 s (one sampled frame).

## 7. Training (reproducibility)

```bash
# 1. Build the dataset from D-Fire (keeps the official test split; grouped val split)
python tools/prepare_dataset.py --src /path/to/D-Fire --out dataset --names smoke,fire
#    then open dataset/previews/ and confirm the class names are on the right boxes

# 2. Optional: hard negatives from normal surveillance videos (whole videos per split)
python tools/add_negatives.py --videos clips/normal_train --out dataset --split train --every 2

# 3. Fine-tune (exact commands used, Google Colab Tesla T4)
python train.py --data dataset/data.yaml --model yolo11n.pt --epochs 40 --batch 32 --workers 2 --device 0 --name y11n_dfire
python train.py --data dataset/data.yaml --model yolo11s.pt --epochs 40 --batch 16 --workers 2 --device 0 --name y11s_dfire
cp models/y11n_dfire.pt models/best.pt      # selected on validation videos

# 4. Tune the alarm settings on the official D-Fire validation videos only, then freeze config.yaml
#    (videos in clips/dfire_videos/Videos, split lists video_valid.txt / video_test.txt)
#    (same command as above, with the low-threshold development config so every box is cached;
#     tools/dev_lowconf.yaml points to models/best.pt, which is YOLO11n)
python infer.py --source clips/dfire_videos/Videos --config tools/dev_lowconf.yaml --output-dir results/cache_y11n --device mps --no-display
python tools/evaluate_videos.py --predictions results/cache_y11n/predictions.jsonl \
    --annotations clips/dfire_videos/annotations_valid.csv --config config.yaml \
    --set detection.confidence.smoke=0.2,0.25,0.3 --set temporal.smoke.window_seconds=2,3,4 \
    --set detection.large_box.area_frac=0,0.25 --csv results/sweep.csv

# 5. Regenerate every number and plot in docs/ (image-level metrics, PR curves, confusion
#    matrices, video-level tables) from the trained weights
bash tools/reproduce_results.sh mps
```

`train.py` adds CCTV-style augmentation (grayscale for night IR footage, blur, JPEG
compression, mild sensor noise) on top of the Ultralytics defaults, and writes a JSON
training record next to each checkpoint. The Colab notebooks that ran these commands, with
their full outputs, are in `notebooks/`. Training curves and settings of both runs are in
`docs/training/`; all results are summarised in `docs/experiments.md`.

## 8. Tests

```bash
python -m pytest tests -q
```
The 11 tests use a stand-in detector, so they need no trained weights. They check the temporal
rules, the output formats, frame sampling at different frame rates, per-video reset,
corrupted files, the motion check, the large-box rule, that the course command line is accepted,
and that the summary pictures are written and a failure while drawing them changes nothing else.

## 9. Project layout

```
infer.py                 headless inference entry point
train.py                 fine-tuning
config.yaml              final fixed configuration (the only config used for testing)
src/                     preprocessing, detector, postprocessing, temporal, alarms, pipeline, visualize
tools/                   dataset preparation, hard negatives, image/video evaluation, sweeps, reproduce_results.sh,
                         visualize_results.py (pictures of infer.py results),
                         dev_lowconf.yaml (low-threshold config used only for tuning, not for testing)
tests/                   automated checks
models/                  trained weights (best.pt)
dataset/                 data.yaml and split manifests (images not included)
examples/, expected_outputs/  verification example (section 6)
docs/                    experiment log, image-level plots, video tables, training curves
notebooks/               the Colab notebooks used to train and test YOLO11n and YOLO11s, with outputs
report.pdf               project report
```

## 10. Licences

D-Fire is released under CC0; its images whose names start with `PublicDataset` come from third-party
sources and remain under their providers' terms. Ultralytics YOLO is AGPL-3.0, which is fine for
academic coursework. The verification clips in `examples/` were recorded by the team.
