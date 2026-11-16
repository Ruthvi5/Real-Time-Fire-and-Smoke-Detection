#!/usr/bin/env bash
# Regenerate every evaluation artifact used in the report, from the trained weights.
#
# Usage (from the project folder, inside the conda environment):
#     bash tools/reproduce_results.sh mps      # or cpu, or 0 for the first CUDA GPU
#
# Needs: dataset/ (built by tools/prepare_dataset.py) and the D-Fire surveillance videos in
# clips/dfire_videos/Videos with video_valid.txt / video_test.txt and the annotation CSVs.
# Writes:
#   docs/image_eval/<model>_<split>/   metrics.json + PR/F1 curves, confusion matrices, example predictions
#   results/cache_<model>/             low-threshold detections on all 100 videos (input to the tables)
#   docs/video_eval/<model>_<split>.md frame-wise vs temporal vs final config tables
set -euo pipefail
DEVICE="${1:-cpu}"
cd "$(dirname "$0")/.."

for M in y11n y11s; do
  for SPLIT in val test; do
    echo "== image-level evaluation: ${M} on ${SPLIT}"
    python tools/evaluate_images.py --model "models/${M}_dfire.pt" --split "$SPLIT" \
      --device "$DEVICE" --out "docs/image_eval/${M}_${SPLIT}"
  done

  echo "== caching detections on the D-Fire videos: ${M}"
  python infer.py --source clips/dfire_videos/Videos --config tools/dev_lowconf.yaml \
    --output-dir "results/cache_${M}" --model "models/${M}_dfire.pt" --device "$DEVICE" --no-display --quiet

  for SPLIT in valid test; do
    echo "== video-level tables: ${M} on ${SPLIT}"
    python tools/video_ablation.py --predictions "results/cache_${M}/predictions.jsonl" \
      --annotations "clips/dfire_videos/annotations_${SPLIT}.csv" --out "docs/video_eval/${M}_${SPLIT}.md"
  done
done
echo "done: see docs/image_eval and docs/video_eval"
