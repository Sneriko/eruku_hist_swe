#!/usr/bin/env bash
# Fresh pretrained long-stage adaptation: raw line crops, 50/50 data, 3000 updates.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run_dir="${ERUKU_OUTPUT_DIR:-output/historical-swedish-replay-64-raw-hist50}"
if [[ -e "$run_dir" ]]; then
  echo "Output already exists: $run_dir. Set ERUKU_OUTPUT_DIR to a new path for a fresh run." >&2
  exit 1
fi
mkdir -p "$run_dir/source/historical_swedish"
cp eruku_continuous_inf.py "$run_dir/source/"
cp historical_swedish/*.py "$run_dir/source/historical_swedish/"
CUDA_VISIBLE_DEVICES="${ERUKU_GPU:-0}" CUDA_LAUNCH_BLOCKING=0 PYTHONUNBUFFERED=1 \
.venv/bin/accelerate launch --num_processes 1 -m historical_swedish.train \
  --manifest /mnt/scratch/emuru_eruku_data/swedish_historical_eruku/split.jsonl \
  --output-dir "$run_dir" \
  --init-checkpoint blowing-up-groundhogs/eruku \
  --replay-train-pattern '/mnt/scratch/emuru_eruku_data/data/font-square-pairs-ft/*.tar' \
  --historical-probability 0.8 --use normalized \
  --short-steps 0 --long-steps 5000 \
  --height 64 --max-width 2048 \
  --batch-size 8 --accumulation 32 --workers 4 \
  --precision bf16 --lr 2e-5 \
  --log-every 1 --save-every 50 --eval-batches 20 \
  --sample-validation-pairs --sample-count 20 --sample-fixed-count 10 \
  --sample-max-width 1024 --sample-cfg-scale 1.25 --seed 24 \
  "$@" \
  2>&1 | tee "$run_dir/training.log"
