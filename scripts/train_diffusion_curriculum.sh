#!/usr/bin/env bash
# Start a new curriculum run, or pass --resume CHECKPOINT with identical settings.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
curriculum_python=${CURRICULUM_PYTHON:-.venv/bin/python}
curriculum_ocr=${CURRICULUM_OCR-'output/eruku-century-prose-128-diffusion-longwarmup/joint-hist80-replay20-steps20-from4500/stage-1-step-9500/testsuite-page-cfg-1375-1500-1750/ocr-model'}
curriculum_ocr_args=()
if [[ -n "$curriculum_ocr" ]]; then
  curriculum_ocr_args=(--sample-ocr "$curriculum_ocr")
fi

exec "$curriculum_python" -m historical_swedish.train \
  --manifest /mnt/scratch/emuru_eruku_data/production-century-prose-128/split.jsonl \
  --replay-train-pattern '/mnt/scratch/emuru_eruku_data/data/font-square-pairs-ft/*.tar' \
  --historical-probability 0.8 \
  --output-dir "${CURRICULUM_OUTPUT:-output/eruku-prose-128-diffusion-curriculum}" \
  --vae "${CURRICULUM_VAE:-output/vae-production-century-balanced/final}" \
  --t5-config google-t5/t5-large --height 128 --max-width 4096 \
  --curriculum three-stage --short-steps "${SHORT_STEPS:-1000}" \
  --medium-steps "${MEDIUM_STEPS:-2000}" --long-steps "${LONG_STEPS:-7000}" \
  --batch-size "${BATCH_SIZE:-4}" --accumulation "${ACCUMULATION:-32}" --workers 4 \
  --precision bf16 --lr 1e-4 --text-dropout 0.05 \
  --target-noise-probability "${TARGET_NOISE_PROBABILITY:-0.5}" --target-noise-std-ratio 0.05 \
  --latent-head diffusion --diffusion-width 256 --diffusion-depth 3 \
  --diffusion-steps 20 --diffusion-latent-scale 1.0 \
  --save-every "${SAVE_EVERY:-500}" --eval-batches 20 --log-every 10 \
  --sample-split val --sample-count 12 --sample-max-width 4096 \
  --sample-cfg-scales 1.0 1.375 --sample-seeds 24 25 --sample-teacher-forced \
  --sample-alt-vae blowing-up-groundhogs/emuru_vae "${curriculum_ocr_args[@]}" \
  --seed 24 "$@"
