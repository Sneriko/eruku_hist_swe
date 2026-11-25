#!/usr/bin/env bash
# Run from any directory. Optional argument: all (default), warmup, or joint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
phase=${1:-all}
case "$phase" in all|warmup|joint) ;; *) echo 'Usage: bash scripts/train_diffusion_experiment.sh [all|warmup|joint]' >&2; exit 2 ;; esac
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
experiment_python=${EXPERIMENT_PYTHON:-.venv/bin/python}
experiment_base=${EXPERIMENT_BASE:-output/eruku-century-prose-128-vae-replay50-noise25-std05-from3250-gpu0/stage-1-step-3750}
experiment_output=${EXPERIMENT_OUTPUT:-output/eruku-century-prose-128-diffusion-pilot}
common=(
  --manifest /mnt/scratch/emuru_eruku_data/production-century-prose-128/split.jsonl
  --replay-train-pattern '/mnt/scratch/emuru_eruku_data/data/font-square-pairs-ft/*.tar'
  --historical-probability 0.5
  --vae output/vae-production-century-balanced/final
  --t5-config google-t5/t5-large
  --height 128 --max-width 4096 --sample-max-width 4096
  --curriculum eruku --short-steps 0
  --batch-size "${BATCH_SIZE:-16}" --accumulation "${ACCUMULATION:-64}" --workers 4
  --precision bf16 --text-dropout 0.05
  --target-noise-probability 0.25 --target-noise-std-ratio 0.05
  --latent-head diffusion --diffusion-width 256 --diffusion-depth 3
  --diffusion-steps 20 --diffusion-latent-scale 1.0
  --save-every 250 --eval-batches 20 --log-every 10
  --sample-split val --sample-count 50 --sample-fixed-count 30 --sample-cfg-scale 1.375
  --seed 24
)
if [[ "$phase" == all || "$phase" == warmup ]]; then
  "$experiment_python" -m historical_swedish.train "${common[@]}" \
    --init-checkpoint "$experiment_base" \
    --output-dir "$experiment_output/warmup" \
    --diffusion-head-only --lr 1e-3 --long-steps 250
fi
if [[ "$phase" == all || "$phase" == joint ]]; then
  "$experiment_python" -m historical_swedish.train "${common[@]}" \
    --init-checkpoint "$experiment_output/warmup/model.pth" \
    --output-dir "$experiment_output/joint" \
    --lr 2e-5 --long-steps 5000
fi
