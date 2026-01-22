#!/bin/bash
# Strong single-task baseline with PADDED environments (for continual learning comparison)
# Run on GPU with: bash scripts/run_strong_baseline_padded.sh

set -e

echo "=============================================="
echo "Strong Baseline - PADDED (Continual Learning)"
echo "=============================================="

# Set JAX to use GPU
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run benchmark
uv run python scripts/bench_single_strong.py \
    --timesteps 10000000 \
    --num-seeds 5 \
    --num-envs 2048 \
    --eval-freq 100000 \
    --padded \
    --use-wandb \
    --wandb-project rejax-minatar \
    --plot

echo "Done! Results saved to results/strong_baseline_padded/"
