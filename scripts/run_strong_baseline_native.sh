#!/bin/bash
# Strong single-task baseline with NATIVE environments (pure single-task)
# Run on GPU with: bash scripts/run_strong_baseline_native.sh

set -e

echo "=============================================="
echo "Strong Baseline - NATIVE (Pure Single-Task)"
echo "=============================================="

# Activate environment if needed
# source venv/bin/activate

# Set JAX to use GPU
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run benchmark
python scripts/bench_single_strong.py \
    --timesteps 10000000 \
    --num-seeds 5 \
    --num-envs 2048 \
    --eval-freq 100000 \
    --use-wandb \
    --wandb-project rejax-minatar \
    --plot

echo "Done! Results saved to results/strong_baseline_native/"
