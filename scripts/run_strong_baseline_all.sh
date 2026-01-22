#!/bin/bash
# Strong single-task baseline - BOTH padded and native
# Run on GPU with: bash scripts/run_strong_baseline_all.sh
#
# Runs PADDED first (more important for continual learning comparison)

set -e

echo "=============================================="
echo "Strong Baseline - ALL (Padded + Native)"
echo "=============================================="
echo ""
echo "This will run:"
echo "  1. PADDED environments (for continual learning comparison)"
echo "  2. NATIVE environments (pure single-task baseline)"
echo ""
echo "Total: 10M steps x 5 games x 5 seeds x 2 modes = 500M steps"
echo ""

# Activate environment if needed
# source venv/bin/activate

# Set JAX to use GPU
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ============================================
# PADDED (more important - continual learning)
# ============================================
echo ""
echo "=============================================="
echo "Part 1/2: PADDED environments"
echo "=============================================="

python scripts/bench_single_strong.py \
    --timesteps 10000000 \
    --num-seeds 5 \
    --num-envs 2048 \
    --eval-freq 100000 \
    --padded \
    --use-wandb \
    --wandb-project rejax-minatar \
    --plot

# ============================================
# NATIVE (pure single-task)
# ============================================
echo ""
echo "=============================================="
echo "Part 2/2: NATIVE environments"
echo "=============================================="

python scripts/bench_single_strong.py \
    --timesteps 10000000 \
    --num-seeds 5 \
    --num-envs 2048 \
    --eval-freq 100000 \
    --use-wandb \
    --wandb-project rejax-minatar \
    --plot

echo ""
echo "=============================================="
echo "All done!"
echo "=============================================="
echo "Results:"
echo "  - Padded: results/strong_baseline_padded/"
echo "  - Native: results/strong_baseline_native/"
