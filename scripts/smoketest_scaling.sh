#!/bin/bash
# Quick smoke test: 1 env, 1 depth, 100k steps, 2 seeds
# Tests all 7 configs with wandb before full run
set -e

echo "=== Smoke test: all configs, minimal resources ==="

# Ortho optimizer grid
for COEFF in 1e-4 1e-3 1e-2 1e-1; do
    echo "Ortho optimizer coeff=$COEFF"
    uv run python scripts/throughput_benchmark.py \
        --envs Breakout-MinAtar --depths 4 --timesteps 100000 \
        --num-envs 512 --num-seeds 2 --eval-freq 0 \
        --ortho-mode optimizer --ortho-coeff $COEFF \
        --activation groupsort --use-wandb
done

# Baseline
echo "Baseline"
uv run python scripts/throughput_benchmark.py \
    --envs Breakout-MinAtar --depths 4 --timesteps 100000 \
    --num-envs 512 --num-seeds 2 --eval-freq 0 \
    --ortho-mode none --activation tanh --use-wandb

# Ortho loss grid
for LAMBDA in 0.02 0.2; do
    echo "Ortho loss lambda=$LAMBDA"
    uv run python scripts/throughput_benchmark.py \
        --envs Breakout-MinAtar --depths 4 --timesteps 100000 \
        --num-envs 512 --num-seeds 2 --eval-freq 0 \
        --ortho-mode loss --ortho-lambda $LAMBDA \
        --activation groupsort --use-wandb
done

echo "=== Smoke test passed - ready for full run ==="
