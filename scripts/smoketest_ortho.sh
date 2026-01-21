#!/bin/bash
# Quick smoke test before full scaling run
set -e

echo "=== Smoke test: 100k steps, depth=4, 2 seeds ==="

echo "1/3: Ortho optimizer..."
uv run python scripts/throughput_benchmark.py \
    --envs Breakout-MinAtar \
    --depths 4 \
    --timesteps 100000 \
    --num-envs 512 \
    --num-seeds 2 \
    --eval-freq 50000 \
    --ortho-mode optimizer \
    --ortho-coeff 1e-3 \
    --activation groupsort

echo "2/3: Baseline..."
uv run python scripts/throughput_benchmark.py \
    --envs Breakout-MinAtar \
    --depths 4 \
    --timesteps 100000 \
    --num-envs 512 \
    --num-seeds 2 \
    --eval-freq 50000 \
    --ortho-mode none \
    --activation tanh

echo "3/3: Ortho loss..."
uv run python scripts/throughput_benchmark.py \
    --envs Breakout-MinAtar \
    --depths 4 \
    --timesteps 100000 \
    --num-envs 512 \
    --num-seeds 2 \
    --eval-freq 50000 \
    --ortho-mode loss \
    --ortho-lambda 0.2 \
    --activation groupsort

echo "=== Smoke test passed ==="
