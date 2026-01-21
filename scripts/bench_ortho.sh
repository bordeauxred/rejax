#!/bin/bash
# Benchmark PPO with ortho modes to measure overhead
# Tests both loss mode and optimizer mode regularization

set -e

echo "=============================================="
echo "Ortho Regularization Throughput Benchmark"
echo "=============================================="
echo "=== Ortho Loss Mode (groupsort) ==="
uv run python scripts/throughput_benchmark.py \
    --envs Breakout-MinAtar \
    --depths 4 8 \
    --timesteps 1000000 \
    --ortho-mode loss \
    --ortho-lambda 0.2 \
    --activation groupsort

echo ""
echo "=== Ortho Optimizer Mode (tanh) ==="
uv run python scripts/throughput_benchmark.py \
    --envs Breakout-MinAtar \
    --depths 4 8 \
    --timesteps 1000000 \
    --ortho-mode optimizer \
    --ortho-coeff 1e-3 \
    --activation tanh

echo ""
echo "=== Ortho Optimizer Mode (groupsort) ==="
uv run python scripts/throughput_benchmark.py \
    --envs Breakout-MinAtar \
    --depths 4 8 \
    --timesteps 1000000 \
    --ortho-mode optimizer \
    --ortho-coeff 1e-3 \
    --activation groupsort

echo ""


echo ""
echo "=== No Ortho (baseline) ==="
uv run python scripts/throughput_benchmark.py \
    --envs Breakout-MinAtar \
    --depths 4 8 \
    --timesteps 1000000 \
    --ortho-mode none \
    --activation tanh

echo ""
echo "=== Ortho Loss Mode (tanh) ==="
uv run python scripts/throughput_benchmark.py \
    --envs Breakout-MinAtar \
    --depths 4 8 \
    --timesteps 1000000 \
    --ortho-mode loss \
    --ortho-lambda 0.2 \
    --activation tanh

echo ""


echo "=============================================="
echo "Benchmark Complete"
echo "=============================================="
