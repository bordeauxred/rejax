#!/bin/bash
# Ortho scaling benchmark: depth 2,4,8,16,32
# Grid search over ortho coefficients
# Logs to disk + wandb
# H100 optimized: 2048 envs, 5 seeds

set -e

ENVS="Breakout-MinAtar Asterix-MinAtar"
DEPTHS="2 4 8 16 32"
TIMESTEPS=10000000  # 10M steps
NUM_ENVS=2048
NUM_SEEDS=5
EVAL_FREQ=500000

echo "=============================================="
echo "Ortho Scaling Benchmark + Grid Search"
echo "Envs: $ENVS"
echo "Depths: $DEPTHS"
echo "Timesteps: $TIMESTEPS"
echo "Num envs: $NUM_ENVS"
echo "Num seeds: $NUM_SEEDS"
echo "=============================================="

# =============================================
# 1. ORTHO OPTIMIZER MODE (grid: 1e-4, 1e-3, 1e-2, 1e-1)
# =============================================

echo ""
echo "=== Ortho Optimizer (groupsort, coeff=1e-4) ==="
uv run python scripts/throughput_benchmark.py \
    --envs $ENVS \
    --depths $DEPTHS \
    --timesteps $TIMESTEPS \
    --num-envs $NUM_ENVS \
    --num-seeds $NUM_SEEDS \
    --eval-freq $EVAL_FREQ \
    --ortho-mode optimizer \
    --ortho-coeff 1e-4 \
    --activation groupsort \
    --use-wandb

echo ""
echo "=== Ortho Optimizer (groupsort, coeff=1e-3) ==="
uv run python scripts/throughput_benchmark.py \
    --envs $ENVS \
    --depths $DEPTHS \
    --timesteps $TIMESTEPS \
    --num-envs $NUM_ENVS \
    --num-seeds $NUM_SEEDS \
    --eval-freq $EVAL_FREQ \
    --ortho-mode optimizer \
    --ortho-coeff 1e-3 \
    --activation groupsort \
    --use-wandb

echo ""
echo "=== Ortho Optimizer (groupsort, coeff=1e-2) ==="
uv run python scripts/throughput_benchmark.py \
    --envs $ENVS \
    --depths $DEPTHS \
    --timesteps $TIMESTEPS \
    --num-envs $NUM_ENVS \
    --num-seeds $NUM_SEEDS \
    --eval-freq $EVAL_FREQ \
    --ortho-mode optimizer \
    --ortho-coeff 1e-2 \
    --activation groupsort \
    --use-wandb

echo ""
echo "=== Ortho Optimizer (groupsort, coeff=1e-1) ==="
uv run python scripts/throughput_benchmark.py \
    --envs $ENVS \
    --depths $DEPTHS \
    --timesteps $TIMESTEPS \
    --num-envs $NUM_ENVS \
    --num-seeds $NUM_SEEDS \
    --eval-freq $EVAL_FREQ \
    --ortho-mode optimizer \
    --ortho-coeff 1e-1 \
    --activation groupsort \
    --use-wandb

# =============================================
# 2. BASELINE PPO (tanh, no ortho)
# =============================================

echo ""
echo "=== Baseline PPO (tanh, no ortho) ==="
uv run python scripts/throughput_benchmark.py \
    --envs $ENVS \
    --depths $DEPTHS \
    --timesteps $TIMESTEPS \
    --num-envs $NUM_ENVS \
    --num-seeds $NUM_SEEDS \
    --eval-freq $EVAL_FREQ \
    --ortho-mode none \
    --activation tanh \
    --use-wandb

# =============================================
# 3. ORTHO LOSS MODE (grid: 0.02, 0.2)
# =============================================

echo ""
echo "=== Ortho Loss (groupsort, lambda=0.02) ==="
uv run python scripts/throughput_benchmark.py \
    --envs $ENVS \
    --depths $DEPTHS \
    --timesteps $TIMESTEPS \
    --num-envs $NUM_ENVS \
    --num-seeds $NUM_SEEDS \
    --eval-freq $EVAL_FREQ \
    --ortho-mode loss \
    --ortho-lambda 0.02 \
    --activation groupsort \
    --use-wandb

echo ""
echo "=== Ortho Loss (groupsort, lambda=0.2) ==="
uv run python scripts/throughput_benchmark.py \
    --envs $ENVS \
    --depths $DEPTHS \
    --timesteps $TIMESTEPS \
    --num-envs $NUM_ENVS \
    --num-seeds $NUM_SEEDS \
    --eval-freq $EVAL_FREQ \
    --ortho-mode loss \
    --ortho-lambda 0.2 \
    --activation groupsort \
    --use-wandb

echo ""
echo "=============================================="
echo "Scaling Benchmark Complete"
echo "Results in: benchmark_results/"
echo "=============================================="
echo ""
echo "Total runs: 7 configs x 2 envs x 5 depths = 70 benchmarks"
