#!/bin/bash
# Single-task MinAtar baseline validation
# Run before continual learning to verify configs work

set -e

TIMESTEPS=20000000
NUM_SEEDS=3
EVAL_FREQ=500000

echo "========================================"
echo "Single-Task MinAtar Baseline Validation"
echo "========================================"
echo "Timesteps: $TIMESTEPS"
echo "Seeds: $NUM_SEEDS"
echo ""

# pgx_baseline (CNN) - recommended
echo ">>> Running pgx_baseline (CNN)..."
uv run python scripts/bench_single_strong.py \
  --config pgx_baseline \
  --timesteps $TIMESTEPS \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --padded \
  --plot

# mlp_baseline
echo ">>> Running mlp_baseline..."
uv run python scripts/bench_single_strong.py \
  --config mlp_baseline \
  --timesteps $TIMESTEPS \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --padded \
  --plot

echo ""
echo "========================================"
echo "Done! Check results/ for outputs"
echo "========================================"
echo ""
echo "Hard targets at 20M steps:"
echo "  Asterix:        ~25"
echo "  Breakout:       ~40"
echo "  Freeway:        ~60"
echo "  Seaquest:       ~55-60"
echo "  SpaceInvaders:  ~150"
