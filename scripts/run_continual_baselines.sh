#!/bin/bash
# Continual learning experiment: Baseline vs AdaMO vs AdaMO+Lyle
# Cycles through all 5 MinAtar games without weight resets

set -e

STEPS_PER_GAME=20000000
NUM_CYCLES=5
NUM_SEEDS=3
EVAL_FREQ=500000
WANDB_PROJECT="rejax-continual-baselines_improved_ppo"

echo "========================================"
echo "Continual Learning: Baseline vs AdaMO"
echo "========================================"
echo "Steps per game: $STEPS_PER_GAME"
echo "Cycles: $NUM_CYCLES"
echo "Seeds: $NUM_SEEDS"
echo "WandB: $WANDB_PROJECT"
echo ""

# CNN comparisons (fair: same architecture, same hyperparams)
echo ">>> [1/6] pgx_baseline (CNN baseline)..."
uv run python scripts/bench_continual.py \
  --configs pgx_baseline \
  --steps-per-game $STEPS_PER_GAME \
  --num-cycles $NUM_CYCLES \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --use-wandb \
  --wandb-project $WANDB_PROJECT

echo ">>> [2/6] cnn_adamo (CNN + AdaMO)..."
uv run python scripts/bench_continual.py \
  --configs cnn_adamo \
  --steps-per-game $STEPS_PER_GAME \
  --num-cycles $NUM_CYCLES \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --use-wandb \
  --wandb-project $WANDB_PROJECT

echo ">>> [3/6] cnn_adamo_lyle_continual (CNN + AdaMO + Lyle schedule)..."
uv run python scripts/bench_continual.py \
  --configs cnn_adamo_lyle_continual \
  --steps-per-game $STEPS_PER_GAME \
  --num-cycles $NUM_CYCLES \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --use-wandb \
  --wandb-project $WANDB_PROJECT

# MLP comparisons
echo ">>> [4/6] mlp_baseline (MLP baseline)..."
uv run python scripts/bench_continual.py \
  --configs mlp_baseline \
  --steps-per-game $STEPS_PER_GAME \
  --num-cycles $NUM_CYCLES \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --use-wandb \
  --wandb-project $WANDB_PROJECT

echo ">>> [5/6] mlp_adamo (MLP + AdaMO + groupsort)..."
uv run python scripts/bench_continual.py \
  --configs mlp_adamo \
  --steps-per-game $STEPS_PER_GAME \
  --num-cycles $NUM_CYCLES \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --use-wandb \
  --wandb-project $WANDB_PROJECT

echo ">>> [6/6] mlp_adamo_lyle_continual (MLP + AdaMO + Lyle schedule)..."
uv run python scripts/bench_continual.py \
  --configs mlp_adamo_lyle_continual \
  --steps-per-game $STEPS_PER_GAME \
  --num-cycles $NUM_CYCLES \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --use-wandb \
  --wandb-project $WANDB_PROJECT

echo ""
echo "========================================"
echo "Done! Check wandb for results"
echo "========================================"
echo ""
echo "Comparisons:"
echo "  CNN: pgx_baseline vs cnn_adamo vs cnn_adamo_lyle_continual"
echo "  MLP: mlp_baseline vs mlp_adamo vs mlp_adamo_lyle_continual"
echo ""
echo "Key metrics to compare:"
echo "  - Forward transfer (first cycle performance)"
echo "  - Backward transfer (later cycles vs cycle 1)"
echo "  - Plasticity loss over 5 cycles"
echo "  - Final performance after 5 cycles"
