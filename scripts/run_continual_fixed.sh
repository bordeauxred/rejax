#!/bin/bash
# Continual learning experiment with OOM fix
# Uses continual_pipeline.py instead of bench_continual.py
#
# Cycles through all 5 MinAtar games without weight resets
# Compares baseline vs AdaMO vs AdaMO+Lyle

set -e

STEPS_PER_GAME=20000000
NUM_CYCLES=5
NUM_SEEDS=2
EVAL_FREQ=500000
WANDB_PROJECT="continual_minatar_fixed"

echo "=========================================="
echo "Continual Learning (OOM-Fixed Pipeline)"
echo "=========================================="
echo "Steps per game: $STEPS_PER_GAME"
echo "Cycles: $NUM_CYCLES"
echo "Seeds: $NUM_SEEDS"
echo "Eval freq: $EVAL_FREQ"
echo "WandB: $WANDB_PROJECT"
echo ""
echo "Using continual_pipeline.py with cached eval (OOM fix)"
echo ""

echo ">>> [1/6] cnn_adamo (CNN + AdaMO)..."
uv run python scripts/continual_pipeline.py \
  --configs cnn_adamo \
  --steps-per-game $STEPS_PER_GAME \
  --num-cycles $NUM_CYCLES \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --use-wandb \
  --wandb-project $WANDB_PROJECT

echo ">>> [2/6] pgx_baseline (CNN baseline)..."
uv run python scripts/continual_pipeline.py \
  --configs pgx_baseline \
  --steps-per-game $STEPS_PER_GAME \
  --num-cycles $NUM_CYCLES \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --use-wandb \
  --wandb-project $WANDB_PROJECT

echo ">>> [3/6] cnn_adamo_lyle_continual (CNN + AdaMO + Lyle schedule)..."
uv run python scripts/continual_pipeline.py \
  --configs cnn_adamo_lyle_continual \
  --steps-per-game $STEPS_PER_GAME \
  --num-cycles $NUM_CYCLES \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --use-wandb \
  --wandb-project $WANDB_PROJECT

echo ">>> [4/6] mlp_baseline (MLP baseline)..."
uv run python scripts/continual_pipeline.py \
  --configs mlp_baseline \
  --steps-per-game $STEPS_PER_GAME \
  --num-cycles $NUM_CYCLES \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --use-wandb \
  --wandb-project $WANDB_PROJECT

echo ">>> [5/6] mlp_adamo (MLP + AdaMO + groupsort)..."
uv run python scripts/continual_pipeline.py \
  --configs mlp_adamo \
  --steps-per-game $STEPS_PER_GAME \
  --num-cycles $NUM_CYCLES \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --use-wandb \
  --wandb-project $WANDB_PROJECT

echo ">>> [6/6] mlp_adamo_lyle_continual (MLP + AdaMO + Lyle schedule)..."
uv run python scripts/continual_pipeline.py \
  --configs mlp_adamo_lyle_continual \
  --steps-per-game $STEPS_PER_GAME \
  --num-cycles $NUM_CYCLES \
  --num-seeds $NUM_SEEDS \
  --eval-freq $EVAL_FREQ \
  --use-wandb \
  --wandb-project $WANDB_PROJECT

echo ""
echo "=========================================="
echo "Done! Check wandb for results"
echo "=========================================="
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
