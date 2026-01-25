#!/bin/bash
# Experiment 1: Small Network
#
# Goal: Force plasticity loss to manifest earlier with reduced capacity
#
# Network:      64-64-64 MLP (vs current 256-256-256-256)
# Steps/game:   20M (same as 4-cycle run)
# Cycles:       4 (same as 4-cycle run)
# Methods:      MLP baseline vs MLP AdaMO (2 configs)
# Seeds:        2
# Permutation:  NONE (pure network size ablation)
#
# HYPOTHESIS: With 25x fewer params, dead neurons have bigger impact.
#             Baseline should show performance degradation by cycle 3-4.
#
# RUNTIME: 20M × 5 games × 4 cycles × 2 methods × 2 seeds = 1.6B steps
#          → ~3h on A100
#
# Usage:
#   ./scripts/run_continual_exp1_small_network.sh                    # Full run
#   ./scripts/run_continual_exp1_small_network.sh 1000000 1 1        # Quick test

set -e

STEPS_PER_GAME=${1:-20000000}
NUM_CYCLES=${2:-4}
NUM_SEEDS=${3:-3}
NUM_ENVS=${4:-2048}
EVAL_FREQ=${5:-500000}

EXPERIMENT_NAME="continual_exp1_small_network_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
WANDB_PROJECT="continual_minatar_small_network"

echo "=============================================================="
echo "EXPERIMENT 1: SMALL NETWORK"
echo "=============================================================="
echo "Steps per game: $STEPS_PER_GAME"
echo "Cycles: $NUM_CYCLES"
echo "Seeds: $NUM_SEEDS"
echo "Envs: $NUM_ENVS"
echo "Eval freq: $EVAL_FREQ"
echo "Output: $OUTPUT_DIR"
echo "W&B Project: $WANDB_PROJECT"
echo "=============================================================="
echo "Configs:"
echo "  1. mlp_baseline_small - MLP 64-64-64, ReLU, standard PPO"
echo "  2. mlp_adamo_small    - MLP 64-64-64, groupsort, AdaMO"
echo "=============================================================="

cd "$(dirname "$0")/.."

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Log start time
echo "Started at: $(date)" | tee "$OUTPUT_DIR/run.log"

uv run python scripts/bench_continual.py \
    --steps-per-game $STEPS_PER_GAME \
    --num-cycles $NUM_CYCLES \
    --num-seeds $NUM_SEEDS \
    --num-envs $NUM_ENVS \
    --eval-freq $EVAL_FREQ \
    --configs mlp_baseline_small mlp_adamo_small \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

echo ""
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run.log"
echo "Results saved to: $OUTPUT_DIR"
