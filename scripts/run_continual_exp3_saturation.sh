#!/bin/bash
# Experiment 3: Full Saturation
#
# Goal: Completely saturate each game before switching
#
# Network:      256-256-256-256 MLP (standard)
# Steps/game:   50M (2.5x more than previous)
# Cycles:       2 (limited for runtime)
# Methods:      MLP baseline vs MLP AdaMO (2 configs)
# Seeds:        2
# Permutation:  NONE (pure saturation test)
#
# HYPOTHESIS: With more steps per game, network fully overfits to each.
#             More catastrophic forgetting on game switch.
#             Even 2 cycles may show clear degradation.
#
# RUNTIME: 50M × 5 games × 2 cycles × 2 methods × 2 seeds = 2B steps
#          → ~12h on A100
#
# Usage:
#   ./scripts/run_continual_exp3_saturation.sh                    # Full run
#   ./scripts/run_continual_exp3_saturation.sh 5000000 1 1        # Quick test

set -e

STEPS_PER_GAME=${1:-100000000}
NUM_CYCLES=${2:-2}
NUM_SEEDS=${3:-3}
NUM_ENVS=${4:-2048}
EVAL_FREQ=${5:-1000000}

EXPERIMENT_NAME="continual_exp3_saturation_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
WANDB_PROJECT="continual_minatar_saturation"

echo "=============================================================="
echo "EXPERIMENT 3: FULL SATURATION (50M STEPS/GAME)"
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
echo "  1. mlp_baseline - MLP 256x4, ReLU, standard PPO"
echo "  2. mlp_adamo    - MLP 256x4, groupsort, AdaMO"
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
    --configs mlp_baseline mlp_adamo \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

echo ""
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run.log"
echo "Results saved to: $OUTPUT_DIR"
