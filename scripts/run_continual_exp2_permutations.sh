#!/bin/bash
# Experiment 2: Permutations + More Cycles
#
# Goal: Break memorization of fixed game order + see long-term degradation
#
# Network:      256-256-256-256 MLP (standard)
# Steps/game:   10M (faster iteration)
# Cycles:       10 (enough to see degradation)
# Methods:      MLP baseline vs MLP AdaMO (2 configs)
# Seeds:        2
# Permutation:  YES - both channels AND game order
#
# FEATURES:
#   - Channel permutation: randomly permute obs channels per game
#   - Game order permutation: shuffle game order each cycle
#   - Both prevent network from memorizing fixed sequences
#
# HYPOTHESIS: Without fixed patterns to exploit, baseline will degrade
#             faster. Should see performance cliff by cycle 6-8.
#
# RUNTIME: 10M × 5 games × 10 cycles × 2 methods × 2 seeds = 2B steps
#          → ~12h on A100
#
# Usage:
#   ./scripts/run_continual_exp2_permutations.sh                    # Full run
#   ./scripts/run_continual_exp2_permutations.sh 1000000 2 1        # Quick test

set -e

STEPS_PER_GAME=${1:-10000000}
NUM_CYCLES=${2:-10}
NUM_SEEDS=${3:-2}
NUM_ENVS=${4:-2048}
EVAL_FREQ=${5:-500000}

EXPERIMENT_NAME="continual_exp2_permutations_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
WANDB_PROJECT="continual_minatar_permutations"

echo "=============================================================="
echo "EXPERIMENT 2: PERMUTATIONS + 10 CYCLES"
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
echo ""
echo "Permutations ENABLED:"
echo "  - Channel permutation: random per game/cycle"
echo "  - Game order shuffle: random per cycle"
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
    --permute-channels \
    --random-game-order \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

echo ""
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run.log"
echo "Results saved to: $OUTPUT_DIR"
