#!/bin/bash
# Experiment 5: Small Network + Permutations + 20 Cycles
#
# Goal: Combine small network stress test with permutation challenge
#
# Network:      64-64-64-64 MLP (small, reduced capacity)
# Steps/game:   15M (moderate training per task)
# Cycles:       20 (long-term degradation)
# Methods:      MLP baseline small vs MLP AdaMO small (2 configs)
# Seeds:        2
# Games:        4 (excludes Seaquest - reward scale outlier)
# Permutation:  YES - all 10 channels shuffled + game order
#
# HYPOTHESIS: Small network + permutations = maximum plasticity stress.
#             Should see clear degradation in baseline by cycle 10-15.
#             AdaMO should maintain performance across all 20 cycles.
#
# RUNTIME: 15M × 4 games × 20 cycles × 2 methods × 2 seeds = 4.8B steps
#          → ~16h on H100
#
# Usage:
#   ./scripts/run_continual_exp5_small_permuted.sh                    # Full run
#   ./scripts/run_continual_exp5_small_permuted.sh 1000000 2 1        # Quick test

set -e

STEPS_PER_GAME=${1:-15000000}
NUM_CYCLES=${2:-20}
NUM_SEEDS=${3:-2}
NUM_ENVS=${4:-2048}
EVAL_FREQ=${5:-500000}

EXPERIMENT_NAME="channel_permutation_4games_64x4_20cycles"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
WANDB_PROJECT="continual_minatar_small_permuted"

echo "=============================================================="
echo "EXPERIMENT 5: SMALL NETWORK + PERMUTATIONS + 20 CYCLES"
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
echo "  1. mlp_baseline_small - MLP 64x4, ReLU, standard PPO"
echo "  2. mlp_adamo_small    - MLP 64x4, groupsort, AdaMO"
echo ""
echo "Permutations ENABLED:"
echo "  - Channel permutation: all 10 channels shuffled per game/cycle"
echo "  - Game order shuffle: random per cycle"
echo ""
echo "Excluded: Seaquest-MinAtar (reward scale outlier)"
echo "Games: Breakout, Asterix, SpaceInvaders, Freeway"
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
    --permute-channels \
    --random-game-order \
    --exclude-games Seaquest-MinAtar \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

echo ""
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run.log"
echo "Results saved to: $OUTPUT_DIR"
