#!/bin/bash
# Experiment 5: NaP (Normalize-and-Project) with Permutations
#
# Goal: Test NaP as plasticity preservation method
#
# Network:      64-64-64-64 MLP (small, same as exp4 style)
# Steps/game:   15M (matching other 64x4 experiments)
# Cycles:       20 (longer to see degradation)
# Methods:      MLP NaP only (baseline already in other experiments)
# Seeds:        2
# Games:        4 (excludes Seaquest - reward scale outlier)
# Permutation:  YES - all 10 channels shuffled + game order
#
# NaP: Normalize-and-Project from Lyle et al. (NeurIPS 2024)
#   - Formula: After each step, W <- (rho * W) / ||W|| where rho = initial norm
#   - Decouples effective learning rate from parameter norm growth
#   - Maintains initial weight scale throughout training
#
# HYPOTHESIS: NaP will maintain plasticity by preventing weight norm growth,
#             showing less performance degradation over cycles compared to baseline.
#
# RUNTIME: 15M x 4 games x 20 cycles x 1 method x 2 seeds = 2.4B steps
#          -> ~4h on H100 (small network is fast)
#
# Usage:
#   ./scripts/run_continual_exp5_nap.sh                    # Full run
#   ./scripts/run_continual_exp5_nap.sh 1000000 2 1        # Quick test

set -e

STEPS_PER_GAME=${1:-15000000}
NUM_CYCLES=${2:-20}
NUM_SEEDS=${3:-2}
NUM_ENVS=${4:-2048}
EVAL_FREQ=${5:-500000}

EXPERIMENT_NAME="continual_exp5_nap_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
WANDB_PROJECT="continual_minatar_permutations"  # Same as exp2 for comparison

echo "=============================================================="
echo "EXPERIMENT 5: NaP (NORMALIZE-AND-PROJECT) + PERMUTATIONS"
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
echo "  1. mlp_nap_small - MLP 64x4, ReLU, NaP projection"
echo "  (baseline already in other experiments)"
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
    --configs mlp_nap_small \
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
