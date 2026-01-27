#!/bin/bash
# Paper-Ready Experiment: Small Network (64x4) + Permutations
#
# Goal: Demonstrate plasticity loss in continual RL with statistical rigor
#
# Network:      64-64-64-64 MLP (small, stress capacity)
# Steps/game:   15M (sufficient for convergence)
# Cycles:       20 (long-term degradation)
# Methods:      MLP baseline small vs MLP AdaMO small
# Seeds:        8 (seeds 26-33, sequential for learning curves)
# Games:        4 (Breakout, Asterix, SpaceInvaders, Freeway)
# Permutation:  YES - channels + game order
#
# FIRST-GAME BALANCE (seeds 26-33):
#   Breakout:      seeds 30, 33
#   Asterix:       seeds 28, 31
#   SpaceInvaders: seeds 26, 32
#   Freeway:       seeds 27, 29
#
# OUTPUT: Full learning curves with intermediate evals (for Lyle-style plots)
#
# RUNTIME: 15M × 4 games × 20 cycles × 2 methods × 8 seeds = 19.2B steps
#          → ~64h on H100 (sequential seeds for learning curve data)
#
# Usage:
#   ./scripts/benchmark_continual_minatar_paper_64x4.sh                                # Full 8 seeds
#   ./scripts/benchmark_continual_minatar_paper_64x4.sh 1000000 2 4 2048 100000        # Quick test

set -e

# Prevent JAX from pre-allocating all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Or use fixed fraction: export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12

STEPS_PER_GAME=${1:-15000000}
NUM_CYCLES=${2:-20}
NUM_SEEDS=${3:-8}
NUM_ENVS=${4:-2048}
EVAL_FREQ=${5:-500000}
BASE_SEED=${6:-26}  # Seeds 26-33 give balanced first-game distribution (2 each)

# Fixed experiment name for reproducibility
EXPERIMENT_NAME="paper_continual_64x4_permuted"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
WANDB_PROJECT="adamo_continual_paper"

echo "=============================================================="
echo "PAPER EXPERIMENT: 64x4 + PERMUTATIONS + 20 CYCLES"
echo "=============================================================="
echo "Steps per game: $STEPS_PER_GAME"
echo "Cycles: $NUM_CYCLES"
echo "Seeds: $NUM_SEEDS (base: $BASE_SEED, range: $BASE_SEED-$((BASE_SEED + NUM_SEEDS - 1)))"
echo "Envs: $NUM_ENVS"
echo "Eval freq: $EVAL_FREQ (for learning curves)"
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
echo ""
echo "rliable stats: 4 games × $NUM_SEEDS seeds = $((4 * NUM_SEEDS)) datapoints"
echo "=============================================================="

cd "$(dirname "$0")/.."

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Log start time
echo "Started at: $(date) | Seeds $BASE_SEED-$((BASE_SEED + NUM_SEEDS - 1))" | tee "$OUTPUT_DIR/run.log"

uv run python scripts/bench_continual.py \
    --steps-per-game $STEPS_PER_GAME \
    --num-cycles $NUM_CYCLES \
    --num-seeds $NUM_SEEDS \
    --seed $BASE_SEED \
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
echo "=============================================================="
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run.log"
echo "Results saved to: $OUTPUT_DIR"
