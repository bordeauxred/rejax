#!/bin/bash
# Experiment 4: L2-Init Regularization with Permutations
#
# Goal: Test L2-Init (regenerative regularization) as plasticity preservation method
#
# Network:      256-256-256-256 MLP (standard)
# Steps/game:   25M (same as exp2)
# Cycles:       10 (same as exp2)
# Methods:      MLP baseline vs MLP L2-Init (λ=0.001) vs MLP L2-Init (λ=0.01)
# Seeds:        2
# Games:        4 (excludes Seaquest - reward scale outlier)
# Permutation:  YES - all 10 channels shuffled + game order
#
# L2-INIT: Regenerative regularization from Lyle et al. (2023)
#   - Formula: loss += λ * ||θ - θ₀||²
#   - Keeps weights close to initial random distribution
#   - Prevents rank collapse (unlike standard L2 towards zero)
#   - λ=0.001 (user recommended) and λ=0.01 (literature) tested
#
# HYPOTHESIS: L2-Init will maintain plasticity by preventing weight drift,
#             showing less performance degradation over cycles compared to baseline.
#
# RUNTIME: 25M × 4 games × 10 cycles × 3 methods × 2 seeds = 6B steps
#          → ~12h on H100
#
# Usage:
#   ./scripts/run_continual_exp4_l2_init.sh                    # Full run
#   ./scripts/run_continual_exp4_l2_init.sh 1000000 2 1        # Quick test

set -e

STEPS_PER_GAME=${1:-25000000}
NUM_CYCLES=${2:-10}
NUM_SEEDS=${3:-2}
NUM_ENVS=${4:-2048}
EVAL_FREQ=${5:-500000}

EXPERIMENT_NAME="continual_exp4_l2_init_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
WANDB_PROJECT="continual_minatar_l2_init"

echo "=============================================================="
echo "EXPERIMENT 4: L2-INIT REGULARIZATION + PERMUTATIONS"
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
echo "  1. mlp_baseline    - MLP 256x4, ReLU, standard PPO"
echo "  2. mlp_l2_init     - MLP 256x4, ReLU, L2-Init (λ=0.001)"
echo "  3. mlp_l2_init_0.01 - MLP 256x4, ReLU, L2-Init (λ=0.01)"
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
    --configs mlp_baseline mlp_l2_init mlp_l2_init_0.01 \
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
