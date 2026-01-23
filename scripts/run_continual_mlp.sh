#!/bin/bash
# Continual Learning Experiments: MLP-based
#
# Experiments:
#   1. mlp_baseline:           MLP 4x256, tanh, standard PPO
#   2. mlp_adamo:              MLP 4x256, groupsort, AdaMO orthogonalization
#   3. mlp_adamo_lyle_continual: MLP 4x256, groupsort, AdaMO + Lyle schedule (warmup+cosine, reset/game)
#
# Setup: 2 cycles through all 5 MinAtar games, 3 seeds
# Expected runtime: ~4-6 hours on A100
#
# Usage:
#   ./scripts/run_continual_mlp.sh                    # Full run
#   ./scripts/run_continual_mlp.sh 1000000 1 1        # Quick test: 1M steps, 1 cycle, 1 seed

set -e

STEPS_PER_GAME=${1:-10000000}
NUM_CYCLES=${2:-2}
NUM_SEEDS=${3:-3}
NUM_ENVS=${4:-2048}

EXPERIMENT_NAME="continual_mlp_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"

echo "=============================================================="
echo "CONTINUAL LEARNING: MLP Experiments"
echo "=============================================================="
echo "Steps per game: $STEPS_PER_GAME"
echo "Cycles: $NUM_CYCLES"
echo "Seeds: $NUM_SEEDS"
echo "Envs: $NUM_ENVS"
echo "Output: $OUTPUT_DIR"
echo "=============================================================="
echo "Configs:"
echo "  1. mlp_baseline           - MLP 4x256, tanh, standard PPO"
echo "  2. mlp_adamo              - MLP 4x256, groupsort, AdaMO"
echo "  3. mlp_adamo_lyle_continual - MLP 4x256, groupsort, AdaMO + Lyle LR (warmup+cosine)"
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
    --configs mlp_baseline mlp_adamo mlp_adamo_lyle_continual \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --use-wandb \
    --wandb-project "rejax-continual" \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

echo ""
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run.log"
echo "Results saved to: $OUTPUT_DIR"
