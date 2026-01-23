#!/bin/bash
# Continual Learning Experiments: CNN-based
#
# Experiments:
#   1. cnn_baseline:           CNN (conv16-k3 + 4x256 MLP), ReLU, standard PPO
#   2. cnn_adamo:              CNN (conv16-k3 + 4x256 MLP), ReLU, AdaMO orthogonalization
#   3. cnn_adamo_lyle_continual: CNN (conv16-k3 + 4x256 MLP), ReLU, AdaMO + Lyle schedule (warmup+cosine, reset/game)
#
# Note: CNN uses ReLU throughout (groupsort incompatible with conv layers)
# Note: CNN has 4x256 MLP on top of conv (same depth as MLP experiments for fair comparison)
#
# Setup: 2 cycles through all 5 MinAtar games, 3 seeds
# Expected runtime: ~4-6 hours on A100
#
# Usage:
#   ./scripts/run_continual_cnn.sh                    # Full run
#   ./scripts/run_continual_cnn.sh 1000000 1 1        # Quick test: 1M steps, 1 cycle, 1 seed

set -e

STEPS_PER_GAME=${1:-10000000}
NUM_CYCLES=${2:-2}
NUM_SEEDS=${3:-3}
NUM_ENVS=${4:-2048}

EXPERIMENT_NAME="continual_cnn_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"

echo "=============================================================="
echo "CONTINUAL LEARNING: CNN Experiments"
echo "=============================================================="
echo "Steps per game: $STEPS_PER_GAME"
echo "Cycles: $NUM_CYCLES"
echo "Seeds: $NUM_SEEDS"
echo "Envs: $NUM_ENVS"
echo "Output: $OUTPUT_DIR"
echo "=============================================================="
echo "Configs:"
echo "  1. cnn_baseline           - CNN conv16 + 4x256 MLP, ReLU, standard PPO"
echo "  2. cnn_adamo              - CNN conv16 + 4x256 MLP, ReLU, AdaMO"
echo "  3. cnn_adamo_lyle_continual - CNN conv16 + 4x256 MLP, ReLU, AdaMO + Lyle LR (warmup+cosine)"
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
    --configs cnn_baseline cnn_adamo cnn_adamo_lyle_continual \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --use-wandb \
    --wandb-project "rejax-continual" \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"

echo ""
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run.log"
echo "Results saved to: $OUTPUT_DIR"
