#!/bin/bash
# Smoke test: All baselines on all MinAtar games (single-task)
# Target: ~20-30 min on A100, verify learning + no bugs before overnight continual runs
#
# Configs tested:
#   - mlp_baseline:    MLP 4x256, tanh, orthogonal init
#   - mlp_adamo:       MLP 4x256, groupsort, AdaMO optimizer
#   - mlp_adamo_lyle:  MLP 4x256, groupsort, AdaMO + Lyle LR (6.25e-5)
#   - cnn_baseline:    CNN pgx-style, relu
#   - cnn_adamo:       CNN pgx-style, relu, AdaMO optimizer
#   - cnn_adamo_lyle:  CNN pgx-style, relu, AdaMO + Lyle LR
#
# Usage:
#   ./scripts/smoketest_single_task.sh              # Default: 5M steps (25% of full), 2 seeds
#   ./scripts/smoketest_single_task.sh 2000000 1    # Fast: 2M steps, 1 seed (~10 min)
#   ./scripts/smoketest_single_task.sh 10000000 3   # Long: 10M steps, 3 seeds (~1hr)

set -e

TIMESTEPS=${1:-5000000}
NUM_SEEDS=${2:-2}
NUM_ENVS=${3:-4096}

echo "=============================================================="
echo "SMOKE TEST: All Baselines on MinAtar (Single-Task)"
echo "=============================================================="
echo "Timesteps: $TIMESTEPS (pgx uses 20M)"
echo "Seeds: $NUM_SEEDS"
echo "Envs: $NUM_ENVS"
echo "=============================================================="

cd "$(dirname "$0")/.."

uv run python scripts/smoketest_all_baselines.py \
    --timesteps $TIMESTEPS \
    --num-seeds $NUM_SEEDS \
    --num-envs $NUM_ENVS

echo ""
echo "Ready for overnight continual learning runs!"
