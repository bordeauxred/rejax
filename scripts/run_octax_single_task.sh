#!/bin/bash
# Octax single-task de-risk experiments
# Validates that our PPO implementation matches paper returns on individual games
#
# Usage:
#   ./scripts/run_octax_single_task.sh                # Full run (10M steps, all games)
#   ./scripts/run_octax_single_task.sh 1000000 1 512  # Quick test (1M steps, 1 seed)

set -e

# Prevent memory fragmentation on multi-GPU systems
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.485  # 48.5% GPU RAM

STEPS=${1:-10000000}   # 10M per game for de-risk (paper uses 5M)
NUM_SEEDS=${2:-3}
NUM_ENVS=${3:-512}     # Paper default: 512

echo "=============================================="
echo "Octax Single-Task De-Risk Experiments"
echo "=============================================="
echo "Steps: $STEPS"
echo "Num seeds: $NUM_SEEDS"
echo "Num envs: $NUM_ENVS"
echo "=============================================="

# Run each game as single-task baseline with paper config
# Expected returns (paper Table 1, 5M steps):
#   - Brix: ~21
#   - Tetris: ~1.5
#   - Tank: ~3.5
#   - SpaceJam: ~12
#   - Deep: ~35
for GAME in brix tetris tank spacejam deep; do
    echo ""
    echo "=== Training on $GAME ==="
    uv run python scripts/bench_octax_continual.py \
        --single-task $GAME \
        --steps-per-task $STEPS \
        --num-seeds $NUM_SEEDS \
        --num-envs $NUM_ENVS \
        --configs paper_256x1 \
        --use-wandb
done

echo ""
echo "Done! Results saved to results/octax_continual/"
