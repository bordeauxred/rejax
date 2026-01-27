#!/bin/bash
# Run 2 Octax games in parallel on GPU (H100)
# Paper config: single 256-unit MLP, 8 epochs
# High env count for better gradient estimates (fewer seeds needed)
#
# Usage: ./scripts/run_octax_2games_gpu.sh [steps] [seeds] [envs]

set -e

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

STEPS=${1:-5000000}
NUM_SEEDS=${2:-1}       # Fewer seeds needed with high env count
NUM_ENVS=${3:-4096}     # H100 can handle 4k+ envs easily

echo "=============================================="
echo "Octax 2-Game Parallel GPU Benchmark (H100)"
echo "=============================================="
echo "Steps: $STEPS | Seeds: $NUM_SEEDS | Envs: $NUM_ENVS"
echo "Games: brix, tetris"
echo "Config: paper_256x1 (single 256 MLP, 8 epochs)"
echo "=============================================="

# Brix (expected ~21) and Tetris (expected ~1.5)
uv run python scripts/bench_octax_single.py \
    --game brix \
    --steps $STEPS \
    --num-seeds $NUM_SEEDS \
    --num-envs $NUM_ENVS \
    --config paper_256x1 \
    --use-wandb &
PID1=$!

uv run python scripts/bench_octax_single.py \
    --game tetris \
    --steps $STEPS \
    --num-seeds $NUM_SEEDS \
    --num-envs $NUM_ENVS \
    --config paper_256x1 \
    --use-wandb &
PID2=$!

echo "Started: brix=$PID1, tetris=$PID2"
wait $PID1 $PID2
echo "Done! Results in results/octax_single/"
