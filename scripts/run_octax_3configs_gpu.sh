#!/bin/bash
# Run 3 configs on 2 games
# Configs: paper_256x1, 64x4, 256x4
# Games: brix, tetris
#
# Usage: ./scripts/run_octax_3configs_gpu.sh [steps] [envs]

set -e

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

STEPS=${1:-5000000}
NUM_ENVS=${2:-4096}

echo "=============================================="
echo "Octax 3-Config Benchmark (H100)"
echo "=============================================="
echo "Steps: $STEPS | Envs: $NUM_ENVS | Seeds: 1"
echo "Configs: paper_256x1, 64x4, 256x4"
echo "Games: brix, tetris"
echo "=============================================="

# Brix - 3 configs in parallel
echo ""
echo "=== BRIX (3 configs) ==="
uv run python scripts/bench_octax_single.py \
    --game brix --steps $STEPS --num-seeds 1 --num-envs $NUM_ENVS \
    --config paper_256x1 --use-wandb &

uv run python scripts/bench_octax_single.py \
    --game brix --steps $STEPS --num-seeds 1 --num-envs $NUM_ENVS \
    --config 64x4 --use-wandb &

uv run python scripts/bench_octax_single.py \
    --game brix --steps $STEPS --num-seeds 1 --num-envs $NUM_ENVS \
    --config 256x4 --use-wandb &

wait
echo "Brix done!"

# Tetris - 3 configs in parallel
echo ""
echo "=== TETRIS (3 configs) ==="
uv run python scripts/bench_octax_single.py \
    --game tetris --steps $STEPS --num-seeds 1 --num-envs $NUM_ENVS \
    --config paper_256x1 --use-wandb &

uv run python scripts/bench_octax_single.py \
    --game tetris --steps $STEPS --num-seeds 1 --num-envs $NUM_ENVS \
    --config 64x4 --use-wandb &

uv run python scripts/bench_octax_single.py \
    --game tetris --steps $STEPS --num-seeds 1 --num-envs $NUM_ENVS \
    --config 256x4 --use-wandb &

wait
echo "Tetris done!"

echo ""
echo "=============================================="
echo "All done! Results in results/octax_single/"
echo "=============================================="
