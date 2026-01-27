#!/bin/bash
# Run 2 Octax games in parallel for quick performance validation
# Uses paper_256x1 config (single 256-unit MLP, 8 epochs)

set -e
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

STEPS=${1:-5000000}
NUM_SEEDS=${2:-3}
NUM_ENVS=${3:-512}

echo "Running Brix and Pong in parallel ($STEPS steps, $NUM_SEEDS seeds, $NUM_ENVS envs)"

# Run both games in parallel
uv run python scripts/bench_octax_single.py --game brix --steps $STEPS --num-seeds $NUM_SEEDS --num-envs $NUM_ENVS --config paper_256x1 --use-wandb &
PID1=$!

uv run python scripts/bench_octax_single.py --game pong --steps $STEPS --num-seeds $NUM_SEEDS --num-envs $NUM_ENVS --config paper_256x1 --use-wandb &
PID2=$!

echo "Brix PID: $PID1, Pong PID: $PID2"
wait $PID1 $PID2
echo "Done!"
