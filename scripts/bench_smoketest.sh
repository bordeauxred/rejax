#!/bin/bash
# Quick smoketest - compares with and without wandb

echo "=== WITHOUT WANDB ==="
uv run python scripts/throughput_benchmark.py --envs Breakout-MinAtar brax/halfcheetah --depths 2 4 8 --timesteps 1000000 --num-seeds 1

echo ""
echo "=== WITH WANDB ==="
uv run python scripts/throughput_benchmark.py --envs Breakout-MinAtar brax/halfcheetah --depths 2 4 --timesteps 1000000 --num-seeds 1 --use-wandb
