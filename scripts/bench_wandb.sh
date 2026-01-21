#!/bin/bash
# Full benchmark with wandb logging
# Add brax/halfcheetah after: uv pip install brax mujoco
uv run python scripts/throughput_benchmark.py --envs Breakout-MinAtar Asterix-MinAtar Pendulum-v1 --depths 2 4 8 16 32 --timesteps 10000000 --num-seeds 3 --use-wandb
