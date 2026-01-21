#!/bin/bash
# Full benchmark with wandb logging
uv run python scripts/throughput_benchmark.py --envs Breakout-MinAtar Asterix-MinAtar brax/halfcheetah --depths 2 4 8 16 32 --timesteps 10000000 --num-seeds 3 --use-wandb
