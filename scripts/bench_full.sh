#!/bin/bash
# Full 10M benchmark (purejaxrl default)
uv run python scripts/throughput_benchmark.py --envs Breakout-MinAtar Asterix-MinAtar brax/halfcheetah Pendulum-v1 --depths 2 4 8 16 32 --timesteps 10000000 --num-seeds 3
