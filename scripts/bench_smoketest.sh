#!/bin/bash
# Quick smoketest (~1 min)
uv run python scripts/throughput_benchmark.py --envs Breakout-MinAtar brax/halfcheetah --depths 2 4 --timesteps 1000000 --num-seeds 1
