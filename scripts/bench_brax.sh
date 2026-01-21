#!/bin/bash
# Brax benchmark - requires: uv pip install brax mujoco
uv run python scripts/throughput_benchmark.py --envs brax/halfcheetah brax/ant brax/hopper --depths 2 4 8 16 --timesteps 10000000 --num-seeds 3 --brax-backend mjx
