#!/bin/bash
# Quick smoketest

uv run python scripts/throughput_benchmark.py \
    --envs Breakout-MinAtar brax/halfcheetah \
    --depths 2 4 \
    --timesteps 1000000 \
    --num-seeds 3 \
    --num-envs 2048 \
    --eval-freq 100000
