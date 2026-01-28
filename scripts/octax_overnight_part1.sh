#!/bin/bash
# Part 1: All 19 games with reward normalization ON
# Time: ~3.5 hours with 2 seeds
# 38 runs total

set -e

STEPS=5000000
SEEDS=2

GAMES="airplane blinky brix deep filter flight_runner missile pong rocket shooting_stars spacejam squash submarine tank tetris ufo vertical_brix wipe_off worm"
MLPS="64x4 256x4"

echo "=== Octax Part 1: All games, norm ON ==="
echo "Started at $(date)"

for GAME in $GAMES; do
    for MLP in $MLPS; do
        echo ""
        echo ">>> $GAME | MLP=$MLP | norm=ON | $(date)"
        uv run python scripts/bench_ppo_octax.py \
            --game $GAME \
            --steps $STEPS \
            --seeds $SEEDS \
            --mlp $MLP \
            --unified \
            --normalize-rewards \
            --use-wandb
    done
done

echo ""
echo "Part 1 done at $(date)"
