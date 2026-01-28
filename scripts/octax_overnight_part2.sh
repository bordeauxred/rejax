#!/bin/bash
# Part 2: 5 key games with reward normalization OFF
# Time: ~1 hour with 2 seeds
# 10 runs total

set -e

STEPS=5000000
SEEDS=2

GAMES="brix pong tetris tank blinky"
MLPS="64x4 256x4"

echo "=== Octax Part 2: Key games, norm OFF ==="
echo "Started at $(date)"

for GAME in $GAMES; do
    for MLP in $MLPS; do
        echo ""
        echo ">>> $GAME | MLP=$MLP | norm=OFF | $(date)"
        uv run python scripts/bench_ppo_octax.py \
            --game $GAME \
            --steps $STEPS \
            --seeds $SEEDS \
            --mlp $MLP \
            --unified \
            --use-wandb
    done
done

echo ""
echo "Part 2 done at $(date)"
