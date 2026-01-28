#!/bin/bash
# Quick test to verify setup before overnight run
# Time: ~2 minutes

set -e

echo "=== Quick Test: Verify PPOOctax + wandb ==="

uv run python scripts/bench_ppo_octax.py \
    --game brix \
    --steps 50000 \
    --seeds 1 \
    --mlp 64x4 \
    --unified \
    --normalize-rewards \
    --use-wandb

echo ""
echo "If you see a return value and wandb logged, everything works!"
