#!/bin/bash
# Octax overnight - 64x4 first (safe), then 256x4
# Total: ~4.5 hours

set -e
echo "Started at $(date)"

# 64x4 with norm ON (all 19 games)
for GAME in airplane blinky brix deep filter flight_runner missile pong rocket shooting_stars spacejam squash submarine tank tetris ufo vertical_brix wipe_off worm; do
  echo ">>> $GAME 64x4 norm=ON $(date)"
  uv run python scripts/bench_ppo_octax.py --game $GAME --steps 5000000 --seeds 2 --mlp 64x4 --unified --normalize-rewards --use-wandb
done

# 64x4 with norm OFF (5 key games)
for GAME in brix pong tetris tank blinky; do
  echo ">>> $GAME 64x4 norm=OFF $(date)"
  uv run python scripts/bench_ppo_octax.py --game $GAME --steps 5000000 --seeds 2 --mlp 64x4 --unified --use-wandb
done

echo "=== 64x4 DONE at $(date) - starting 256x4 ==="

# 256x4 with norm ON (all 19 games)
for GAME in airplane blinky brix deep filter flight_runner missile pong rocket shooting_stars spacejam squash submarine tank tetris ufo vertical_brix wipe_off worm; do
  echo ">>> $GAME 256x4 norm=ON $(date)"
  uv run python scripts/bench_ppo_octax.py --game $GAME --steps 5000000 --seeds 2 --mlp 256x4 --unified --normalize-rewards --use-wandb
done

# 256x4 with norm OFF (5 key games)
for GAME in brix pong tetris tank blinky; do
  echo ">>> $GAME 256x4 norm=OFF $(date)"
  uv run python scripts/bench_ppo_octax.py --game $GAME --steps 5000000 --seeds 2 --mlp 256x4 --unified --use-wandb
done

echo "=== ALL DONE at $(date) ==="
