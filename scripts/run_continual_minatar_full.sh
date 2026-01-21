#!/bin/bash
# Full continual learning experiment on MinAtar
# Following Lyle et al. NaP methodology
#
# Compares:
#   - baseline: standard PPO with tanh activation
#   - ortho_adamo: PPO + orthogonal optimizer with groupsort
#   - ortho_adamo_lyle_lr: same + Lyle et al. linear LR decay
#
# Usage:
#   bash scripts/run_continual_full.sh

set -e

PROJECT_NAME="rejax-ppo-continual-minatar"

echo "=============================================="
echo "Continual Learning Experiment"
echo "Project: ${PROJECT_NAME}"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  - 10M steps per game"
echo "  - 5 games (Breakout, Asterix, SpaceInvaders, Freeway, Seaquest)"
echo "  - 2 cycles through all games"
echo "  - 3 seeds"
echo "  - 3 configs: baseline, ortho_adamo, ortho_adamo_lyle_lr"
echo "  - Network: 4 layers x 256 units"
echo ""
echo "Total: 100M steps per seed per config = 900M steps total"
echo ""

uv run python scripts/bench_continual.py \
    --steps-per-game 10000000 \
    --num-cycles 2 \
    --num-seeds 3 \
    --configs baseline ortho_adamo ortho_adamo_lyle_lr \
    --use-wandb \
    --wandb-project "${PROJECT_NAME}" \
    --eval-freq 500000 \
    --checkpoint-dir "checkpoints/${PROJECT_NAME}" \
    --output-dir "results/${PROJECT_NAME}"

echo ""
echo "=============================================="
echo "Experiment completed!"
echo "Results saved to: results/${PROJECT_NAME}"
echo "Checkpoints saved to: checkpoints/${PROJECT_NAME}"
echo "=============================================="
