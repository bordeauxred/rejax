#!/bin/bash
# Full benchmark script for strong PPO baseline
# Run on GPU for full MinAtar benchmark

set -e

cd "$(dirname "$0")/.."

echo "=============================================="
echo "Full MinAtar Benchmark - Strong PPO Baseline"
echo "=============================================="
echo ""
echo "This will train PPO on all 5 MinAtar games"
echo "- 10M timesteps per game"
echo "- 5 seeds per game"
echo "- ~30-60 min total on modern GPU"
echo ""

# Parse args
USE_WANDB=""
PADDED=""
TIMESTEPS=10000000

while [[ $# -gt 0 ]]; do
    case $1 in
        --use-wandb)
            USE_WANDB="--use-wandb"
            shift
            ;;
        --padded)
            PADDED="--padded"
            shift
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Settings:"
echo "  Timesteps: $TIMESTEPS"
echo "  Padded: ${PADDED:-no}"
echo "  W&B: ${USE_WANDB:-no}"
echo ""

python scripts/bench_single_strong.py \
    --timesteps "$TIMESTEPS" \
    --num-seeds 5 \
    --eval-freq 100000 \
    --plot \
    $PADDED \
    $USE_WANDB

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "=============================================="
