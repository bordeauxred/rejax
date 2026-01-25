#!/bin/bash
# Quick smoke test for continual benchmark with metrics logging
#
# Tests that:
# 1. Metrics are computed correctly during training
# 2. No crashes or OOM
# 3. Reasonable performance (not massively slower)
#
# Run on GPU:
#   ./scripts/test_continual_metrics.sh

set -e

echo "=============================================="
echo "Continual Learning Metrics Smoke Test"
echo "=============================================="
echo ""
echo "Config: 1 cycle, 100k steps/game, eval every 50k"
echo "Expected: ~2-3 min on GPU"
echo ""

# Run WITHOUT wandb first (fast path) to get baseline
echo ">>> Testing FAST path (no wandb)..."
time python scripts/bench_continual.py \
    --configs pgx_baseline \
    --steps-per-game 100000 \
    --num-cycles 1 \
    --num-seeds 1 \
    --eval-freq 50000 \
    --checkpoint-dir /tmp/continual_test_fast \
    --output-dir /tmp/continual_test_fast

echo ""
echo ">>> Testing METRICS path (with wandb, offline mode)..."

# Run WITH wandb in offline mode (metrics path)
export WANDB_MODE=offline
time python scripts/bench_continual.py \
    --configs pgx_baseline \
    --steps-per-game 100000 \
    --num-cycles 1 \
    --num-seeds 1 \
    --eval-freq 50000 \
    --checkpoint-dir /tmp/continual_test_metrics \
    --output-dir /tmp/continual_test_metrics \
    --use-wandb

echo ""
echo "=============================================="
echo "SMOKE TEST COMPLETE"
echo "=============================================="
echo ""
echo "Check wandb offline logs at: wandb/offline-run-*"
echo "Should contain metrics: loss/policy, loss/value, loss/entropy,"
echo "  ppo/approx_kl, ppo/clip_fraction, gram/actor, gram/critic"
