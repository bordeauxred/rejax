#!/bin/bash
# Fast OOM fix smoke test
#
# This test verifies that the cached eval fix works by running
# enough cycles and evaluations to trigger OOM in the old code.
#
# Expected behavior:
# - OLD CODE: OOM around game 15-19 (cycle 3-4)
# - FIXED CODE: Completes all 5 cycles successfully
#
# Should complete in ~3-5 minutes on GPU

set -e

echo "=== OOM Fix Smoke Test ==="
echo ""
echo "Parameters:"
echo "  5 cycles x 5 games = 25 trainings"
echo "  500k steps/game, eval_freq=100k -> 5 evals/game -> 125 total evals"
echo ""
echo "Old code: 125 eval compilations -> OOM around game 15-19"
echo "Fixed code: 5 eval compilations (1 per game) -> completes successfully"
echo ""

uv run python scripts/continual_pipeline.py \
  --configs pgx_baseline \
  --steps-per-game 500000 \
  --num-cycles 5 \
  --num-seeds 1 \
  --eval-freq 100000 \
  --checkpoint-dir checkpoints/oom_test \
  --output-dir results/oom_test

echo ""
echo "=== SUCCESS: No OOM! Fix works. ==="
echo ""
echo "The test completed all 5 cycles without running out of memory."
echo "The cached eval function is working correctly."
