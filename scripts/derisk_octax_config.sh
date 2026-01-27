#!/bin/bash
# De-risk Octax configuration: 64x4 vs 256x4, raw vs normalized rewards
#
# Runs 4 combinations on a representative game to pick the best config
# before running full continual learning experiments.
#
# Usage:
#   ./scripts/derisk_octax_config.sh              # Default: 500k steps, brix
#   ./scripts/derisk_octax_config.sh 1000000      # Custom steps
#   ./scripts/derisk_octax_config.sh 500000 tetris # Custom steps and game

set -e

# Prevent JAX from pre-allocating all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

STEPS=${1:-500000}
GAME=${2:-brix}
OUTPUT_DIR="results/octax_derisk"
NUM_SEEDS=1

echo "============================================================"
echo "OCTAX CONFIG DE-RISK"
echo "============================================================"
echo "Game: $GAME"
echo "Steps: $STEPS"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# Run all 4 combinations IN PARALLEL
echo ""
echo "Launching all 4 configs in parallel..."

echo ">>> [1/4] 64x4 raw rewards"
uv run python scripts/bench_octax_single.py \
  --mode single --game "$GAME" --steps "$STEPS" --num-seeds "$NUM_SEEDS" \
  --config 64x4 --output-dir "$OUTPUT_DIR" --use-wandb \
  > "$OUTPUT_DIR/64x4_raw.log" 2>&1 &
PID1=$!

echo ">>> [2/4] 64x4 normalized rewards"
uv run python scripts/bench_octax_single.py \
  --mode single --game "$GAME" --steps "$STEPS" --num-seeds "$NUM_SEEDS" \
  --config 64x4 --normalize-rewards --output-dir "$OUTPUT_DIR" --use-wandb \
  > "$OUTPUT_DIR/64x4_norm.log" 2>&1 &
PID2=$!

echo ">>> [3/4] 256x4 raw rewards"
uv run python scripts/bench_octax_single.py \
  --mode single --game "$GAME" --steps "$STEPS" --num-seeds "$NUM_SEEDS" \
  --config 256x4 --output-dir "$OUTPUT_DIR" --use-wandb \
  > "$OUTPUT_DIR/256x4_raw.log" 2>&1 &
PID3=$!

echo ">>> [4/4] 256x4 normalized rewards"
uv run python scripts/bench_octax_single.py \
  --mode single --game "$GAME" --steps "$STEPS" --num-seeds "$NUM_SEEDS" \
  --config 256x4 --normalize-rewards --output-dir "$OUTPUT_DIR" --use-wandb \
  > "$OUTPUT_DIR/256x4_norm.log" 2>&1 &
PID4=$!

echo ""
echo "All 4 jobs launched. PIDs: $PID1 $PID2 $PID3 $PID4"
echo "Logs: $OUTPUT_DIR/*.log"
echo ""
echo "Waiting for all jobs to complete..."

# Wait for all and track failures
FAILED=0
wait $PID1 || { echo "64x4 raw FAILED"; FAILED=1; }
wait $PID2 || { echo "64x4 norm FAILED"; FAILED=1; }
wait $PID3 || { echo "256x4 raw FAILED"; FAILED=1; }
wait $PID4 || { echo "256x4 norm FAILED"; FAILED=1; }

if [ $FAILED -eq 1 ]; then
  echo "Some jobs failed. Check logs in $OUTPUT_DIR/"
fi

# Summary
echo ""
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"

uv run python -c "
import json
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
results = []

for f in sorted(output_dir.glob('${GAME}_*.json')):
    with open(f) as fp:
        r = json.load(fp)
        norm_str = 'normalized' if r.get('normalize_rewards', False) else 'raw'
        results.append({
            'config': r['config'],
            'rewards': norm_str,
            'return': r['mean_return'],
            'steps_per_sec': r['steps_per_sec'],
        })

print(f\"{'Config':10} {'Rewards':12} {'Return':>10} {'Steps/sec':>12}\")
print('-' * 50)
for r in results:
    print(f\"{r['config']:10} {r['rewards']:12} {r['return']:>10.1f} {r['steps_per_sec']:>12,.0f}\")

# Best config
best = max(results, key=lambda x: x['return'])
print()
print(f\"Best config: {best['config']} {best['rewards']} (return={best['return']:.1f})\")
"

echo ""
echo "Results saved to: $OUTPUT_DIR"
