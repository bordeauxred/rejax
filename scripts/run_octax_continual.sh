#!/bin/bash
# Octax Continual Learning Experiment
#
# Config: 8 games x 3 cycles, 64x4 MLP, 2 seeds
# Configs: baseline, adamo_groupsort, adamo_relu, adamo_groupsort_norm
# Estimated time: ~15 hours total (4 configs x ~3.7 hours each)
#
# Usage:
#   # Smoke test first (verifies pipeline on GPU)
#   ./scripts/run_octax_continual.sh smoke
#
#   # Full experiment (all 4 configs)
#   ./scripts/run_octax_continual.sh
#
#   # Single config only
#   ./scripts/run_octax_continual.sh baseline
#
#   # Plot existing results
#   ./scripts/run_octax_continual.sh plot

set -e

# Memory settings for JAX (45% of GPU RAM)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

OUTPUT_DIR="results/octax_continual"

if [ "$1" == "smoke" ]; then
    echo "Running smoke test..."
    uv run python scripts/bench_octax_continual.py --smoke-test --output-dir "$OUTPUT_DIR"
    echo ""
    echo "Smoke test passed! Run full experiment with:"
    echo "  ./scripts/run_octax_continual.sh"
    exit 0
fi

if [ "$1" == "plot" ]; then
    echo "Plotting results..."
    uv run python scripts/bench_octax_continual.py --plot-only --output-dir "$OUTPUT_DIR"
    exit 0
fi

# Check if specific config requested
if [ -n "$1" ]; then
    CONFIGS="$1"
else
    CONFIGS="baseline adamo_groupsort adamo_relu adamo_groupsort_norm"
fi

echo "======================================================================"
echo "Octax Continual Learning Experiment"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  Tasks: brix, submarine, filter, tank, blinky, missile, ufo, wipe_off"
echo "  Cycles: 3"
echo "  Steps per task: 5M"
echo "  MLP: 64x4"
echo "  Seeds: 2"
echo "  Configs: $CONFIGS"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Estimated time: ~3.7 hours per config"
echo ""

python scripts/bench_octax_continual.py \
    --steps-per-task 5000000 \
    --num-cycles 3 \
    --num-seeds 2 \
    --eval-freq 250000 \
    --num-envs 512 \
    --configs $CONFIGS \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Done! Results saved to: $OUTPUT_DIR"
echo "Plot with: ./scripts/run_octax_continual.sh plot"
