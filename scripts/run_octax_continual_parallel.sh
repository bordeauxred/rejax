#!/bin/bash
# Octax Continual Learning: Full AdaMo Experiment (All Configs Parallel)
#
# Runs PPOOctax with shared backbone on 8 Octax games in continual setting.
# Tests AdaMo variants vs baseline - all 8 runs in parallel on H100.
#
# Network:      64-64-64-64 MLP (shared CNN backbone)
# Steps/game:   5M
# Cycles:       3
# Methods:      baseline, adamo_groupsort, adamo_relu, adamo_groupsort_norm
# Seeds:        2 (0-1)
# Games:        8 (brix, submarine, filter, tank, blinky, missile, ufo, wipe_off)
#
# Usage:
#   ./scripts/run_octax_continual_parallel.sh                      # Full run
#   ./scripts/run_octax_continual_parallel.sh 1000000 1 250000     # Quick test

set -e

STEPS_PER_TASK=${1:-5000000}
NUM_CYCLES=${2:-3}
EVAL_FREQ=${3:-250000}
NUM_ENVS=${4:-512}

# 8 processes on H100 (80GB): ~10GB each
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export PYTHONUNBUFFERED=1

OUTPUT_DIR="results/octax_continual"
WANDB_PROJECT="octax-continual"

echo "=============================================================="
echo "OCTAX CONTINUAL LEARNING: Full AdaMo Experiment"
echo "=============================================================="
echo "Steps per task: $STEPS_PER_TASK"
echo "Cycles: $NUM_CYCLES"
echo "Seeds: 2 (per config)"
echo "Envs: $NUM_ENVS"
echo "Eval freq: $EVAL_FREQ"
echo "Output: $OUTPUT_DIR"
echo "W&B Project: $WANDB_PROJECT"
echo "=============================================================="
echo "Configs (4 × 2 seeds = 8 parallel processes):"
echo "  1. baseline - ReLU, no ortho"
echo "  2. adamo_groupsort - GroupSort + ortho optimizer"
echo "  3. adamo_relu - ReLU + ortho optimizer (ablation)"
echo "  4. adamo_groupsort_norm - GroupSort + ortho + reward norm"
echo "=============================================================="

cd "$(dirname "$0")/.."

mkdir -p "$OUTPUT_DIR"

echo "Started at: $(date)" | tee -a "$OUTPUT_DIR/run.log"

# All 4 configs × 2 seeds = 8 parallel processes
CONFIGS=("baseline" "adamo_groupsort" "adamo_relu" "adamo_groupsort_norm")
SEEDS=(0 1)
PIDS=()
LABELS=()

for CONFIG in "${CONFIGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LABEL="${CONFIG}_seed${SEED}"
        echo "Launching $LABEL..."

        uv run python scripts/bench_octax_continual.py \
            --steps-per-task $STEPS_PER_TASK \
            --num-cycles $NUM_CYCLES \
            --num-seeds 1 \
            --seed $SEED \
            --num-envs $NUM_ENVS \
            --eval-freq $EVAL_FREQ \
            --configs $CONFIG \
            --output-dir "$OUTPUT_DIR" \
            --wandb \
            --wandb-project "$WANDB_PROJECT" \
            > "$OUTPUT_DIR/${LABEL}.log" 2>&1 &

        PIDS+=($!)
        LABELS+=("$LABEL")
        sleep 2  # Stagger launches
    done
done

echo ""
echo "Launched ${#PIDS[@]} processes: ${PIDS[*]}"
echo "Logs: $OUTPUT_DIR/*.log"
echo ""
echo "Monitor with:"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f $OUTPUT_DIR/baseline_seed0.log"
echo ""

# Wait for all processes
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    label=${LABELS[$i]}
    if wait $pid; then
        echo "$label (PID $pid) completed successfully"
    else
        echo "$label (PID $pid) FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=============================================================="
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run.log"
echo "Results: $OUTPUT_DIR/"
if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED runs failed. Check logs."
    exit 1
fi
