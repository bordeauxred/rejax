#!/bin/bash
# Octax Continual Learning: 4 Parallel (1 seed per config)
#
# Network:      64-64-64-64 MLP (shared CNN backbone)
# Steps/game:   5M
# Cycles:       3
# Methods:      baseline, adamo_groupsort, adamo_relu, adamo_groupsort_norm
# Seeds:        1 per config (seed 0), then run again with seed 1
# Games:        8 (brix, submarine, filter, tank, blinky, missile, ufo, wipe_off)
#
# Usage:
#   ./scripts/run_octax_continual_4parallel.sh                      # Full run (seed 0)
#   ./scripts/run_octax_continual_4parallel.sh 5000000 3 250000 0   # Seed 0
#   ./scripts/run_octax_continual_4parallel.sh 5000000 3 250000 1   # Seed 1

set -e

STEPS_PER_TASK=${1:-5000000}
NUM_CYCLES=${2:-3}
EVAL_FREQ=${3:-250000}
SEED=${4:-0}
NUM_ENVS=${5:-512}

# 4 processes on H100 (80GB): ~18GB each
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.22
export PYTHONUNBUFFERED=1

OUTPUT_DIR="results/octax_continual_v2"
WANDB_PROJECT="octax-continual-v2"

echo "=============================================================="
echo "OCTAX CONTINUAL LEARNING: 4 Parallel (seed $SEED)"
echo "=============================================================="
echo "Steps per task: $STEPS_PER_TASK"
echo "Cycles: $NUM_CYCLES"
echo "Seed: $SEED"
echo "Envs: $NUM_ENVS"
echo "Eval freq: $EVAL_FREQ"
echo "Output: $OUTPUT_DIR"
echo "W&B Project: $WANDB_PROJECT"
echo "=============================================================="
echo "Configs (4 parallel):"
echo "  1. baseline - ReLU, no ortho"
echo "  2. adamo_groupsort - GroupSort + ortho optimizer"
echo "  3. adamo_relu - ReLU + ortho optimizer (ablation)"
echo "  4. adamo_groupsort_norm - GroupSort + ortho + reward norm"
echo "=============================================================="

cd "$(dirname "$0")/.."

mkdir -p "$OUTPUT_DIR"

echo "Started at: $(date)" | tee -a "$OUTPUT_DIR/run_seed${SEED}.log"

CONFIGS=("baseline" "adamo_groupsort" "adamo_relu" "adamo_groupsort_norm")
PIDS=()
LABELS=()

for CONFIG in "${CONFIGS[@]}"; do
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
    sleep 3
done

echo ""
echo "Launched ${#PIDS[@]} processes: ${PIDS[*]}"
echo "Logs: $OUTPUT_DIR/*.log"
echo ""
echo "Monitor with:"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f $OUTPUT_DIR/baseline_seed${SEED}.log"
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
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run_seed${SEED}.log"
echo "Results: $OUTPUT_DIR/"
if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED runs failed. Check logs."
    exit 1
else
    echo "SUCCESS! Run seed 1 with:"
    echo "  ./scripts/run_octax_continual_4parallel.sh $STEPS_PER_TASK $NUM_CYCLES $EVAL_FREQ 1"
fi
