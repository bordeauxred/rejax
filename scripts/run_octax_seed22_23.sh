#!/bin/bash
# Octax Continual: Seeds 22 and 23
set -e

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.22
export PYTHONUNBUFFERED=1

OUTPUT_DIR="results/octax_continual_v2"
WANDB_PROJECT="octax-continual-v2"
CONFIGS=("baseline" "adamo_groupsort" "adamo_relu" "adamo_groupsort_norm")

cd "$(dirname "$0")/.."
mkdir -p "$OUTPUT_DIR"

echo "Started at: $(date)" | tee "$OUTPUT_DIR/run_seed22_23.log"

for SEED in 22 23; do
    echo ""
    echo "=== SEED $SEED starting at $(date) ===" | tee -a "$OUTPUT_DIR/run_seed22_23.log"

    PIDS=()
    for CONFIG in "${CONFIGS[@]}"; do
        echo "Launching ${CONFIG}_seed${SEED}..."
        uv run python scripts/bench_octax_continual.py \
            --configs $CONFIG --num-seeds 1 --seed $SEED \
            --output-dir "$OUTPUT_DIR" \
            --wandb --wandb-project "$WANDB_PROJECT" \
            > "$OUTPUT_DIR/${CONFIG}_seed${SEED}.log" 2>&1 &
        PIDS+=($!)
        sleep 2
    done

    echo "Waiting for seed $SEED (PIDs: ${PIDS[*]})..."
    for pid in "${PIDS[@]}"; do wait $pid; done
    echo "=== SEED $SEED done at $(date) ===" | tee -a "$OUTPUT_DIR/run_seed22_23.log"
done

echo ""
echo "ALL DONE at $(date)" | tee -a "$OUTPUT_DIR/run_seed22_23.log"
