#!/bin/bash
# Smoke test: Launch 8 parallel instances to check memory usage
#
# With XLA_PYTHON_CLIENT_PREALLOCATE=false, nvidia-smi shows ACTUAL usage
#
# Usage:
#   ./scripts/benchmark_continual_minatar_paper_64x4_smoke.sh      # 8 parallel
#   ./scripts/benchmark_continual_minatar_paper_64x4_smoke.sh 4    # 4 parallel

set -e

NUM_PARALLEL=${1:-8}
STEPS_PER_GAME=500000    # Short test
NUM_CYCLES=1
NUM_ENVS=2048
EVAL_FREQ=250000

# Prevent JAX from pre-allocating all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_NAME="smoke_test_parallel_${NUM_PARALLEL}"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"

echo "=============================================================="
echo "SMOKE TEST: ${NUM_PARALLEL} parallel instances"
echo "=============================================================="
echo "Steps per game: $STEPS_PER_GAME"
echo "Cycles: $NUM_CYCLES"
echo "Each instance: 1 seed"
echo ""
echo "Watch memory with: watch -n 1 nvidia-smi"
echo "=============================================================="

cd "$(dirname "$0")/.."
mkdir -p "$OUTPUT_DIR"

# Seeds 26-33 for balanced first-game distribution
SEEDS=(26 27 28 29 30 31 32 33)

# Launch instances in parallel
PIDS=()
for i in $(seq 0 $((NUM_PARALLEL - 1))); do
    SEED=${SEEDS[$i]}
    echo "Launching seed $SEED..."

    uv run python scripts/bench_continual.py \
        --steps-per-game $STEPS_PER_GAME \
        --num-cycles $NUM_CYCLES \
        --num-seeds 1 \
        --seed $SEED \
        --num-envs $NUM_ENVS \
        --eval-freq $EVAL_FREQ \
        --configs mlp_baseline_small mlp_adamo_small \
        --permute-channels \
        --random-game-order \
        --exclude-games Seaquest-MinAtar \
        --output-dir "$OUTPUT_DIR/seed_$SEED" \
        > "$OUTPUT_DIR/seed_${SEED}.log" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} processes: ${PIDS[*]}"
echo ""
echo "Waiting 30s for compilation, then checking memory..."
sleep 30

nvidia-smi

echo ""
echo "Waiting for all processes to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "  PID $pid completed"
done

echo ""
echo "=============================================================="
echo "Smoke test complete!"
echo "Check logs in: $OUTPUT_DIR/"
echo "=============================================================="
