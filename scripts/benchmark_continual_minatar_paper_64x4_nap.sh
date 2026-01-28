#!/bin/bash
# Paper Baseline: NaP (Normalize-and-Project) (64x4 MLP)
#
# Same setup as main paper experiment but with NaP baseline
#
# Network:      64-64-64-64 MLP
# Steps/game:   15M
# Cycles:       20
# Methods:      mlp_nap_small (NaP projection after each update)
# Seeds:        8 (26-33, same as main experiment)
# Games:        4 (Breakout, Asterix, SpaceInvaders, Freeway)
#
# Logs to same wandb project and output dir as main experiment for comparison
#
# Usage:
#   ./scripts/benchmark_continual_minatar_paper_64x4_nap.sh                    # Full run
#   ./scripts/benchmark_continual_minatar_paper_64x4_nap.sh 1000000 2 500000   # Quick test

set -e

STEPS_PER_GAME=${1:-15000000}
NUM_CYCLES=${2:-20}
EVAL_FREQ=${3:-500000}
NUM_ENVS=${4:-2048}

# Prevent JAX from pre-allocating all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Use platform allocator to reduce fragmentation (allocates contiguous blocks)
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Same output locations as main experiment for easy comparison
EXPERIMENT_NAME="paper_continual_64x4_permuted"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
WANDB_PROJECT="adamo_continual_paper"

echo "=============================================================="
echo "NAP BASELINE: 64x4 + PERMUTATIONS + 20 CYCLES"
echo "=============================================================="
echo "Steps per game: $STEPS_PER_GAME"
echo "Cycles: $NUM_CYCLES"
echo "Seeds: 8 (all parallel)"
echo "Envs: $NUM_ENVS"
echo "Eval freq: $EVAL_FREQ"
echo "Output: $OUTPUT_DIR"
echo "W&B Project: $WANDB_PROJECT"
echo "=============================================================="
echo "Configs:"
echo "  1. mlp_nap_small - NaP (Normalize-and-Project)"
echo "=============================================================="

cd "$(dirname "$0")/.."

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "Started at: $(date)" | tee -a "$OUTPUT_DIR/run_nap.log"

# Seeds 26-33 for balanced first-game distribution (same as main experiment)
SEEDS=(26 27 28 29 30 31 32 33)
PIDS=()

for SEED in "${SEEDS[@]}"; do
    echo "Launching seed $SEED..."

    uv run python scripts/bench_continual.py \
        --steps-per-game $STEPS_PER_GAME \
        --num-cycles $NUM_CYCLES \
        --num-seeds 1 \
        --seed $SEED \
        --num-envs $NUM_ENVS \
        --eval-freq $EVAL_FREQ \
        --configs mlp_nap_small \
        --permute-channels \
        --random-game-order \
        --exclude-games Seaquest-MinAtar \
        --checkpoint-dir "$CHECKPOINT_DIR/seed_$SEED" \
        --output-dir "$OUTPUT_DIR/seed_$SEED" \
        --use-wandb \
        --wandb-project "$WANDB_PROJECT" \
        > "$OUTPUT_DIR/seed_${SEED}_nap.log" 2>&1 &

    PIDS+=($!)
    sleep 5  # Stagger launches to reduce GPU memory fragmentation
done

echo ""
echo "Launched ${#PIDS[@]} processes: ${PIDS[*]}"
echo "Logs: $OUTPUT_DIR/seed_*_nap.log"
echo ""
echo "Monitor with:"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f $OUTPUT_DIR/seed_26_nap.log"
echo ""

# Wait for all processes
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    seed=${SEEDS[$i]}
    if wait $pid; then
        echo "Seed $seed (PID $pid) completed successfully"
    else
        echo "Seed $seed (PID $pid) FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=============================================================="
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run_nap.log"
echo "Results: $OUTPUT_DIR/"
if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED seeds failed. Check logs."
    exit 1
fi
