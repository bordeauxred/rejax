#!/bin/bash
# PopArt Single-Task Benchmark
#
# Runs 7 games × 3 architectures = 21 jobs with 2 seeds each
# Parallelizes across jobs on single GPU (H100 40GB)
#
# Games: blinky, brix, pong, tank, flight_runner, tetris, worm
# Archs: 256x1, 64x4, 256x4
#
# Usage:
#   ./scripts/run_popart_single_task.sh                      # Full run (5M steps)
#   ./scripts/run_popart_single_task.sh 1000000              # Quick test (1M steps)
#   ./scripts/run_popart_single_task.sh 5000000 4            # 4 parallel jobs

set -e

# Quick test mode: just run one job (brix + 256x1)
if [ "$1" = "--test" ] || [ "$1" = "-t" ]; then
    echo "=== TEST MODE: Single job (brix + 256x1) ==="
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
    STEPS=${2:-1000000}
    NUM_ENVS=${3:-2048}
    mkdir -p results/popart_single_task/logs
    uv run python scripts/bench_octax_popart.py \
        --game brix --arch 256x1 \
        --steps $STEPS --num-envs $NUM_ENVS --num-seeds 2 \
        --output results/popart_single_task/brix_256x1.json \
        2>&1 | tee results/popart_single_task/logs/brix_256x1.log
    exit 0
fi

STEPS=${1:-5000000}
PARALLEL=${2:-3}  # Number of parallel jobs (3 fits comfortably on H100 40GB)
NUM_ENVS=${3:-2048}
NUM_SEEDS=${4:-2}
POPART_BETA=${5:-0.0001}

# Prevent JAX from pre-allocating all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

OUTPUT_DIR="results/popart_single_task"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

GAMES=(blinky brix pong tank flight_runner tetris worm)
ARCHS=(256x1 64x4 256x4)

echo "=============================================================="
echo "PopArt Single-Task Benchmark"
echo "=============================================================="
echo "Steps: $STEPS"
echo "Parallel jobs: $PARALLEL"
echo "Num envs: $NUM_ENVS"
echo "Seeds: $NUM_SEEDS"
echo "PopArt beta: $POPART_BETA"
echo "Output: $OUTPUT_DIR"
echo "=============================================================="
echo "Games (${#GAMES[@]}): ${GAMES[*]}"
echo "Archs (${#ARCHS[@]}): ${ARCHS[*]}"
echo "Total jobs: $((${#GAMES[@]} * ${#ARCHS[@]}))"
echo "=============================================================="

cd "$(dirname "$0")/.."
mkdir -p "$OUTPUT_DIR/logs"

echo "Started at: $(date)" | tee "$OUTPUT_DIR/run_${TIMESTAMP}.log"

# Build job list
JOBS=()
for game in "${GAMES[@]}"; do
    for arch in "${ARCHS[@]}"; do
        JOBS+=("${game}_${arch}")
    done
done

echo "Jobs: ${#JOBS[@]}"

# Function to run a single job
run_job() {
    local job=$1
    local game=${job%_*}
    local arch=${job#*_}

    echo "[$(date +%H:%M:%S)] Starting: $game + $arch"

    uv run python scripts/bench_octax_popart.py \
        --game "$game" \
        --arch "$arch" \
        --steps "$STEPS" \
        --num-envs "$NUM_ENVS" \
        --num-seeds "$NUM_SEEDS" \
        --popart-beta "$POPART_BETA" \
        --output "$OUTPUT_DIR/${game}_${arch}.json" \
        > "$OUTPUT_DIR/logs/${game}_${arch}.log" 2>&1

    local status=$?
    if [ $status -eq 0 ]; then
        # Extract final return from JSON
        local ret=$(python -c "import json; d=json.load(open('$OUTPUT_DIR/${game}_${arch}.json')); print(f\"{d['final_return_mean']:.1f}\")" 2>/dev/null || echo "?")
        echo "[$(date +%H:%M:%S)] Done: $game + $arch -> return=$ret"
    else
        echo "[$(date +%H:%M:%S)] FAILED: $game + $arch (see logs/${game}_${arch}.log)"
    fi
    return $status
}

# Run jobs with parallelism
PIDS=()
RUNNING_JOBS=()
FAILED=0
COMPLETED=0

for job in "${JOBS[@]}"; do
    # Wait if we have too many running jobs
    while [ ${#PIDS[@]} -ge $PARALLEL ]; do
        # Check for completed jobs
        NEW_PIDS=()
        NEW_JOBS=()
        for i in "${!PIDS[@]}"; do
            pid=${PIDS[$i]}
            rjob=${RUNNING_JOBS[$i]}
            if kill -0 $pid 2>/dev/null; then
                NEW_PIDS+=($pid)
                NEW_JOBS+=("$rjob")
            else
                wait $pid || FAILED=$((FAILED + 1))
                COMPLETED=$((COMPLETED + 1))
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
        RUNNING_JOBS=("${NEW_JOBS[@]}")

        if [ ${#PIDS[@]} -ge $PARALLEL ]; then
            sleep 2
        fi
    done

    # Launch new job
    run_job "$job" &
    PIDS+=($!)
    RUNNING_JOBS+=("$job")
    sleep 3  # Stagger launches to reduce memory fragmentation
done

# Wait for remaining jobs
echo ""
echo "Waiting for ${#PIDS[@]} remaining jobs..."
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    rjob=${RUNNING_JOBS[$i]}
    if wait $pid; then
        COMPLETED=$((COMPLETED + 1))
    else
        echo "Job $rjob (PID $pid) FAILED"
        FAILED=$((FAILED + 1))
        COMPLETED=$((COMPLETED + 1))
    fi
done

echo ""
echo "=============================================================="
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run_${TIMESTAMP}.log"
echo "Completed: $COMPLETED/${#JOBS[@]}"
echo "Failed: $FAILED"
echo "Results: $OUTPUT_DIR/*.json"
echo "=============================================================="

# Print summary table
echo ""
echo "Summary (Final Returns):"
echo "--------------------------------------------------------------"
printf "%-15s %10s %10s %10s\n" "Game" "256x1" "64x4" "256x4"
echo "--------------------------------------------------------------"
for game in "${GAMES[@]}"; do
    printf "%-15s" "$game"
    for arch in "${ARCHS[@]}"; do
        if [ -f "$OUTPUT_DIR/${game}_${arch}.json" ]; then
            ret=$(python -c "import json; d=json.load(open('$OUTPUT_DIR/${game}_${arch}.json')); print(f\"{d['final_return_mean']:.1f}\")" 2>/dev/null || echo "ERR")
            printf " %10s" "$ret"
        else
            printf " %10s" "-"
        fi
    done
    echo ""
done
echo "--------------------------------------------------------------"

# Plot if matplotlib available
if python -c "import matplotlib" 2>/dev/null; then
    echo ""
    echo "Generating plots..."
    python scripts/plot_popart_results.py "$OUTPUT_DIR" --output "figures/popart_${TIMESTAMP}"
fi

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "WARNING: $FAILED jobs failed. Check logs in $OUTPUT_DIR/logs/"
    exit 1
fi
