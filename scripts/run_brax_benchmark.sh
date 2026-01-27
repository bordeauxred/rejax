#!/bin/bash
# Brax continual learning benchmark
# Requires: brax, mujoco (for MJX backend)
#
# Usage:
#   ./scripts/run_brax_benchmark.sh smoketest     # Quick test (spring)
#   ./scripts/run_brax_benchmark.sh single        # Single task baselines (mjx, parallel)
#   ./scripts/run_brax_benchmark.sh continual     # Full continual experiment (mjx)
#
# Note: MJX backend is slower (~4x) but produces correct physics.
#       Spring fails on walker2d and hopper.

set -e

MODE=${1:-smoketest}
BACKEND=${2:-mjx}  # mjx for correct physics, spring only for quick tests

case $MODE in
    smoketest)
        echo "=== Brax PPO Smoketest ==="
        echo "Backend: $BACKEND"
        echo "Quick test on halfcheetah (500k steps, 1 seed)"
        echo ""
        uv run python scripts/bench_brax_continual.py \
            --single-task halfcheetah \
            --steps-per-task 500000 \
            --num-seeds 1 \
            --num-envs 2048 \
            --backend $BACKEND \
            --configs baseline_256x4_minatar
        ;;

    single)
        echo "=== Single Task Baselines ==="
        echo "Backend: $BACKEND"
        echo ""
        echo "Configs (2 total, MinAtar high UTD):"
        echo "  baseline_64x4_minatar  - 64x4 network, 4×128=512 grad steps"
        echo "  baseline_256x4_minatar - 256x4 network, 4×128=512 grad steps"
        echo ""
        echo "Tasks: hopper, halfcheetah, walker2d, ant"
        echo "Steps: 15M per task"
        echo "Seeds: 3"
        echo "Reward scaling: 10 (Brax default)"
        echo ""
        echo "Running sequentially (45% GPU memory, other job using remaining)"
        echo ""

        MEM_FRAC=0.45
        CONFIGS="baseline_64x4_minatar baseline_256x4_minatar"
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_DIR="results/brax_single_${BACKEND}_${TIMESTAMP}"

        echo "Output directory: $OUTPUT_DIR"
        echo ""

        for task in hopper halfcheetah walker2d ant; do
            echo "=========================================="
            echo "Task: $task"
            echo "=========================================="
            XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRAC uv run python scripts/bench_brax_continual.py \
                --single-task $task \
                --steps-per-task 15000000 \
                --num-seeds 3 \
                --num-envs 2048 \
                --backend $BACKEND \
                --configs $CONFIGS \
                --output-dir $OUTPUT_DIR \
                --use-wandb
            echo ""
        done

        echo "=========================================="
        echo "All single task experiments complete!"
        echo "Check wandb for results."
        echo "=========================================="
        ;;

    continual)
        echo "=== Continual Learning Experiment ==="
        echo "Backend: $BACKEND"
        echo ""
        echo "Tasks: hopper → halfcheetah → walker2d → ant"
        echo "Cycles: 2"
        echo "Steps: 15M per task"
        echo "Seeds: 3"
        echo ""
        echo "Configs: baseline_256x4_minatar vs adamo_256x4"
        echo ""
        uv run python scripts/bench_brax_continual.py \
            --steps-per-task 15000000 \
            --num-cycles 2 \
            --num-seeds 3 \
            --num-envs 2048 \
            --backend $BACKEND \
            --configs baseline_256x4_minatar adamo_256x4 \
            --use-wandb
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Usage: $0 {smoketest|single|continual} [backend]"
        echo ""
        echo "Modes:"
        echo "  smoketest  - Quick 500k step test on halfcheetah"
        echo "  single     - Single task baselines (2 configs × 4 tasks × 3 seeds)"
        echo "  continual  - Continual learning: baseline vs AdaMo (2 cycles × 3 seeds)"
        echo ""
        echo "Backends:"
        echo "  mjx    - MuJoCo XLA, accurate physics (default, recommended)"
        echo "  spring - Fast but fails on walker2d/hopper"
        echo ""
        echo "Examples:"
        echo "  $0 single           # Run single task baselines with mjx"
        echo "  $0 single spring    # Quick test with spring backend"
        echo "  $0 continual        # Run continual learning experiment"
        exit 1
        ;;
esac
