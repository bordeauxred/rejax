#!/bin/bash
# Brax continual learning benchmark
# Requires: brax, mujoco (for MJX backend)
#
# Usage:
#   ./scripts/run_brax_benchmark.sh smoketest     # Quick test
#   ./scripts/run_brax_benchmark.sh compare       # Compare backends
#   ./scripts/run_brax_benchmark.sh baseline      # Find best baseline (3 configs, all tasks)
#   ./scripts/run_brax_benchmark.sh single        # Single task baselines
#   ./scripts/run_brax_benchmark.sh continual     # Full continual experiment

set -e

MODE=${1:-smoketest}
BACKEND=${2:-spring}  # spring is fast, mjx is more accurate

case $MODE in
    smoketest)
        echo "=== Brax PPO Smoketest ==="
        echo "Backend: $BACKEND"
        uv run python scripts/bench_brax_continual.py \
            --single-task halfcheetah \
            --steps-per-task 500000 \
            --num-seeds 1 \
            --num-envs 2048 \
            --backend $BACKEND \
            --configs baseline_256x4
        ;;

    compare)
        echo "=== Comparing Brax Backends ==="
        uv run python scripts/bench_brax_continual.py \
            --compare-backends \
            --steps-per-task 1000000 \
            --num-envs 2048
        ;;

    baseline)
        echo "=== Brax PPO Baseline Study ==="
        echo "Backend: $BACKEND"
        echo ""
        echo "Configs (4 total):"
        echo "  Brax paper (low UTD):   4×16=64 grad steps,  UTD≈0.003"
        echo "  MinAtar-like (high UTD): 4×128=512 grad steps, UTD≈0.025"
        echo "  × 2 network sizes: 64x4 and 256x4"
        echo ""
        echo "Tasks: hopper → halfcheetah → walker2d → ant (15M each)"
        echo "Seeds: 3"
        echo ""
        uv run python scripts/bench_brax_continual.py \
            --steps-per-task 15000000 \
            --num-cycles 1 \
            --num-seeds 3 \
            --num-envs 2048 \
            --backend $BACKEND \
            --configs baseline_64x4_brax baseline_256x4_brax baseline_64x4_minatar baseline_256x4_minatar \
            --use-wandb
        ;;

    single)
        echo "=== Single Task Baselines (UTD study) ==="
        echo "Backend: $BACKEND"
        for task in hopper halfcheetah walker2d ant; do
            echo "--- $task ---"
            uv run python scripts/bench_brax_continual.py \
                --single-task $task \
                --steps-per-task 15000000 \
                --num-seeds 3 \
                --num-envs 2048 \
                --backend $BACKEND \
                --configs baseline_256x4_utd_low baseline_256x4_utd_med baseline_256x4_utd_high
        done
        ;;

    continual)
        echo "=== Continual Learning Experiment ==="
        echo "Backend: $BACKEND"
        uv run python scripts/bench_brax_continual.py \
            --steps-per-task 15000000 \
            --num-cycles 2 \
            --num-seeds 3 \
            --num-envs 2048 \
            --backend $BACKEND \
            --configs baseline_256x4 adamo_256x4 \
            --use-wandb
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 {smoketest|compare|baseline|single|continual} [backend]"
        echo ""
        echo "Modes:"
        echo "  smoketest  - Quick 500k step test on halfcheetah"
        echo "  compare    - Compare mjx vs spring vs generalized backends"
        echo "  baseline   - Find best baseline (64x4, 128x4, 256x4) on all tasks"
        echo "  single     - Single task baselines (15M steps)"
        echo "  continual  - Full continual learning experiment"
        echo ""
        echo "Backends: spring (fast), mjx (accurate), generalized (deprecated)"
        exit 1
        ;;
esac
