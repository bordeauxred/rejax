#!/bin/bash
# Brax continual learning benchmark
# Requires: brax, mujoco (for MJX backend)
#
# Usage:
#   ./scripts/run_brax_benchmark.sh smoketest     # Quick test
#   ./scripts/run_brax_benchmark.sh compare       # Compare backends
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
            --steps-per-task 100000 \
            --num-seeds 1 \
            --num-envs 512 \
            --backend $BACKEND
        ;;

    compare)
        echo "=== Comparing Brax Backends ==="
        uv run python scripts/bench_brax_continual.py \
            --compare-backends \
            --steps-per-task 500000 \
            --num-envs 1024
        ;;

    single)
        echo "=== Single Task Baselines ==="
        echo "Backend: $BACKEND"
        for task in hopper halfcheetah walker2d ant; do
            echo "--- $task ---"
            uv run python scripts/bench_brax_continual.py \
                --single-task $task \
                --steps-per-task 15000000 \
                --num-seeds 3 \
                --num-envs 2048 \
                --backend $BACKEND \
                --configs baseline baseline_deep
        done
        ;;

    continual)
        echo "=== Continual Learning Experiment ==="
        echo "Backend: $BACKEND"
        uv run python scripts/bench_brax_continual.py \
            --steps-per-task 5000000 \
            --num-cycles 2 \
            --num-seeds 3 \
            --num-envs 2048 \
            --backend $BACKEND \
            --configs baseline baseline_deep adamo \
            --use-wandb
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 {smoketest|compare|single|continual} [backend]"
        echo "Backends: spring (fast), mjx (accurate), generalized"
        exit 1
        ;;
esac
