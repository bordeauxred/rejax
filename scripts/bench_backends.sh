#!/bin/bash
# Compare brax backends: mjx vs spring

echo "=== BRAX BACKEND COMPARISON ==="
echo ""

for backend in mjx spring; do
    echo "========================================"
    echo "Backend: $backend"
    echo "========================================"
    uv run python scripts/throughput_benchmark.py \
        --envs brax/halfcheetah \
        --depths 2 4 \
        --timesteps 1000000 \
        --num-seeds 3 \
        --num-envs 2048 \
        --eval-freq 100000 \
        --brax-backend $backend
    echo ""
done

echo "=== COMPARISON COMPLETE ==="
