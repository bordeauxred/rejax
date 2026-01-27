#!/bin/bash
# Octax throughput test
#
# Usage:
#   ./scripts/octax_throughput.sh

set -e

# Prevent JAX from pre-allocating all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run python scripts/bench_octax_single.py --mode throughput --output-dir results/octax_derisk
