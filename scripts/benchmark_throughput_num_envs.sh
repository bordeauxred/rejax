#!/bin/bash
# Quick throughput test: find optimal num_envs
#
# Tests different num_envs values and reports steps/sec
# Scales num_minibatches to keep minibatch_size ~ 2048
#
# num_envs | num_minibatches | minibatch_size
# 512      | 32              | 2048
# 1024     | 64              | 2048
# 2048     | 128             | 2048
# 4096     | 256             | 2048
# 8192     | 512             | 2048

set -e

export XLA_PYTHON_CLIENT_PREALLOCATE=false

OUTPUT_DIR="results/throughput_test"
mkdir -p "$OUTPUT_DIR"

echo "=============================================================="
echo "THROUGHPUT TEST: Finding optimal num_envs"
echo "(num_minibatches scaled to keep minibatch_size=2048)"
echo "=============================================================="

cd "$(dirname "$0")/.."

# Arrays for num_envs and corresponding num_minibatches
ENVS=(512 1024 2048 4096 8192)
MINIBATCHES=(32 64 128 256 512)

for i in "${!ENVS[@]}"; do
    NUM_ENVS=${ENVS[$i]}
    NUM_MB=${MINIBATCHES[$i]}

    echo ""
    echo "Testing num_envs=$NUM_ENVS, num_minibatches=$NUM_MB..."

    # Create a temporary config that overrides num_minibatches
    uv run python -c "
import sys
sys.path.insert(0, '.')
from scripts.bench_continual import *

# Override the config
config = dict(EXPERIMENT_CONFIGS_DICT['mlp_baseline_small'])
config['num_minibatches'] = $NUM_MB

env, env_params = create_minatar_env('Breakout-MinAtar')
ppo = create_ppo_for_game_with_config('Breakout-MinAtar', config, 1000000, $NUM_ENVS)

import time
rng = jax.random.PRNGKey(26)
ts = ppo.init_state(rng)

# Warmup
ts = ppo.train_iteration(ts)
jax.block_until_ready(ts)

# Benchmark
start = time.time()
for _ in range(10):
    ts = ppo.train_iteration(ts)
jax.block_until_ready(ts)
elapsed = time.time() - start

steps = 10 * $NUM_ENVS * ppo.num_steps
print(f'  {steps/elapsed:,.0f} steps/s')
" 2>&1 | tail -1

    echo "  Memory:"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
done

echo ""
echo "=============================================================="
echo "Done! Pick the num_envs with highest steps/s"
echo "=============================================================="
