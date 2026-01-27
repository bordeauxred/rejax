#!/bin/bash
# Octax Single-Task Benchmark with Parallel Seeds
#
# Runs 5 representative games × 2 MLP configs × 2 reward modes = 20 combinations
# Each combination runs 3 seeds in parallel
#
# Memory: ~3GB per process, runs 16 parallel (fills ~48GB)
#
# Usage:
#   ./scripts/bench_octax_single_parallel.sh                    # Full run (5M steps)
#   ./scripts/bench_octax_single_parallel.sh 1000000            # Quick test (1M steps)
#   ./scripts/bench_octax_single_parallel.sh 100000 throughput  # Throughput only

set -e

STEPS=${1:-5000000}
MODE=${2:-full}  # full, throughput

# Prevent JAX from pre-allocating all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_NAME="octax_single_task"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"
WANDB_PROJECT="rejax-octax-single"

echo "=============================================================="
echo "OCTAX SINGLE-TASK BENCHMARK"
echo "=============================================================="
echo "Steps per game: $STEPS"
echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR"
echo "=============================================================="

cd "$(dirname "$0")/.."
mkdir -p "$OUTPUT_DIR"

# ============================================================
# PHASE 1: Throughput Test
# ============================================================
if [ "$MODE" = "throughput" ] || [ "$MODE" = "full" ]; then
    echo ""
    echo "=== PHASE 1: Throughput Test ==="
    echo ""

    uv run python scripts/bench_octax_single.py \
        --mode throughput \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/throughput.log"

    if [ "$MODE" = "throughput" ]; then
        echo "Throughput test complete."
        exit 0
    fi
fi

# ============================================================
# PHASE 2: Single-Task Evaluation (Parallel Seeds)
# ============================================================
echo ""
echo "=== PHASE 2: Single-Task Evaluation ==="
echo "Games: tetris, brix, tank, spacejam, deep"
echo "Configs: 64x4, 256x4"
echo "Reward modes: raw, normalized"
echo "Seeds per combination: 3 (parallel)"
echo ""

GAMES=(tetris brix tank spacejam deep)
CONFIGS=(64x4 256x4)
REWARD_MODES=(raw normalized)
SEEDS=(0 1 2)

# Track all background processes
declare -A PIDS
RUNNING=0
MAX_PARALLEL=16  # ~48GB total

run_job() {
    local game=$1
    local config=$2
    local reward_mode=$3
    local seed=$4

    local norm_flag=""
    if [ "$reward_mode" = "normalized" ]; then
        norm_flag="--normalize-rewards"
    fi

    local job_name="${game}_${config}_${reward_mode}_s${seed}"
    local log_file="$OUTPUT_DIR/${job_name}.log"

    echo "  Launching: $job_name"

    uv run python -c "
import sys
sys.path.insert(0, 'scripts')
from bench_octax_single import run_single_task, EXPERIMENT_CONFIGS, create_unified_env, UNIFIED_ACTIONS, create_ppo_config
from rejax import PPO
import jax
import json
import time

game = '$game'
config_name = '$config'
seed = $seed
steps = $STEPS
num_envs = 2048
normalize_rewards = '$reward_mode' == 'normalized'

config = EXPERIMENT_CONFIGS[config_name]
mlp_hidden_sizes = config['mlp_hidden_sizes']

print(f'Game: {game}, Config: {config_name}, Seed: {seed}, Normalize: {normalize_rewards}')

env, params = create_unified_env(game)

ppo_config = {
    'env': env,
    'env_params': params,
    'agent_kwargs': {
        'mlp_hidden_sizes': mlp_hidden_sizes,
        'activation': 'relu',
        'use_bias': True,
        'use_orthogonal_init': True,
    },
    'num_envs': num_envs,
    'num_steps': 32,
    'num_epochs': 4,
    'num_minibatches': 32,
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_eps': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'total_timesteps': steps,
    'eval_freq': max(steps // 20, 10000),
    'normalize_observations': False,
    'normalize_rewards': normalize_rewards,
    'skip_initial_evaluation': True,
    'discrete_cnn_type': 'octax',
}

ppo = PPO.create(**ppo_config)

# Compile
print('Compiling...', flush=True)
rng = jax.random.PRNGKey(seed)
compile_start = time.time()
ts, _ = PPO.train(ppo, rng)
jax.block_until_ready(ts)
compile_time = time.time() - compile_start
print(f'Compiled in {compile_time:.1f}s', flush=True)

# Train
print('Training...', flush=True)
start = time.time()
ts, eval_data = PPO.train(ppo, rng)
jax.block_until_ready(ts)
runtime = time.time() - start

_, returns = eval_data
if returns.ndim == 2:
    final_return = float(returns[-1].mean())
else:
    final_return = float(returns.mean())

steps_per_sec = steps / runtime
print(f'Done: {steps_per_sec:,.0f} steps/s, return={final_return:.1f}', flush=True)

# Save result
result = {
    'game': game,
    'config': config_name,
    'seed': seed,
    'normalize_rewards': normalize_rewards,
    'steps': steps,
    'compile_time': compile_time,
    'runtime': runtime,
    'steps_per_sec': steps_per_sec,
    'final_return': final_return,
}
with open('$OUTPUT_DIR/${job_name}.json', 'w') as f:
    json.dump(result, f, indent=2)
" > "$log_file" 2>&1 &

    PIDS[$job_name]=$!
    RUNNING=$((RUNNING + 1))
}

wait_for_slot() {
    while [ $RUNNING -ge $MAX_PARALLEL ]; do
        for job_name in "${!PIDS[@]}"; do
            pid=${PIDS[$job_name]}
            if ! kill -0 $pid 2>/dev/null; then
                wait $pid || echo "  WARNING: $job_name failed"
                unset PIDS[$job_name]
                RUNNING=$((RUNNING - 1))
            fi
        done
        sleep 1
    done
}

echo "Started at: $(date)" | tee "$OUTPUT_DIR/run.log"
echo ""

# Launch all jobs with parallelism control
for game in "${GAMES[@]}"; do
    for config in "${CONFIGS[@]}"; do
        for reward_mode in "${REWARD_MODES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                wait_for_slot
                run_job "$game" "$config" "$reward_mode" "$seed"
                sleep 0.5  # Stagger launches
            done
        done
    done
done

# Wait for remaining jobs
echo ""
echo "Waiting for remaining jobs..."
for job_name in "${!PIDS[@]}"; do
    pid=${PIDS[$job_name]}
    if wait $pid; then
        echo "  $job_name completed"
    else
        echo "  $job_name FAILED"
    fi
done

echo ""
echo "=============================================================="
echo "Finished at: $(date)" | tee -a "$OUTPUT_DIR/run.log"
echo "Results: $OUTPUT_DIR/"
echo "=============================================================="

# Aggregate results
echo ""
echo "=== RESULTS SUMMARY ==="
uv run python -c "
import json
from pathlib import Path
import sys

output_dir = Path('$OUTPUT_DIR')
results = []

for f in output_dir.glob('*.json'):
    if f.name == 'throughput.json':
        continue
    try:
        with open(f) as fp:
            results.append(json.load(fp))
    except:
        pass

if not results:
    print('No results found')
    sys.exit(0)

# Group by game, config, reward_mode
from collections import defaultdict
grouped = defaultdict(list)
for r in results:
    key = (r['game'], r['config'], r.get('normalize_rewards', False))
    grouped[key].append(r['final_return'])

print(f\"{'Game':12} {'Config':8} {'Norm':6} {'Return':>15} {'N':>3}\")
print('-' * 50)
for (game, config, norm), returns in sorted(grouped.items()):
    import numpy as np
    mean = np.mean(returns)
    std = np.std(returns) if len(returns) > 1 else 0
    norm_str = 'yes' if norm else 'no'
    print(f'{game:12} {config:8} {norm_str:6} {mean:>7.1f} +/- {std:>4.1f} {len(returns):>3}')
"
