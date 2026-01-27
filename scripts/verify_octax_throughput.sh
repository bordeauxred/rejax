#!/bin/bash
# Verify Octax throughput - RAW ENV vs FULL PPO
#
# This measures:
# 1. RAW ENV THROUGHPUT (no network) - should get ~350k steps/s
# 2. Our PPO throughput for comparison
#
# Usage:
#   ./scripts/verify_octax_throughput.sh

set -e

# Prevent JAX from pre-allocating all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

OUTPUT_DIR="results/octax_throughput_verify"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "OCTAX RAW ENV THROUGHPUT VERIFICATION"
echo "============================================================"
echo "Paper claims ~350k steps/s on RTX 3090 at 8192 envs"
echo "============================================================"
echo ""

# Inline Python script for raw env throughput (no dependencies)
uv run python << 'PYEOF'
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import time
from octax.environments import create_environment

def measure_env_throughput(game="brix", num_envs=2048, num_steps=1000, warmup=3, repeats=10):
    """Measure raw env.step() throughput (no network, fixed action)."""
    env, metadata = create_environment(game)

    @jax.jit
    def rollout(rng, states, obs):
        def env_step(carry, _):
            rng, states, obs = carry
            action = jnp.zeros(num_envs, dtype=jnp.int32)  # Fixed action
            next_states, next_obs, rewards, terminated, truncated, info = jax.vmap(
                lambda s, a: env.step(s, a)
            )(states, action)
            return (rng, next_states, next_obs), None
        return jax.lax.scan(env_step, (rng, states, obs), length=num_steps)

    # Setup
    rng = jax.random.PRNGKey(0)
    reset_rngs = jax.random.split(rng, num_envs)
    states, obs, _ = jax.vmap(env.reset)(reset_rngs)

    # Compile
    print(f"Compiling for {num_envs} envs...")
    start = time.perf_counter()
    _ = jax.block_until_ready(rollout(rng, states, obs))
    compile_time = time.perf_counter() - start
    print(f"Compile time: {compile_time:.2f}s")

    # Warmup
    for _ in range(warmup):
        jax.block_until_ready(rollout(rng, states, obs))

    # Benchmark
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(rollout(rng, states, obs))
        times.append(time.perf_counter() - start)

    total_steps = num_envs * num_steps
    mean_time = sum(times) / len(times)
    steps_per_sec = total_steps / mean_time

    return {
        "game": game,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "mean_time": mean_time,
        "steps_per_sec": steps_per_sec,
        "total_steps": total_steps,
    }

print("=" * 60)
print("RAW ENV THROUGHPUT (env.step only, no network)")
print("=" * 60)

results = []
for num_envs in [512, 2048, 8192]:
    # More steps for larger env counts to amortize overhead
    num_steps = 10000 if num_envs >= 2048 else 5000
    print(f"\n>>> {num_envs} parallel envs, {num_steps} steps")
    try:
        r = measure_env_throughput(game="brix", num_envs=num_envs, num_steps=num_steps)
        results.append(r)
        print(f"    Steps/sec: {r['steps_per_sec']:,.0f}")
    except Exception as e:
        print(f"    FAILED: {e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Envs':>8} {'Steps/sec':>15}")
print("-" * 25)
for r in results:
    print(f"{r['num_envs']:>8} {r['steps_per_sec']:>15,.0f}")

print("\nCompare to our full PPO: ~15k steps/s")
print("Gap is expected: PPO includes CNN forward/backward + updates")

# Now test with our wrappers to see if they add overhead
print("\n" + "=" * 60)
print("WRAPPER OVERHEAD TEST")
print("=" * 60)

from rejax.compat.octax2gymnax import create_octax, HWCObsWrapper

def measure_wrapped_throughput(game="brix", num_envs=2048, num_steps=5000, warmup=3, repeats=10):
    """Measure throughput with our HWCObsWrapper."""
    env, env_params = create_octax(game)

    @jax.jit
    def rollout(rng, states, obs):
        def env_step(carry, _):
            rng, states, obs = carry
            rng, step_rng = jax.random.split(rng)
            rngs = jax.random.split(step_rng, num_envs)
            action = jnp.zeros(num_envs, dtype=jnp.int32)
            # Use the wrapped step (with transpose)
            obs, states, rewards, dones, info = jax.vmap(
                lambda r, s, a: env.step(r, s, a, env_params)
            )(rngs, states, action)
            return (rng, states, obs), None
        return jax.lax.scan(env_step, (rng, states, obs), length=num_steps)

    # Setup
    rng = jax.random.PRNGKey(0)
    reset_rngs = jax.random.split(rng, num_envs)
    obs, states = jax.vmap(lambda r: env.reset(r, env_params))(reset_rngs)

    # Compile
    print(f"Compiling wrapped env for {num_envs} envs...")
    start = time.perf_counter()
    _ = jax.block_until_ready(rollout(rng, states, obs))
    compile_time = time.perf_counter() - start
    print(f"Compile time: {compile_time:.2f}s")

    # Warmup
    for _ in range(warmup):
        jax.block_until_ready(rollout(rng, states, obs))

    # Benchmark
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(rollout(rng, states, obs))
        times.append(time.perf_counter() - start)

    total_steps = num_envs * num_steps
    mean_time = sum(times) / len(times)
    steps_per_sec = total_steps / mean_time

    return steps_per_sec

print(f"\n>>> Testing with HWCObsWrapper (our transpose)")
try:
    wrapped_sps = measure_wrapped_throughput(game="brix", num_envs=2048, num_steps=5000)
    print(f"    Steps/sec with wrapper: {wrapped_sps:,.0f}")

    # Compare to raw
    raw_sps = results[-1]['steps_per_sec'] if results else 0  # Use 2048 env result
    if raw_sps > 0:
        overhead = (raw_sps - wrapped_sps) / raw_sps * 100
        print(f"    Overhead vs raw env: {overhead:.1f}%")
except Exception as e:
    print(f"    FAILED: {e}")
PYEOF
