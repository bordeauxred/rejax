#!/bin/bash
# Comprehensive Octax performance diagnostics
#
# Replicates the paper's setup to validate:
# 1. Raw env throughput (~350k steps/s at 8192 envs)
# 2. PPO training return (~21 on Brix at 5M steps)
#
# Usage:
#   ./scripts/diagnose_octax_perf.sh

set -e

export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "============================================================"
echo "OCTAX PERFORMANCE DIAGNOSTICS"
echo "============================================================"
echo "Replicating paper (arXiv 2510.01764) settings"
echo "============================================================"
echo ""

uv run python << 'PYEOF'
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import time
import numpy as np

print(f"JAX devices: {jax.devices()}")
print("")

# =============================================================================
# TEST 1: Raw Env Throughput (Paper's measure_time.py approach)
# =============================================================================
print("=" * 60)
print("TEST 1: RAW ENV THROUGHPUT")
print("=" * 60)
print("Paper claims ~350k steps/s at 8192 envs on RTX 3090")
print("")

from octax.environments import create_environment

def measure_raw_env_throughput(game="brix", num_envs=2048, num_steps=10000, warmup=3, repeats=5):
    """Raw env.step() with fixed action (no network)."""
    env, metadata = create_environment(game)

    @jax.jit
    def rollout(rng, states, obs):
        def step_fn(carry, _):
            rng, states, obs = carry
            action = jnp.zeros(num_envs, dtype=jnp.int32)
            next_states, next_obs, rewards, terminated, truncated, info = jax.vmap(
                lambda s, a: env.step(s, a)
            )(states, action)
            return (rng, next_states, next_obs), None
        return jax.lax.scan(step_fn, (rng, states, obs), length=num_steps)

    rng = jax.random.PRNGKey(0)
    reset_rngs = jax.random.split(rng, num_envs)
    states, obs, _ = jax.vmap(env.reset)(reset_rngs)

    # Compile
    print(f"  Compiling for {num_envs} envs...")
    start = time.perf_counter()
    _ = jax.block_until_ready(rollout(rng, states, obs))
    print(f"  Compile time: {time.perf_counter() - start:.2f}s")

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
    mean_time = np.mean(times)
    std_time = np.std(times)
    sps = total_steps / mean_time

    return sps, mean_time, std_time

print(f"{'Envs':>8} {'Steps':>8} {'Steps/sec':>12} {'Time':>10}")
print("-" * 45)

for num_envs in [512, 2048, 8192]:
    try:
        steps = 10000 if num_envs >= 2048 else 5000
        sps, mean_t, std_t = measure_raw_env_throughput("brix", num_envs, steps)
        print(f"{num_envs:>8} {steps:>8} {sps:>12,.0f} {mean_t:>10.3f}s")
    except Exception as e:
        print(f"{num_envs:>8} FAILED: {e}")

print("")

# =============================================================================
# TEST 2: Octax's Own PPO (Paper's implementation)
# =============================================================================
print("=" * 60)
print("TEST 2: OCTAX'S OWN PPO IMPLEMENTATION")
print("=" * 60)
print("Using vendor/octax/octax/agents/ppo.py")
print("Paper settings: hidden_layer_sizes=(256,), num_epochs=8")
print("")

from octax.wrappers import OctaxGymnaxWrapper
from octax.environments import create_environment

# Import the paper's PPO
import sys
sys.path.insert(0, "vendor/octax")
from octax.agents.ppo import PPOOctax

def train_with_octax_ppo(game="brix", total_steps=500_000, num_envs=512):
    """Train using Octax's own PPO implementation."""
    octax_env, metadata = create_environment(game)
    env = OctaxGymnaxWrapper(octax_env)
    env_params = env.default_params

    # Paper's config (from PPOOctax defaults and paper)
    config = {
        "env": env,
        "env_params": env_params,
        "num_envs": num_envs,
        "num_steps": 32,
        "num_epochs": 8,  # Paper default
        "num_minibatches": 32,
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": total_steps,
        "eval_freq": total_steps,  # Eval at end only
        "skip_initial_evaluation": True,
        "agent_kwargs": {
            "hidden_layer_sizes": (256,),  # Paper default: SINGLE 256-unit layer
            "activation": "relu",
        },
    }

    ppo = PPOOctax.create(**config)
    rng = jax.random.PRNGKey(0)

    # Train
    print(f"  Training {game} for {total_steps:,} steps...")
    print(f"  Config: num_envs={num_envs}, hidden_layers=(256,), num_epochs=8")

    start = time.perf_counter()
    ts, (_, returns) = PPOOctax.train(ppo, rng)
    jax.block_until_ready(ts)
    train_time = time.perf_counter() - start

    # Extract final return
    if returns.ndim == 2:
        final_return = float(returns[-1].mean())
    else:
        final_return = float(returns.mean())

    steps_per_sec = total_steps / train_time

    return final_return, steps_per_sec, train_time

print("Training with Octax's PPO (paper's implementation)...")
print("")

try:
    # Quick test first
    ret, sps, t = train_with_octax_ppo("brix", total_steps=100_000, num_envs=512)
    print(f"  100k steps: return={ret:.1f}, {sps:,.0f} steps/s, {t:.1f}s")

    # Longer test
    ret, sps, t = train_with_octax_ppo("brix", total_steps=500_000, num_envs=512)
    print(f"  500k steps: return={ret:.1f}, {sps:,.0f} steps/s, {t:.1f}s")

    print("")
    print(f"Paper claims ~21 return on Brix at 5M steps")
    print(f"Extrapolated: need 10x more training to reach paper's return")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

print("")

# =============================================================================
# TEST 3: Our PPO with Paper Config
# =============================================================================
print("=" * 60)
print("TEST 3: OUR PPO WITH PAPER CONFIG")
print("=" * 60)
print("Rejax PPO with OctaxCNN, paper's hyperparameters")
print("")

from rejax import PPO
from rejax.compat.octax2gymnax import create_octax

def train_with_rejax_ppo(game="brix", total_steps=500_000, num_envs=512, mlp_sizes=(256,)):
    """Train using Rejax PPO with paper's config."""
    env, env_params = create_octax(game)

    # Unified action wrapper for continual learning compatibility
    from scripts.bench_octax_single import UnifiedOctaxEnv, OCTAX_GAMES
    unified_env = UnifiedOctaxEnv(env, game, OCTAX_GAMES[game]["actions"])

    config = {
        "env": unified_env,
        "env_params": env_params,
        "num_envs": num_envs,
        "num_steps": 32,
        "num_epochs": 8,  # Paper's setting
        "num_minibatches": 32,
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": total_steps,
        "eval_freq": total_steps,
        "skip_initial_evaluation": True,
        "discrete_cnn_type": "octax",
        "agent_kwargs": {
            "mlp_hidden_sizes": mlp_sizes,
            "activation": "relu",
            "use_bias": True,
            "use_orthogonal_init": True,
        },
    }

    ppo = PPO.create(**config)
    rng = jax.random.PRNGKey(0)

    print(f"  Training {game} for {total_steps:,} steps...")
    print(f"  Config: num_envs={num_envs}, mlp_sizes={mlp_sizes}, num_epochs=8")

    start = time.perf_counter()
    ts, (_, returns) = PPO.train(ppo, rng)
    jax.block_until_ready(ts)
    train_time = time.perf_counter() - start

    if returns.ndim == 2:
        final_return = float(returns[-1].mean())
    else:
        final_return = float(returns.mean())

    steps_per_sec = total_steps / train_time

    return final_return, steps_per_sec, train_time

print("A) Single-layer MLP (256,) - Paper's architecture")
try:
    ret, sps, t = train_with_rejax_ppo("brix", 500_000, 512, mlp_sizes=(256,))
    print(f"   Return: {ret:.1f}, {sps:,.0f} steps/s, {t:.1f}s")
except Exception as e:
    print(f"   FAILED: {e}")

print("")
print("B) Four-layer MLP (256,256,256,256) - Our original")
try:
    ret, sps, t = train_with_rejax_ppo("brix", 500_000, 512, mlp_sizes=(256, 256, 256, 256))
    print(f"   Return: {ret:.1f}, {sps:,.0f} steps/s, {t:.1f}s")
except Exception as e:
    print(f"   FAILED: {e}")

print("")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Expected from paper (arXiv 2510.01764):
- Raw env throughput: ~350k steps/s at 8192 envs (RTX 3090)
- Brix return: ~21 at 5M steps
- Hidden layers: (256,) - SINGLE 256-unit layer
- num_epochs: 8
- num_envs: 512 (scales to 8192)

If our results don't match:
1. Check GPU utilization (nvidia-smi)
2. Verify JIT compilation (no Python overhead)
3. Compare network architectures carefully
""")
PYEOF

echo ""
echo "Diagnostics complete."
