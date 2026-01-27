"""
Test native Brax PPO speed (without gymnax wrapper).

This tests if the slowdown is due to our gymnax wrapper or a system issue.

Usage:
    uv run python scripts/test_native_brax.py
"""
import time
import jax

print("=" * 60)
print("Native Brax PPO Speed Test")
print("=" * 60)
print(f"JAX devices: {jax.devices()}")
print()

from brax.envs import create
from brax.training.agents.ppo import train as ppo_train

NUM_TIMESTEPS = 1_000_000
NUM_ENVS = 2048

for backend in ['spring', 'mjx']:
    print(f"\nBackend: {backend}")
    print("-" * 40)

    try:
        env = create('halfcheetah', backend=backend)
        print(f"  Env created: obs={env.observation_size}, act={env.action_size}")

        print(f"  Training {NUM_TIMESTEPS:,} steps with {NUM_ENVS} envs...")
        start = time.time()

        # Brax requires: batch_size * num_minibatches % num_envs == 0
        make_policy, params, metrics = ppo_train.train(
            environment=env,
            num_timesteps=NUM_TIMESTEPS,
            num_envs=NUM_ENVS,
            episode_length=1000,
            num_evals=2,
            reward_scaling=10,
            normalize_observations=True,
            batch_size=1024,
            num_minibatches=32,  # 1024 * 32 = 32768, % 2048 = 0
            unroll_length=10,
            num_updates_per_batch=4,
        )

        elapsed = time.time() - start
        steps_per_sec = NUM_TIMESTEPS / elapsed
        final_reward = metrics["eval/episode_reward"][-1]

        print(f"  Steps/sec: {steps_per_sec:,.0f}")
        print(f"  Final reward: {final_reward:.1f}")
        print(f"  Elapsed: {elapsed:.1f}s")

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()

print("\n" + "=" * 60)
print("If native Brax is 100k+ steps/sec, the gymnax wrapper is the issue.")
print("If native Brax is also slow (~10k), it's a system/JAX config issue.")
print("=" * 60)
