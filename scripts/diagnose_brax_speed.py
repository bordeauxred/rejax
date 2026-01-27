"""
Diagnose Brax PPO speed issues.

Run on GPU server:
    uv run python scripts/diagnose_brax_speed.py
"""
import jax
import jax.numpy as jnp
import time


def test_gpu():
    """Verify GPU is being used."""
    print("=" * 60)
    print("1. GPU Check")
    print("=" * 60)
    print(f"Available devices: {jax.devices()}")
    print(f"Default device: {jax.devices()[0]}")

    # Force GPU
    gpu = jax.devices('gpu')[0]
    with jax.default_device(gpu):
        x = jnp.ones((2000, 2000))
        start = time.time()
        for _ in range(100):
            x = x @ x
            x = x / x.max()
        jax.block_until_ready(x)
        elapsed = time.time() - start
        print(f"GPU matmul test: {elapsed:.2f}s (should be < 1s)")


def test_native_brax():
    """Test native Brax PPO speed (no gymnax wrapper)."""
    print("\n" + "=" * 60)
    print("2. Native Brax PPO (no gymnax wrapper)")
    print("=" * 60)

    from brax.envs import create
    from brax.training.agents.ppo import train as ppo_train

    for backend in ['spring', 'mjx']:
        print(f"\nBackend: {backend}")
        try:
            env = create('halfcheetah', backend=backend)

            start = time.time()
            make_policy, params, metrics = ppo_train.train(
                environment=env,
                num_timesteps=500_000,
                num_envs=2048,
                episode_length=1000,
                num_evals=1,
            )
            elapsed = time.time() - start

            steps_per_sec = 500_000 / elapsed
            final_reward = metrics["eval/episode_reward"][-1]
            print(f"  Steps/sec: {steps_per_sec:,.0f}")
            print(f"  Final reward: {final_reward:.1f}")
            print(f"  Elapsed: {elapsed:.1f}s")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()


def test_rejax_brax():
    """Test rejax PPO with Brax (through gymnax wrapper)."""
    print("\n" + "=" * 60)
    print("3. Rejax PPO with Brax (gymnax wrapper)")
    print("=" * 60)

    from rejax import PPO
    from rejax.compat.brax2gymnax import create_brax

    for backend in ['spring', 'mjx']:
        print(f"\nBackend: {backend}")
        try:
            env, env_params = create_brax('halfcheetah', backend=backend)

            config = {
                "env": env,
                "env_params": env_params,
                "agent_kwargs": {"hidden_layer_sizes": (256, 256)},
                "num_envs": 2048,
                "num_steps": 10,
                "total_timesteps": 500_000,
                "eval_freq": 500_000,
            }
            ppo = PPO.create(**config)

            rng = jax.random.PRNGKey(0)

            # Compile
            print("  Compiling...", end=" ", flush=True)
            compile_start = time.time()
            ts, _ = PPO.train(ppo, rng)
            jax.block_until_ready(ts)
            compile_time = time.time() - compile_start
            print(f"{compile_time:.1f}s")

            # Benchmark
            print("  Running...", end=" ", flush=True)
            start = time.time()
            ts, (_, returns) = PPO.train(ppo, rng)
            jax.block_until_ready(ts)
            elapsed = time.time() - start

            steps_per_sec = 500_000 / elapsed
            print(f"done")
            print(f"  Steps/sec: {steps_per_sec:,.0f}")
            print(f"  Final return: {returns.mean():.1f}")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    test_gpu()
    test_native_brax()
    test_rejax_brax()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If native Brax is fast but rejax is slow, the issue is the gymnax wrapper.")
    print("If both are slow, the issue is GPU/JAX configuration.")
