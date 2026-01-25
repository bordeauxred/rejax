#!/usr/bin/env python
"""
Smoke test to measure overhead of metrics logging vs fast path.

Run on GPU:
    python scripts/test_metrics_overhead.py

Expected output:
- Fast path and metrics path timing comparison
- Overhead percentage (should be <20% ideally)
"""
import time
import jax
import gymnax
from rejax import PPO

def main():
    print("=" * 60)
    print("PPO Metrics Overhead Test")
    print("=" * 60)

    # Create PPO for timing test (similar to bench_continual settings)
    env, env_params = gymnax.make('CartPole-v1')
    ppo = PPO.create(
        env=env,
        env_params=env_params,
        num_envs=2048,
        num_steps=128,
        num_epochs=4,
        num_minibatches=4,
        total_timesteps=1000000,
    )

    rng = jax.random.PRNGKey(0)
    ts = ppo.init_state(rng)

    N_ITERS = 50  # ~13M steps total

    print(f"\nConfig:")
    print(f"  num_envs: {ppo.num_envs}")
    print(f"  num_steps: {ppo.num_steps}")
    print(f"  num_epochs: {ppo.num_epochs}")
    print(f"  num_minibatches: {ppo.num_minibatches}")
    print(f"  iterations: {N_ITERS}")
    print(f"  steps/iter: {ppo.num_envs * ppo.num_steps:,}")
    print(f"  total steps: {N_ITERS * ppo.num_envs * ppo.num_steps:,}")

    # JIT compile both versions with fixed length
    print("\nCompiling fast path...")
    compile_start = time.perf_counter()

    @jax.jit
    def train_fast(ts):
        def body(_, ts):
            return ppo.train_iteration(ts)
        return jax.lax.fori_loop(0, N_ITERS, body, ts)

    ts1 = train_fast(ts)
    jax.block_until_ready(ts1)
    fast_compile = time.perf_counter() - compile_start
    print(f"  Compiled in {fast_compile:.1f}s")

    print("\nCompiling metrics path...")
    compile_start = time.perf_counter()

    @jax.jit
    def train_with_metrics(ts):
        def body(ts, _):
            ts, metrics = ppo.train_iteration_with_metrics(ts)
            return ts, metrics
        ts, all_metrics = jax.lax.scan(body, ts, None, length=N_ITERS)
        return ts, jax.tree.map(lambda x: x.mean(), all_metrics)

    ts2, metrics = train_with_metrics(ts)
    jax.block_until_ready(ts2)
    metrics_compile = time.perf_counter() - compile_start
    print(f"  Compiled in {metrics_compile:.1f}s")

    # Show sample metrics
    print(f"\nSample metrics from compilation run:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {float(v):.4f}")

    # Timing runs
    N_RUNS = 5
    print(f"\n{'=' * 60}")
    print(f"Timing {N_ITERS} iterations x {N_RUNS} runs...")
    print(f"{'=' * 60}")

    fast_times = []
    metrics_times = []

    for run in range(N_RUNS):
        # Fast path
        ts_fast = ts
        start = time.perf_counter()
        ts_fast = train_fast(ts_fast)
        jax.block_until_ready(ts_fast)
        fast_t = time.perf_counter() - start
        fast_times.append(fast_t)

        # Metrics path
        ts_metrics = ts
        start = time.perf_counter()
        ts_metrics, _ = train_with_metrics(ts_metrics)
        jax.block_until_ready(ts_metrics)
        metrics_t = time.perf_counter() - start
        metrics_times.append(metrics_t)

        print(f"  Run {run+1}: fast={fast_t:.3f}s, metrics={metrics_t:.3f}s")

    import numpy as np
    fast_mean = np.mean(fast_times)
    fast_std = np.std(fast_times)
    metrics_mean = np.mean(metrics_times)
    metrics_std = np.std(metrics_times)
    overhead = (metrics_mean - fast_mean) / fast_mean * 100

    steps_per_run = N_ITERS * ppo.num_envs * ppo.num_steps

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Fast path:    {fast_mean:.3f}s ± {fast_std:.3f}s ({fast_mean/N_ITERS*1000:.1f}ms/iter)")
    print(f"With metrics: {metrics_mean:.3f}s ± {metrics_std:.3f}s ({metrics_mean/N_ITERS*1000:.1f}ms/iter)")
    print(f"Overhead:     {overhead:+.1f}%")
    print(f"Slowdown:     {metrics_mean/fast_mean:.2f}x")
    print(f"\nThroughput:")
    print(f"  Fast:    {steps_per_run/fast_mean:,.0f} steps/s")
    print(f"  Metrics: {steps_per_run/metrics_mean:,.0f} steps/s")

    # Pass/fail
    print(f"\n{'=' * 60}")
    if overhead < 30:
        print(f"✓ PASS: Overhead {overhead:.1f}% is acceptable (<30%)")
    else:
        print(f"✗ WARN: Overhead {overhead:.1f}% is high (>30%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
