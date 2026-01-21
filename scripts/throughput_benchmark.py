"""
Throughput benchmark for rejax PPO at various network depths.

PureJAXRL-style settings for maximum throughput comparison.
Tests: MinAtar (Breakout, Asterix), Brax (HalfCheetah), Gymnax (Pendulum)

Usage:
    # Quick smoketest
    python scripts/throughput_benchmark.py --envs Breakout-MinAtar --depths 2 4 --timesteps 100000

    # Full benchmark (10M steps like purejaxrl)
    python scripts/throughput_benchmark.py --depths 2 4 8 16 32

    # Brax with MJX backend
    python scripts/throughput_benchmark.py --envs brax/halfcheetah --brax-backend mjx
"""
import argparse
import time
import jax
import jax.numpy as jnp
from rejax import PPO


def make_progress_callback(ppo, run_id, total_timesteps):
    """Create a callback that prints progress during training."""
    eval_callback = ppo.eval_callback
    start_time = [time.time()]  # mutable container for closure

    def progress_callback(ppo, train_state, rng):
        lengths, returns = eval_callback(ppo, train_state, rng)

        def log(step, lengths, returns):
            elapsed = time.time() - start_time[0]
            pct = 100 * step.item() / total_timesteps
            steps_per_sec = step.item() / elapsed if elapsed > 0 else 0
            print(f"    [{run_id}] {step.item():,}/{total_timesteps:,} ({pct:.0f}%) "
                  f"| return={returns.mean().item():.1f} | {steps_per_sec:,.0f} steps/s")

        jax.experimental.io_callback(
            log, (), train_state.global_step, lengths, returns
        )
        return lengths, returns  # still return for final result

    return progress_callback


def benchmark_config(env_name, hidden_layers, num_envs, num_seeds, total_timesteps,
                     brax_backend=None, eval_freq=None):
    """Benchmark a single configuration and return steps/second."""
    depth = len(hidden_layers)
    run_id = f"{env_name.replace('/', '_')}_d{depth}"

    # PureJAXRL-style config for max throughput
    config = {
        "env": env_name,
        "agent_kwargs": {
            "hidden_layer_sizes": hidden_layers,
            "activation": "tanh",  # Match purejaxrl
        },
        "num_envs": num_envs,
        "num_steps": 128,         # purejaxrl default
        "num_epochs": 4,          # purejaxrl UPDATE_EPOCHS
        "num_minibatches": 4,     # purejaxrl default
        "learning_rate": 2.5e-4,  # purejaxrl LR
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,     # purejaxrl default
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq if eval_freq else total_timesteps,
        "skip_initial_evaluation": True,
    }

    # Add brax backend if specified and env is brax
    if brax_backend and env_name.startswith("brax/"):
        config["env_params"] = {"backend": brax_backend}

    ppo = PPO.create(**config)

    # Add progress callback for intermediate logging
    if eval_freq and eval_freq > 0:
        progress_callback = make_progress_callback(ppo, run_id, total_timesteps)
        ppo = ppo.replace(eval_callback=progress_callback)

    keys = jax.random.split(jax.random.PRNGKey(0), num_seeds)

    # Compile
    vmap_train = jax.jit(jax.vmap(PPO.train, in_axes=(None, 0)))

    backend_str = f" [{brax_backend}]" if brax_backend and env_name.startswith("brax/") else ""
    print(f"  Compiling depth={depth} ({hidden_layers[0]}x{depth}){backend_str}...", end=" ", flush=True)
    start = time.time()
    ts, _ = vmap_train(ppo, keys)
    jax.block_until_ready(ts)
    compile_time = time.time() - start
    print(f"compiled in {compile_time:.1f}s")

    # Benchmark (3 runs, or 1 if logging progress)
    num_runs = 1 if (eval_freq and eval_freq > 0) else 3
    times = []
    final_returns = None
    for i in range(num_runs):
        if num_runs > 1:
            print(f"    Run {i+1}/{num_runs}...", end=" ", flush=True)
        start = time.time()
        ts, eval_results = vmap_train(ppo, keys)
        jax.block_until_ready(ts)
        elapsed = time.time() - start
        times.append(elapsed)
        if num_runs > 1:
            print(f"{elapsed:.1f}s")
        # Capture final returns (eval_results is (lengths, returns) from last eval)
        if eval_results is not None and len(eval_results) == 2:
            _, returns = eval_results
            final_returns = float(returns.mean())

    avg_time = sum(times) / len(times)
    total_steps = total_timesteps * num_seeds
    steps_per_sec = total_steps / avg_time

    result = {
        "env": env_name,
        "depth": depth,
        "width": hidden_layers[0],
        "hidden_layers": str(hidden_layers),
        "num_envs": num_envs,
        "num_seeds": num_seeds,
        "total_timesteps": total_timesteps,
        "compile_time_s": compile_time,
        "avg_runtime_s": avg_time,
        "steps_per_second": steps_per_sec,
        "steps_per_second_per_seed": steps_per_sec / num_seeds,
        "final_return": final_returns,
    }
    if brax_backend and env_name.startswith("brax/"):
        result["brax_backend"] = brax_backend
    return result


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark (purejaxrl-style)")
    parser.add_argument("--envs", nargs="+",
                        default=["Breakout-MinAtar", "Asterix-MinAtar"],
                        help="Environments to test")
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4, 8, 16, 32],
                        help="Network depths to test")
    parser.add_argument("--width", type=int, default=256, help="Network width")
    parser.add_argument("--num-envs", type=int, default=2048,
                        help="Parallel environments (purejaxrl high-throughput)")
    parser.add_argument("--num-seeds", type=int, default=3,
                        help="Parallel seeds (3 is enough for speed test)")
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="Timesteps per run (purejaxrl default: 10M)")
    parser.add_argument("--eval-freq", type=int, default=500_000,
                        help="Progress print frequency (default: 500k, set 0 to disable)")
    parser.add_argument("--brax-backend", type=str, default="mjx",
                        choices=["mjx", "spring", "positional", "generalized"],
                        help="Brax physics backend: mjx (MuJoCo XLA, recommended), "
                             "spring (fast/simple), generalized (accurate/slow)")
    parser.add_argument("--use-wandb", action="store_true", help="Log to WandB")
    args = parser.parse_args()

    if args.use_wandb:
        import wandb
        wandb.init(project="rejax-throughput", config=vars(args))

    results = []
    for env in args.envs:
        print(f"\n{'='*60}")
        print(f"Environment: {env}")
        print(f"{'='*60}")

        for depth in args.depths:
            hidden = (args.width,) * depth
            try:
                result = benchmark_config(
                    env, hidden, args.num_envs, args.num_seeds, args.timesteps,
                    brax_backend=args.brax_backend,
                    eval_freq=args.eval_freq,
                )
                results.append(result)
                print(f"  depth={depth}: {result['steps_per_second']:,.0f} steps/sec "
                      f"({result['steps_per_second_per_seed']:,.0f}/seed)")

                if args.use_wandb:
                    import wandb
                    wandb.log({f"summary/{k}": v for k, v in result.items()
                              if isinstance(v, (int, float))})
            except Exception as e:
                print(f"  depth={depth}: FAILED - {e}")
                import traceback
                traceback.print_exc()

    # Summary table
    print("\n" + "="*90)
    print(f"{'Env':<20} {'Depth':>6} {'Width':>6} {'Steps/sec':>12} {'Per seed':>12} {'Return':>10} {'Compile':>10}")
    print("="*90)
    for r in results:
        ret_str = f"{r['final_return']:.1f}" if r.get('final_return') is not None else "N/A"
        print(f"{r['env']:<20} {r['depth']:>6} {r['width']:>6} "
              f"{r['steps_per_second']:>12,.0f} {r['steps_per_second_per_seed']:>12,.0f} "
              f"{ret_str:>10} {r['compile_time_s']:>10.1f}s")

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
