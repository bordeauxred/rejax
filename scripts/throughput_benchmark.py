"""
Throughput benchmark for rejax PPO at various network depths.

PureJAXRL-style settings for maximum throughput comparison.
Tests: MinAtar (Breakout, Asterix), Brax (HalfCheetah), Gymnax (Pendulum)

Usage:
    python scripts/throughput_benchmark.py --depths 2 4 8 16 32
"""
import argparse
import time
import jax
import jax.numpy as jnp
from rejax import PPO


def benchmark_config(env_name, hidden_layers, num_envs, num_seeds, total_timesteps,
                     brax_backend=None):
    """Benchmark a single configuration and return steps/second."""
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
        "eval_freq": total_timesteps,  # Only eval at end
        "skip_initial_evaluation": True,
    }

    # Add brax backend if specified and env is brax
    if brax_backend and env_name.startswith("brax/"):
        config["env_params"] = {"backend": brax_backend}

    ppo = PPO.create(**config)
    keys = jax.random.split(jax.random.PRNGKey(0), num_seeds)

    # Compile
    vmap_train = jax.jit(jax.vmap(PPO.train, in_axes=(None, 0)))

    # Warmup compilation
    depth = len(hidden_layers)
    print(f"  Compiling depth={depth} ({hidden_layers[0]}x{depth})...", end=" ", flush=True)
    start = time.time()
    ts, _ = vmap_train(ppo, keys)
    jax.block_until_ready(ts)
    compile_time = time.time() - start
    print(f"compiled in {compile_time:.1f}s")

    # Benchmark (3 runs)
    times = []
    for _ in range(3):
        start = time.time()
        ts, _ = vmap_train(ppo, keys)
        jax.block_until_ready(ts)
        times.append(time.time() - start)

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
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Timesteps per run")
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
                    brax_backend=args.brax_backend
                )
                results.append(result)
                print(f"  depth={depth}: {result['steps_per_second']:,.0f} steps/sec "
                      f"({result['steps_per_second_per_seed']:,.0f}/seed)")

                if args.use_wandb:
                    wandb.log(result)
            except Exception as e:
                print(f"  depth={depth}: FAILED - {e}")

    # Summary table
    print("\n" + "="*80)
    print(f"{'Env':<20} {'Depth':>6} {'Width':>6} {'Steps/sec':>12} {'Per seed':>12} {'Compile':>10}")
    print("="*80)
    for r in results:
        print(f"{r['env']:<20} {r['depth']:>6} {r['width']:>6} "
              f"{r['steps_per_second']:>12,.0f} {r['steps_per_second_per_seed']:>12,.0f} "
              f"{r['compile_time_s']:>10.1f}s")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
