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

    # With wandb (logs per-seed training curves)
    python scripts/throughput_benchmark.py --use-wandb --timesteps 1000000
"""
import argparse
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from rejax import PPO


def make_progress_callback(ppo, run_id, total_timesteps):
    """Create a callback that prints progress during training."""
    eval_callback = ppo.eval_callback
    start_time = [time.time()]

    def progress_callback(ppo, train_state, rng):
        lengths, returns = eval_callback(ppo, train_state, rng)

        def log(step, returns):
            elapsed = time.time() - start_time[0]
            pct = 100 * step.item() / total_timesteps
            steps_per_sec = step.item() / elapsed if elapsed > 0 else 0
            # returns shape: (num_seeds, num_eval_episodes) or (num_eval_episodes,)
            mean_ret = returns.mean().item()
            print(f"    [{run_id}] {step.item():,}/{total_timesteps:,} ({pct:.0f}%) "
                  f"| return={mean_ret:.1f} | {steps_per_sec:,.0f} steps/s")

        jax.experimental.io_callback(log, (), train_state.global_step, returns)
        return lengths, returns

    return progress_callback


def benchmark_config(env_name, hidden_layers, num_envs, num_seeds, total_timesteps,
                     brax_backend=None, eval_freq=None, ortho_mode=None,
                     ortho_lambda=0.2, ortho_coeff=1e-3, activation="tanh"):
    """Benchmark a single configuration and return steps/second + per-seed data."""
    depth = len(hidden_layers)
    width = hidden_layers[0]

    # Descriptive run_id: env_act_d{depth}_mode{coeff}
    env_short = env_name.replace('-MinAtar', '').replace('/', '_')
    act_short = activation[:4] if activation != "groupsort" else "gsort"
    if ortho_mode == "optimizer":
        ortho_str = f"_opt{ortho_coeff}"
    elif ortho_mode == "loss":
        ortho_str = f"_loss{ortho_lambda}"
    else:
        ortho_str = ""
    run_id = f"{env_short}_{act_short}_d{depth}{ortho_str}"

    # PureJAXRL-style config for max throughput
    config = {
        "env": env_name,
        "agent_kwargs": {
            "hidden_layer_sizes": hidden_layers,
            "activation": activation,
        },
        "num_envs": num_envs,
        "num_steps": 128,
        "num_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": 2.5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq if eval_freq else total_timesteps,
        "skip_initial_evaluation": True,
    }

    # Add ortho config if enabled
    if ortho_mode and ortho_mode != "none":
        config["ortho_mode"] = ortho_mode
        config["ortho_lambda"] = ortho_lambda
        config["ortho_coeff"] = ortho_coeff

    if brax_backend and env_name.startswith("brax/"):
        config["env_params"] = {"backend": brax_backend}

    ppo = PPO.create(**config)

    # Add progress callback for both compile and benchmark runs
    if eval_freq and eval_freq > 0:
        progress_callback = make_progress_callback(ppo, run_id, total_timesteps)
        ppo = ppo.replace(eval_callback=progress_callback)

    keys = jax.random.split(jax.random.PRNGKey(0), num_seeds)
    vmap_train = jax.jit(jax.vmap(PPO.train, in_axes=(None, 0)))

    backend_str = f" [{brax_backend}]" if brax_backend and env_name.startswith("brax/") else ""
    ortho_print = f" ortho={ortho_mode}" if ortho_mode and ortho_mode != "none" else ""
    print(f"  Compiling {run_id}{backend_str}{ortho_print}...")

    start = time.time()
    ts, _ = vmap_train(ppo, keys)
    jax.block_until_ready(ts)
    compile_time = time.time() - start
    print(f"  compiled in {compile_time:.1f}s")

    # Benchmark run
    print(f"  Running benchmark...")
    start = time.time()
    ts, eval_results = vmap_train(ppo, keys)
    jax.block_until_ready(ts)
    runtime = time.time() - start

    # Extract per-seed data
    # eval_results shape: (lengths, returns) where each is (num_seeds, num_evals, num_eval_episodes)
    per_seed_data = None
    if eval_results is not None and len(eval_results) == 2:
        lengths, returns = eval_results
        # returns: (num_seeds, num_evals, num_eval_episodes) -> mean over episodes
        returns_np = np.array(returns)
        if returns_np.ndim == 3:
            # (num_seeds, num_evals, num_eval_episodes) -> (num_seeds, num_evals)
            per_seed_returns = returns_np.mean(axis=-1)
        elif returns_np.ndim == 2:
            # (num_seeds, num_eval_episodes) -> (num_seeds,) - single eval
            per_seed_returns = returns_np.mean(axis=-1, keepdims=True)
        else:
            per_seed_returns = returns_np.reshape(num_seeds, -1)

        # Compute eval steps
        num_evals = per_seed_returns.shape[1]
        eval_steps = np.linspace(eval_freq or total_timesteps, total_timesteps, num_evals).astype(int)

        per_seed_data = {
            "eval_steps": eval_steps.tolist(),
            "returns_per_seed": per_seed_returns.tolist(),  # (num_seeds, num_evals)
            "returns_mean": per_seed_returns.mean(axis=0).tolist(),
            "returns_std": per_seed_returns.std(axis=0).tolist(),
            "final_returns_per_seed": per_seed_returns[:, -1].tolist(),
        }

    total_steps = total_timesteps * num_seeds
    steps_per_sec = total_steps / runtime

    result = {
        "run_id": run_id,
        "env": env_name,
        "depth": depth,
        "width": width,
        "hidden_layers": str(hidden_layers),
        "activation": activation,
        "ortho_mode": ortho_mode,
        "num_envs": num_envs,
        "num_seeds": num_seeds,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "compile_time_s": compile_time,
        "runtime_s": runtime,
        "steps_per_second": steps_per_sec,
        "steps_per_second_per_seed": steps_per_sec / num_seeds,
        "per_seed_data": per_seed_data,
    }

    if brax_backend and env_name.startswith("brax/"):
        result["brax_backend"] = brax_backend
    if ortho_mode and ortho_mode != "none":
        result["ortho_lambda"] = ortho_lambda
        result["ortho_coeff"] = ortho_coeff

    if per_seed_data:
        result["final_return_mean"] = np.mean(per_seed_data["final_returns_per_seed"])
        result["final_return_std"] = np.std(per_seed_data["final_returns_per_seed"])

    return result


def log_to_wandb(result, wandb):
    """Log per-seed training curves to wandb."""
    # Use run_id as prefix for clear identification
    prefix = result.get("run_id", f"{result['env'].replace('/', '_')}_d{result['depth']}")

    # Log summary stats
    wandb.log({
        f"{prefix}/steps_per_second": result["steps_per_second"],
        f"{prefix}/compile_time_s": result["compile_time_s"],
        f"{prefix}/final_return_mean": result.get("final_return_mean"),
        f"{prefix}/final_return_std": result.get("final_return_std"),
    })

    # Log per-seed training curves
    per_seed = result.get("per_seed_data")
    if per_seed:
        eval_steps = per_seed["eval_steps"]
        returns_per_seed = per_seed["returns_per_seed"]

        # Log each seed's curve
        for seed_idx, seed_returns in enumerate(returns_per_seed):
            for step, ret in zip(eval_steps, seed_returns):
                wandb.log({
                    f"{prefix}/seed_{seed_idx}/return": ret,
                    f"{prefix}/seed_{seed_idx}/step": step,
                }, step=step)

        # Log mean+std curve
        for step, mean, std in zip(eval_steps, per_seed["returns_mean"], per_seed["returns_std"]):
            wandb.log({
                f"{prefix}/return_mean": mean,
                f"{prefix}/return_std": std,
            }, step=step)


def save_results(results, output_dir):
    """Save results to JSON for offline plotting."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark (purejaxrl-style)")
    parser.add_argument("--envs", nargs="+",
                        default=["Breakout-MinAtar", "Asterix-MinAtar"],
                        help="Environments to test")
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4, 8, 16, 32],
                        help="Network depths to test")
    parser.add_argument("--width", type=int, default=256, help="Network width")
    parser.add_argument("--num-envs", type=int, default=2048,
                        help="Parallel environments")
    parser.add_argument("--num-seeds", type=int, default=3,
                        help="Parallel seeds")
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="Timesteps per run (purejaxrl default: 10M)")
    parser.add_argument("--eval-freq", type=int, default=500_000,
                        help="Eval frequency (default: 500k, set 0 to disable progress)")
    parser.add_argument("--brax-backend", type=str, default="mjx",
                        choices=["mjx", "spring", "positional", "generalized"],
                        help="Brax physics backend")
    parser.add_argument("--use-wandb", action="store_true", help="Log to WandB")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save JSON results")
    # Ortho regularization args
    parser.add_argument("--ortho-mode", type=str, choices=["none", "loss", "optimizer"],
                        default="none", help="Ortho regularization mode")
    parser.add_argument("--ortho-lambda", type=float, default=0.2,
                        help="Ortho lambda for loss mode")
    parser.add_argument("--ortho-coeff", type=float, default=1e-3,
                        help="Ortho coefficient for optimizer mode")
    parser.add_argument("--activation", type=str, default="tanh",
                        help="Activation function (tanh, relu, swish, groupsort, groupsort4, etc.)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="WandB run name (auto-generated if not set)")
    args = parser.parse_args()

    # Generate run name if not provided
    if args.run_name is None:
        if args.ortho_mode == "optimizer":
            ortho_str = f"_opt{args.ortho_coeff}"
        elif args.ortho_mode == "loss":
            ortho_str = f"_loss{args.ortho_lambda}"
        else:
            ortho_str = ""
        args.run_name = f"ppo_{args.activation}{ortho_str}_d{'-'.join(map(str, args.depths))}"

    if args.use_wandb:
        import wandb
        wandb.init(project="rejax-throughput", name=args.run_name, config=vars(args))

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
                    ortho_mode=args.ortho_mode,
                    ortho_lambda=args.ortho_lambda,
                    ortho_coeff=args.ortho_coeff,
                    activation=args.activation,
                )
                results.append(result)

                final_mean = result.get("final_return_mean", "N/A")
                final_std = result.get("final_return_std", 0)
                if isinstance(final_mean, float):
                    print(f"  depth={depth}: {result['steps_per_second']:,.0f} steps/sec | "
                          f"return={final_mean:.1f}±{final_std:.1f}")
                else:
                    print(f"  depth={depth}: {result['steps_per_second']:,.0f} steps/sec")

                if args.use_wandb:
                    log_to_wandb(result, wandb)

            except Exception as e:
                print(f"  depth={depth}: FAILED - {e}")
                import traceback
                traceback.print_exc()

    # Summary table
    print("\n" + "="*100)
    print(f"{'Env':<20} {'Depth':>6} {'Width':>6} {'Steps/sec':>12} {'Per seed':>12} {'Return':>15} {'Compile':>10}")
    print("="*100)
    for r in results:
        mean = r.get('final_return_mean')
        std = r.get('final_return_std', 0)
        ret_str = f"{mean:.1f}±{std:.1f}" if mean is not None else "N/A"
        print(f"{r['env']:<20} {r['depth']:>6} {r['width']:>6} "
              f"{r['steps_per_second']:>12,.0f} {r['steps_per_second_per_seed']:>12,.0f} "
              f"{ret_str:>15} {r['compile_time_s']:>10.1f}s")

    # Save results
    save_results(results, args.output_dir)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
