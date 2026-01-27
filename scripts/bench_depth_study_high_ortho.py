"""
Depth Study: High ortho_coeff comparison for AdaMO and Scale-AdaMO

Tests whether stronger orthogonality enforcement helps deep networks.
Logs to same WandB project as original depth study.
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

import sys
sys.path.insert(0, str(Path(__file__).parent))

from rejax import PPO
from gymnax.environments.minatar.breakout import MinBreakout


def check_orthogonality(params):
    """Return mean Gram deviation across hidden layers."""
    flat = flatten_dict(params, sep='/')
    devs = []
    for path, W in flat.items():
        if '/kernel' in path and W.ndim == 2 and 'Dense_' in path:
            n_in, n_out = W.shape
            if n_in < n_out:
                gram = W @ W.T
                target = jnp.eye(n_in) * (n_out / n_in)
            else:
                gram = W.T @ W
                target = jnp.eye(n_out)
            devs.append(float(jnp.linalg.norm(gram - target)))
    return sum(devs) / len(devs) if devs else 0.0


def get_singular_value_stats(params):
    """Return min/max singular values across hidden layers."""
    flat = flatten_dict(params, sep='/')
    s_mins, s_maxs = [], []
    for path, W in flat.items():
        if '/kernel' in path and W.ndim == 2 and 'Dense_' in path:
            s = jnp.linalg.svd(W, compute_uv=False)
            s_mins.append(float(s.min()))
            s_maxs.append(float(s.max()))
    return min(s_mins) if s_mins else 0, max(s_maxs) if s_maxs else 0


def run_experiment(
    depths,
    configs,
    timesteps,
    eval_freq,
    num_envs,
    num_seeds,
    use_wandb,
    wandb_project,
):
    env = MinBreakout()
    env_params = env.default_params

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=wandb_project,
            name=f"depth_high_ortho_{time.strftime('%Y%m%d_%H%M%S')}",
            config={
                "experiment": "depth_study_high_ortho",
                "depths": depths,
                "configs": [c["name"] for c in configs],
                "timesteps": timesteps,
                "num_seeds": num_seeds,
            },
        )

    results = []

    print("=" * 100)
    print(f"{'Config':<22} {'Depth':<6} {'Return':>12} {'Std':>8} {'Gram Dev':>10} {'s_min':>8} {'s_max':>8}")
    print("=" * 100)

    for depth in depths:
        for cfg in configs:
            hidden = tuple([256] * depth)

            ppo_kwargs = {
                "env": env,
                "env_params": env_params,
                "agent_kwargs": {
                    "network_type": "mlp",
                    "hidden_layer_sizes": hidden,
                    "activation": cfg["activation"],
                    "use_orthogonal_init": True,
                    "use_bias": cfg["use_bias"],
                },
                "num_envs": num_envs,
                "total_timesteps": timesteps,
                "eval_freq": eval_freq,
                "learning_rate": 2.5e-4,
                "num_minibatches": 128,
                "num_epochs": 4,
                "num_steps": 128,
            }

            if cfg["ortho_mode"]:
                ppo_kwargs["ortho_mode"] = cfg["ortho_mode"]
                ppo_kwargs["ortho_coeff"] = cfg["ortho_coeff"]

            if cfg.get("scale_enabled"):
                ppo_kwargs["scale_enabled"] = True
                ppo_kwargs["scale_reg_coeff"] = 0.01

            ppo = PPO.create(**ppo_kwargs)

            # Run multiple seeds
            keys = jax.random.split(jax.random.PRNGKey(42), num_seeds)
            vmap_train = jax.jit(jax.vmap(PPO.train, in_axes=(None, 0)))

            # Compile
            print(f"  Compiling {cfg['name']} depth={depth}...", end=" ", flush=True)
            start = time.time()
            _ = vmap_train(ppo, keys)
            jax.block_until_ready(_)
            compile_time = time.time() - start
            print(f"compiled in {compile_time:.1f}s, training...", end=" ", flush=True)

            # Train
            start = time.time()
            ts, (lengths, returns) = vmap_train(ppo, keys)
            jax.block_until_ready(ts)
            runtime = time.time() - start

            # Get results (use first seed for orthogonality check)
            returns_np = np.array(returns)  # (num_seeds, num_evals, num_episodes)
            final_returns = returns_np[:, -1, :].mean(axis=1)  # per-seed final mean
            final_mean = float(final_returns.mean())
            final_std = float(final_returns.std())

            # Check orthogonality on first seed
            first_seed_params = jax.tree.map(lambda x: x[0], ts.actor_ts.params)
            gram_dev = check_orthogonality(first_seed_params)
            s_min, s_max = get_singular_value_stats(first_seed_params)

            print(f"done in {runtime:.1f}s")
            print(f"{cfg['name']:<22} {depth:<6} {final_mean:>12.1f} {final_std:>8.1f} {gram_dev:>10.2f} {s_min:>8.3f} {s_max:>8.3f}")

            result = {
                "config": cfg["name"],
                "depth": depth,
                "final_return_mean": final_mean,
                "final_return_std": final_std,
                "gram_deviation": gram_dev,
                "s_min": s_min,
                "s_max": s_max,
                "runtime_s": runtime,
                "per_seed_returns": final_returns.tolist(),
            }
            results.append(result)

            # Log to wandb
            if use_wandb and wandb_run:
                import wandb
                config_name = f"{cfg['name']}_depth{depth}"
                wandb.log({
                    f"{config_name}/final_return_mean": final_mean,
                    f"{config_name}/final_return_std": final_std,
                    f"{config_name}/gram_deviation": gram_dev,
                    f"{config_name}/s_min": s_min,
                    f"{config_name}/s_max": s_max,
                    f"{config_name}/depth": depth,
                })

                # Log learning curves as table
                eval_steps = np.arange(returns_np.shape[1]) * eval_freq
                mean_curve = returns_np.mean(axis=(0, 2))  # mean over seeds and episodes
                curve_data = [[config_name, depth, int(step), float(ret)]
                              for step, ret in zip(eval_steps, mean_curve)]
                curve_table = wandb.Table(
                    columns=["config", "depth", "step", "return_mean"],
                    data=curve_data
                )
                wandb.log({f"{config_name}/learning_curve": curve_table})

        print("-" * 100)

    if use_wandb:
        import wandb
        wandb.finish()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--eval-freq", type=int, default=500_000)
    parser.add_argument("--depths", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="rejax-depth-study-ppo-mlp")
    parser.add_argument("--output-dir", type=str, default="results/depth_study_breakout_ppo")
    args = parser.parse_args()

    configs = [
        # AdaMO with stronger ortho (skip 0.1, already ran)
        {"name": "adamo_0.5", "ortho_mode": "optimizer", "ortho_coeff": 0.5,
         "scale_enabled": False, "activation": "groupsort", "use_bias": False},
        {"name": "adamo_1.0", "ortho_mode": "optimizer", "ortho_coeff": 1.0,
         "scale_enabled": False, "activation": "groupsort", "use_bias": False},

        # Scale-AdaMO with stronger ortho
        {"name": "scale_adamo_0.5", "ortho_mode": "optimizer", "ortho_coeff": 0.5,
         "scale_enabled": True, "activation": "groupsort", "use_bias": False},
        {"name": "scale_adamo_1.0", "ortho_mode": "optimizer", "ortho_coeff": 1.0,
         "scale_enabled": True, "activation": "groupsort", "use_bias": False},
    ]

    print("=" * 100)
    print("Depth Study: High Ortho Coefficient Comparison")
    print("=" * 100)
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Seeds: {args.num_seeds}")
    print(f"Depths: {args.depths}")
    print(f"Configs: {[c['name'] for c in configs]}")
    print("=" * 100)

    results = run_experiment(
        depths=args.depths,
        configs=configs,
        timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        num_envs=args.num_envs,
        num_seeds=args.num_seeds,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"high_ortho_{time.strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump({
            "experiment": "depth_study_high_ortho",
            "timesteps": args.timesteps,
            "num_seeds": args.num_seeds,
            "depths": args.depths,
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for depth in args.depths:
        print(f"\nDepth {depth}:")
        depth_results = [r for r in results if r["depth"] == depth]
        for r in depth_results:
            print(f"  {r['config']:<22}: {r['final_return_mean']:>6.1f} +/- {r['final_return_std']:<5.1f} "
                  f"(gram={r['gram_deviation']:.2f}, s=[{r['s_min']:.2f}, {r['s_max']:.2f}])")


if __name__ == "__main__":
    main()
