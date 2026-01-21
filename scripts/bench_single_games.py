"""
Per-game baselines for MinAtar to establish upper bounds for continual learning comparison.

Runs ortho optimizer on each MinAtar game independently (no continual learning).

Usage:
    # Quick test
    python scripts/bench_single_games.py --timesteps 100000 --num-seeds 1

    # Full baseline (same steps as one game in continual)
    python scripts/bench_single_games.py --timesteps 10000000 --num-seeds 3 --use-wandb

    # Single game
    python scripts/bench_single_games.py --games Breakout-MinAtar --timesteps 1000000
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rejax import PPO
from bench_continual import (
    GAME_ORDER,
    MINATAR_GAMES,
    EXPERIMENT_CONFIGS,
    create_padded_env,
)


def make_progress_callback(game_name: str, config_name: str, total_timesteps: int, original_eval_callback):
    """Create a callback that prints progress during training."""
    start_time = [time.time()]

    def progress_callback(ppo, train_state, rng):
        lengths, returns = original_eval_callback(ppo, train_state, rng)

        def log(step, returns):
            elapsed = time.time() - start_time[0]
            pct = 100 * step.item() / total_timesteps
            steps_per_sec = step.item() / elapsed if elapsed > 0 else 0
            mean_ret = returns.mean().item()
            print(f"    [{config_name}|{game_name}] {step.item():,}/{total_timesteps:,} ({pct:.0f}%) "
                  f"| return={mean_ret:.1f} | {steps_per_sec:,.0f} steps/s")

        jax.experimental.io_callback(log, (), train_state.global_step, returns)
        return lengths, returns

    return progress_callback


def benchmark_single_game(
    game_name: str,
    config_name: str,
    experiment_config: Dict,
    total_timesteps: int,
    num_seeds: int,
    num_envs: int = 2048,
    eval_freq: int = 500_000,
    use_wandb: bool = False,
) -> Dict:
    """Benchmark a single game with given configuration."""
    print(f"\n{'='*60}")
    print(f"Game: {game_name} | Config: {config_name}")
    print(f"{'='*60}")

    # Create padded environment
    env, env_params = create_padded_env(game_name)

    # Create PPO config
    config = {
        "env": env,
        "env_params": env_params,
        "agent_kwargs": {
            "hidden_layer_sizes": (256, 256),
            "activation": experiment_config.get("activation", "tanh"),
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
        "eval_freq": eval_freq,
        "skip_initial_evaluation": True,
    }

    ortho_mode = experiment_config.get("ortho_mode")
    if ortho_mode and ortho_mode != "none":
        config["ortho_mode"] = ortho_mode
        config["ortho_coeff"] = experiment_config.get("ortho_coeff", 0.1)

    ppo = PPO.create(**config)

    # Add progress callback (wrap the original eval_callback)
    original_eval_callback = ppo.eval_callback
    progress_callback = make_progress_callback(game_name, config_name, total_timesteps, original_eval_callback)
    ppo = ppo.replace(eval_callback=progress_callback)

    # Prepare training
    keys = jax.random.split(jax.random.PRNGKey(0), num_seeds)
    vmap_train = jax.jit(jax.vmap(PPO.train, in_axes=(None, 0)))

    # Compile
    print(f"  Compiling...")
    start = time.time()
    ts, _ = vmap_train(ppo, keys)
    jax.block_until_ready(ts)
    compile_time = time.time() - start
    print(f"  Compiled in {compile_time:.1f}s")

    # Benchmark
    print(f"  Training...")
    start = time.time()
    ts, eval_results = vmap_train(ppo, keys)
    jax.block_until_ready(ts)
    runtime = time.time() - start

    # Extract results
    lengths, returns = eval_results
    returns_np = np.array(returns)

    # Returns shape: (num_seeds, num_evals, num_eval_episodes)
    if returns_np.ndim == 3:
        per_seed_returns = returns_np.mean(axis=-1)  # (num_seeds, num_evals)
        final_returns = per_seed_returns[:, -1]
    else:
        final_returns = returns_np.mean(axis=-1)

    result = {
        "game": game_name,
        "config_name": config_name,
        "experiment_config": experiment_config,
        "num_seeds": num_seeds,
        "total_timesteps": total_timesteps,
        "compile_time_s": compile_time,
        "runtime_s": runtime,
        "steps_per_second": (total_timesteps * num_seeds) / runtime,
        "final_return_mean": float(np.mean(final_returns)),
        "final_return_std": float(np.std(final_returns)),
        "final_returns_per_seed": final_returns.tolist(),
    }

    print(f"  Result: return={result['final_return_mean']:.1f}±{result['final_return_std']:.1f}")

    if use_wandb:
        import wandb
        wandb.log({
            f"{config_name}/{game_name}/final_return_mean": result["final_return_mean"],
            f"{config_name}/{game_name}/final_return_std": result["final_return_std"],
            f"{config_name}/{game_name}/runtime_s": runtime,
        })

    return result


def run_all_baselines(
    games: List[str],
    configs: List[Dict],
    total_timesteps: int,
    num_seeds: int,
    num_envs: int,
    eval_freq: int,
    use_wandb: bool,
    wandb_project: str,
) -> Dict:
    """Run baselines for all games and configs."""
    results = {
        "games": games,
        "configs": [c["name"] for c in configs],
        "total_timesteps": total_timesteps,
        "num_seeds": num_seeds,
        "per_game_results": [],
    }

    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=f"single_game_baselines_{time.strftime('%Y%m%d_%H%M%S')}",
            config={
                "games": games,
                "configs": [c["name"] for c in configs],
                "total_timesteps": total_timesteps,
                "num_seeds": num_seeds,
            },
        )

    for config in configs:
        config_name = config["name"]
        print(f"\n{'#'*70}")
        print(f"# Config: {config_name}")
        print(f"{'#'*70}")

        for game_name in games:
            result = benchmark_single_game(
                game_name=game_name,
                config_name=config_name,
                experiment_config=config,
                total_timesteps=total_timesteps,
                num_seeds=num_seeds,
                num_envs=num_envs,
                eval_freq=eval_freq,
                use_wandb=use_wandb,
            )
            results["per_game_results"].append(result)

    if use_wandb:
        import wandb
        wandb.finish()

    return results


def save_results(results: Dict, output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"single_game_baselines_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return output_file


def print_summary(results: Dict):
    """Print summary table of results."""
    print("\n" + "="*90)
    print(f"{'Game':<25} {'Config':<20} {'Return':>15} {'Runtime':>12} {'Steps/s':>12}")
    print("="*90)

    for r in results["per_game_results"]:
        ret_str = f"{r['final_return_mean']:.1f}±{r['final_return_std']:.1f}"
        print(f"{r['game']:<25} {r['config_name']:<20} {ret_str:>15} "
              f"{r['runtime_s']:>10.1f}s {r['steps_per_second']:>12,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Per-game MinAtar baselines")
    parser.add_argument("--games", nargs="+", default=GAME_ORDER,
                        choices=GAME_ORDER, help="Games to benchmark")
    parser.add_argument("--configs", nargs="+",
                        default=["baseline", "ortho_opt"],
                        choices=["baseline", "ortho_opt", "ortho_opt_linear_lr"],
                        help="Experiment configurations to run")
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="Training timesteps per game")
    parser.add_argument("--num-seeds", type=int, default=3,
                        help="Number of seeds")
    parser.add_argument("--num-envs", type=int, default=2048,
                        help="Parallel environments")
    parser.add_argument("--eval-freq", type=int, default=500_000,
                        help="Evaluation frequency")
    parser.add_argument("--output-dir", type=str, default="results/single_games",
                        help="Directory for results")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="rejax-continual",
                        help="W&B project name")

    args = parser.parse_args()

    # Filter configs
    configs_to_run = [c for c in EXPERIMENT_CONFIGS if c["name"] in args.configs]

    results = run_all_baselines(
        games=args.games,
        configs=configs_to_run,
        total_timesteps=args.timesteps,
        num_seeds=args.num_seeds,
        num_envs=args.num_envs,
        eval_freq=args.eval_freq,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )

    # Save and print summary
    save_results(results, Path(args.output_dir))
    print_summary(results)


if __name__ == "__main__":
    main()
