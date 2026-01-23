"""
MLP vs CNN comparison for MinAtar games.

Compares network architectures:
1. MLP PPO: Flattened 10x10xC input, (64, 64) hidden layers, tanh
2. CNN PPO: pgx-style conv(32, k=2) + avgpool + mlp(64,64,64), relu

Usage:
    # Quick test
    python scripts/bench_mlp_vs_cnn.py --timesteps 100000 --num-seeds 1

    # Full comparison
    python scripts/bench_mlp_vs_cnn.py --timesteps 20000000 --num-seeds 5 --use-wandb
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from rejax import PPO
from gymnax.environments.minatar.breakout import MinBreakout
from gymnax.environments.minatar.asterix import MinAsterix
from gymnax.environments.minatar.space_invaders import MinSpaceInvaders
from gymnax.environments.minatar.freeway import MinFreeway
from minatar_seaquest_fixed import MinSeaquestFixed


MINATAR_GAMES = {
    "Breakout-MinAtar": MinBreakout,
    "Asterix-MinAtar": MinAsterix,
    "SpaceInvaders-MinAtar": MinSpaceInvaders,
    "Freeway-MinAtar": MinFreeway,
    "Seaquest-MinAtar": MinSeaquestFixed,
}

GAME_ORDER = [
    "Breakout-MinAtar",
    "Asterix-MinAtar",
    "SpaceInvaders-MinAtar",
    "Freeway-MinAtar",
    "Seaquest-MinAtar",
]

# Network configurations to compare
NETWORK_CONFIGS = {
    "mlp": {
        "name": "MLP PPO",
        "agent_kwargs": {
            "network_type": "mlp",
            "hidden_layer_sizes": (64, 64),
            "activation": "tanh",
            "use_orthogonal_init": True,
        },
    },
    "cnn": {
        "name": "CNN PPO (pgx)",
        "agent_kwargs": {
            "network_type": "cnn",
            "conv_channels": (32,),
            "mlp_hidden_sizes": (64, 64, 64),
            "kernel_size": 2,
            "use_avgpool": True,
            "pool_size": 2,
            "activation": "relu",
            "use_orthogonal_init": False,
        },
    },
}


def create_env(game_name: str) -> Tuple[Any, Any]:
    """Create a native MinAtar environment."""
    env = MINATAR_GAMES[game_name]()
    return env, env.default_params


def create_ppo_config(
    env,
    env_params,
    network_config: Dict,
    total_timesteps: int,
    eval_freq: int,
    num_envs: int,
) -> Dict:
    """Create PPO config for comparison."""
    return {
        "env": env,
        "env_params": env_params,
        "agent_kwargs": network_config["agent_kwargs"].copy(),
        # pgx MinAtar PPO hyperparameters
        "num_envs": num_envs,
        "num_steps": 128,
        "num_epochs": 3,
        "num_minibatches": 1,
        "learning_rate": 3e-4,
        "anneal_lr": False,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "skip_initial_evaluation": False,
    }


@dataclass
class ComparisonResult:
    """Results for a single game comparison."""
    game: str
    results: Dict[str, Dict]  # network_type -> metrics


def benchmark_game(
    game_name: str,
    total_timesteps: int,
    num_seeds: int,
    num_envs: int,
    eval_freq: int,
) -> ComparisonResult:
    """Benchmark both network types on a single game."""
    print(f"\n{'='*70}")
    print(f"Game: {game_name}")
    print(f"{'='*70}")

    env, env_params = create_env(game_name)
    results = {}

    for net_type, net_config in NETWORK_CONFIGS.items():
        print(f"\n  Network: {net_config['name']}")

        config = create_ppo_config(
            env=env,
            env_params=env_params,
            network_config=net_config,
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            num_envs=num_envs,
        )

        ppo = PPO.create(**config)

        # Prepare seeds
        keys = jax.random.split(jax.random.PRNGKey(42), num_seeds)

        # Vectorized training
        vmap_train = jax.jit(jax.vmap(PPO.train, in_axes=(None, 0)))

        # Compile
        print(f"    Compiling...")
        start = time.time()
        _ = vmap_train(ppo, keys)
        jax.block_until_ready(_)
        compile_time = time.time() - start
        print(f"    Compiled in {compile_time:.1f}s")

        # Train
        print(f"    Training {num_seeds} seeds...")
        start = time.time()
        ts, eval_results = vmap_train(ppo, keys)
        jax.block_until_ready(ts)
        runtime = time.time() - start

        steps_per_second = (total_timesteps * num_seeds) / runtime

        # Extract final returns
        lengths, returns = eval_results
        final_returns = np.array(returns)[:, -1, :].mean(axis=-1)  # (num_seeds,)

        results[net_type] = {
            "name": net_config["name"],
            "final_return_mean": float(final_returns.mean()),
            "final_return_std": float(final_returns.std()),
            "runtime_s": runtime,
            "steps_per_second": steps_per_second,
        }

        print(f"    Final return: {final_returns.mean():.1f} +/- {final_returns.std():.1f}")
        print(f"    Runtime: {runtime:.1f}s ({steps_per_second:,.0f} steps/s)")

    return ComparisonResult(game=game_name, results=results)


def run_comparison(
    games: List[str],
    total_timesteps: int,
    num_seeds: int,
    num_envs: int,
    eval_freq: int,
    use_wandb: bool,
    wandb_project: str,
) -> Dict:
    """Run full comparison across all games."""
    all_results = {
        "experiment": "mlp_vs_cnn_comparison",
        "total_timesteps": total_timesteps,
        "num_seeds": num_seeds,
        "num_envs": num_envs,
        "eval_freq": eval_freq,
        "network_configs": {k: v["agent_kwargs"] for k, v in NETWORK_CONFIGS.items()},
        "games": games,
        "per_game_results": [],
    }

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=wandb_project,
            name=f"mlp_vs_cnn_{time.strftime('%Y%m%d_%H%M%S')}",
            config=all_results,
        )

    for game_name in games:
        result = benchmark_game(
            game_name=game_name,
            total_timesteps=total_timesteps,
            num_seeds=num_seeds,
            num_envs=num_envs,
            eval_freq=eval_freq,
        )

        all_results["per_game_results"].append({
            "game": result.game,
            "results": result.results,
        })

        if use_wandb:
            import wandb
            for net_type, metrics in result.results.items():
                wandb.log({
                    f"{game_name}/{net_type}/final_return": metrics["final_return_mean"],
                    f"{game_name}/{net_type}/runtime_s": metrics["runtime_s"],
                })

    if use_wandb:
        import wandb
        wandb.finish()

    return all_results


def print_summary(results: Dict):
    """Print comparison summary table."""
    print("\n" + "=" * 90)
    print(f"{'Game':<25} {'MLP Return':>15} {'CNN Return':>15} {'Winner':>15}")
    print("=" * 90)

    mlp_wins = 0
    cnn_wins = 0

    for r in results["per_game_results"]:
        mlp_ret = r["results"]["mlp"]["final_return_mean"]
        cnn_ret = r["results"]["cnn"]["final_return_mean"]
        mlp_std = r["results"]["mlp"]["final_return_std"]
        cnn_std = r["results"]["cnn"]["final_return_std"]

        if cnn_ret > mlp_ret:
            winner = "CNN"
            cnn_wins += 1
        else:
            winner = "MLP"
            mlp_wins += 1

        print(f"{r['game']:<25} {mlp_ret:>7.1f}+/-{mlp_std:>4.1f} {cnn_ret:>7.1f}+/-{cnn_std:>4.1f} {winner:>15}")

    print("=" * 90)
    print(f"\nOverall: MLP wins {mlp_wins}, CNN wins {cnn_wins}")


def save_results(results: Dict, output_dir: Path) -> Path:
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"mlp_vs_cnn_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="MLP vs CNN comparison for MinAtar")
    parser.add_argument(
        "--games", nargs="+", default=GAME_ORDER,
        choices=GAME_ORDER, help="Games to benchmark"
    )
    parser.add_argument(
        "--timesteps", type=int, default=20_000_000,
        help="Training timesteps per game (pgx default: 20M)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5,
        help="Number of seeds per game"
    )
    parser.add_argument(
        "--num-envs", type=int, default=4096,
        help="Parallel environments"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=500_000,
        help="Evaluation frequency (timesteps)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/mlp_vs_cnn",
        help="Directory for results"
    )
    parser.add_argument(
        "--use-wandb", action="store_true",
        help="Enable W&B logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="rejax-minatar",
        help="W&B project name"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MLP vs CNN Comparison for MinAtar")
    print("=" * 70)
    print(f"Games: {args.games}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Seeds: {args.num_seeds}")
    print(f"Envs: {args.num_envs}")
    print("=" * 70)
    print("Networks:")
    print("  MLP: (64, 64), tanh, orthogonal init")
    print("  CNN: conv(32,k=2) + avgpool + mlp(64,64,64), relu")
    print("=" * 70)

    results = run_comparison(
        games=args.games,
        total_timesteps=args.timesteps,
        num_seeds=args.num_seeds,
        num_envs=args.num_envs,
        eval_freq=args.eval_freq,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )

    output_dir = Path(args.output_dir)
    save_results(results, output_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
