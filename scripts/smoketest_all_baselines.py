"""
Fast smoke test: All baselines on all MinAtar games (single-task).
Verifies learning + no bugs before overnight continual runs.

Target: ~20 min on A100 with 1M steps, 2 seeds

Usage:
    uv run python scripts/smoketest_all_baselines.py
    uv run python scripts/smoketest_all_baselines.py --timesteps 500000 --num-seeds 1  # faster
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

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


GAMES = {
    "Breakout": MinBreakout,
    "Asterix": MinAsterix,
    "SpaceInvaders": MinSpaceInvaders,
    "Freeway": MinFreeway,
    "Seaquest": MinSeaquestFixed,
}

# All baseline configs to test
CONFIGS = {
    # MLP baselines (4x256, Lyle et al. style)
    "mlp_baseline": {
        "network_type": "mlp",
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "tanh",
        "use_orthogonal_init": True,
        "use_bias": True,
        "ortho_mode": None,
        "learning_rate": 2.5e-4,
    },
    "mlp_adamo": {
        "network_type": "mlp",
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "groupsort",
        "use_orthogonal_init": False,
        "use_bias": False,
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "learning_rate": 2.5e-4,
    },
    "mlp_adamo_lyle": {
        "network_type": "mlp",
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "groupsort",
        "use_orthogonal_init": False,
        "use_bias": False,
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "learning_rate": 6.25e-5,  # Lyle LR
    },
    # CNN baselines (pgx style)
    "cnn_baseline": {
        "network_type": "cnn",
        "conv_channels": (32,),
        "mlp_hidden_sizes": (64, 64, 64),
        "kernel_size": 2,
        "use_avgpool": True,
        "activation": "relu",
        "use_orthogonal_init": False,
        "use_bias": True,
        "ortho_mode": None,
        "learning_rate": 3e-4,
    },
    "cnn_adamo": {
        "network_type": "cnn",
        "conv_channels": (32,),
        "mlp_hidden_sizes": (64, 64, 64),
        "kernel_size": 2,
        "use_avgpool": True,
        "activation": "relu",
        "use_orthogonal_init": False,
        "use_bias": False,
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "learning_rate": 3e-4,
    },
    "cnn_adamo_lyle": {
        "network_type": "cnn",
        "conv_channels": (32,),
        "mlp_hidden_sizes": (64, 64, 64),
        "kernel_size": 2,
        "use_avgpool": True,
        "activation": "relu",
        "use_orthogonal_init": False,
        "use_bias": False,
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "learning_rate": 6.25e-5,
    },
}


def create_ppo(game_name: str, config: Dict, timesteps: int, num_envs: int) -> PPO:
    """Create PPO instance for a game with given config."""
    env = GAMES[game_name]()
    env_params = env.default_params

    agent_kwargs = {k: v for k, v in config.items()
                    if k in ["network_type", "hidden_layer_sizes", "conv_channels",
                             "mlp_hidden_sizes", "kernel_size", "use_avgpool",
                             "activation", "use_orthogonal_init", "use_bias"]}

    ppo_config = {
        "env": env,
        "env_params": env_params,
        "agent_kwargs": agent_kwargs,
        "num_envs": num_envs,
        "num_steps": 128,
        "num_epochs": 3,
        "num_minibatches": 1,
        "learning_rate": config["learning_rate"],
        "anneal_lr": False,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": timesteps,
        "eval_freq": timesteps,  # eval only at end
        "skip_initial_evaluation": True,
    }

    if config.get("ortho_mode"):
        ppo_config["ortho_mode"] = config["ortho_mode"]
        ppo_config["ortho_coeff"] = config.get("ortho_coeff", 0.1)

    return PPO.create(**ppo_config)


def run_config(
    config_name: str,
    config: Dict,
    timesteps: int,
    num_seeds: int,
    num_envs: int,
) -> Dict[str, Dict]:
    """Run a single config on all games."""
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"{'='*60}")

    results = {}

    for game_name in GAMES:
        print(f"  {game_name}...", end=" ", flush=True)

        ppo = create_ppo(game_name, config, timesteps, num_envs)
        keys = jax.random.split(jax.random.PRNGKey(42), num_seeds)

        vmap_train = jax.jit(jax.vmap(PPO.train, in_axes=(None, 0)))

        start = time.time()
        ts, (lengths, returns) = vmap_train(ppo, keys)
        jax.block_until_ready(ts)
        elapsed = time.time() - start

        # Final returns (last eval point)
        final_returns = np.array(returns)[:, -1, :].mean(axis=-1)
        mean_ret = float(final_returns.mean())
        std_ret = float(final_returns.std())

        results[game_name] = {
            "mean": mean_ret,
            "std": std_ret,
            "time_s": elapsed,
        }

        print(f"{mean_ret:6.1f} ± {std_ret:4.1f}  ({elapsed:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Smoke test all baselines")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Steps per game (default: 1M)")
    parser.add_argument("--num-seeds", type=int, default=2,
                        help="Seeds per game (default: 2)")
    parser.add_argument("--num-envs", type=int, default=4096,
                        help="Parallel envs (default: 4096)")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        choices=list(CONFIGS.keys()),
                        help="Configs to run")
    parser.add_argument("--output-dir", type=str, default="results/smoketest",
                        help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("SMOKE TEST: All Baselines on MinAtar (Single-Task)")
    print("=" * 70)
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Seeds: {args.num_seeds}")
    print(f"Envs: {args.num_envs}")
    print(f"Configs: {args.configs}")
    print(f"Games: {list(GAMES.keys())}")
    print("=" * 70)

    total_start = time.time()
    all_results = {}

    for config_name in args.configs:
        config = CONFIGS[config_name]
        results = run_config(
            config_name=config_name,
            config=config,
            timesteps=args.timesteps,
            num_seeds=args.num_seeds,
            num_envs=args.num_envs,
        )
        all_results[config_name] = results

    total_time = time.time() - total_start

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"smoketest_{time.strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump({
            "timesteps": args.timesteps,
            "num_seeds": args.num_seeds,
            "results": all_results,
            "total_time_s": total_time,
        }, f, indent=2)

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Header
    header = f"{'Config':<20}"
    for game in GAMES:
        header += f" {game:>12}"
    header += f" {'Mean':>10}"
    print(header)
    print("-" * 90)

    # Results
    for config_name in args.configs:
        row = f"{config_name:<20}"
        game_means = []
        for game in GAMES:
            r = all_results[config_name][game]
            row += f" {r['mean']:>12.1f}"
            game_means.append(r['mean'])
        row += f" {np.mean(game_means):>10.1f}"
        print(row)

    print("=" * 90)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
