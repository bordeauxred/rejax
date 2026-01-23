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
# CleanRL MinAtar CNN: conv(16, k=3, VALID) -> flatten -> dense(128)
CONFIGS = {
    # CNN baselines FIRST (CleanRL MinAtar style)
    "cnn_baseline": {
        "network_type": "cnn",
        "conv_channels": 16,  # CleanRL MinAtar
        "mlp_hidden_size": 128,  # CleanRL MinAtar
        "kernel_size": 3,
        "activation": "relu",
        "use_orthogonal_init": True,  # CleanRL uses ortho init
        "use_bias": True,
        "ortho_mode": None,
        "learning_rate": 2.5e-4,  # CleanRL default
    },
    "cnn_adamo": {
        "network_type": "cnn",
        "conv_channels": 16,
        "mlp_hidden_size": 128,
        "kernel_size": 3,
        "activation": "relu",
        "use_orthogonal_init": False,  # AdaMO handles orthogonality
        "use_bias": False,
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "learning_rate": 2.5e-4,
    },
    "cnn_adamo_lyle": {
        "network_type": "cnn",
        "conv_channels": 16,
        "mlp_hidden_size": 128,
        "kernel_size": 3,
        "activation": "relu",
        "use_orthogonal_init": False,
        "use_bias": False,
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "learning_rate": 6.25e-5,  # Lyle LR
    },
    # MLP baselines (4x256, Lyle et al. style)
    # "mlp_baseline": {
    #     "network_type": "mlp",
    #     "hidden_layer_sizes": (256, 256, 256, 256),
    #     "activation": "tanh",
    #     "use_orthogonal_init": True,
    #     "use_bias": True,
    #     "ortho_mode": None,
    #     "learning_rate": 2.5e-4,
    # },
    # "mlp_adamo": {
    #     "network_type": "mlp",
    #     "hidden_layer_sizes": (256, 256, 256, 256),
    #     "activation": "groupsort",
    #     "use_orthogonal_init": False,
    #     "use_bias": False,
    #     "ortho_mode": "optimizer",
    #     "ortho_coeff": 0.1,
    #     "learning_rate": 2.5e-4,
    # },
    # "mlp_adamo_lyle": {
    #     "network_type": "mlp",
    #     "hidden_layer_sizes": (256, 256, 256, 256),
    #     "activation": "groupsort",
    #     "use_orthogonal_init": False,
    #     "use_bias": False,
    #     "ortho_mode": "optimizer",
    #     "ortho_coeff": 0.1,
    #     "learning_rate": 6.25e-5,
    # },
}


def create_ppo(game_name: str, config: Dict, timesteps: int, num_envs: int, eval_freq: int) -> PPO:
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
        "eval_freq": eval_freq,
        "skip_initial_evaluation": False,
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
    default_num_envs: int,
    eval_freq: int,
) -> Dict[str, Dict]:
    """Run a single config on all games."""
    # Use per-config num_envs if specified, else default
    num_envs = config.get("num_envs", default_num_envs)

    print(f"\n{'='*60}")
    print(f"Config: {config_name} (num_envs={num_envs})")
    print(f"{'='*60}")

    results = {}

    for game_name in GAMES:
        print(f"  {game_name}...", end=" ", flush=True)

        ppo = create_ppo(game_name, config, timesteps, num_envs, eval_freq)
        keys = jax.random.split(jax.random.PRNGKey(42), num_seeds)

        vmap_train = jax.jit(jax.vmap(PPO.train, in_axes=(None, 0)))

        start = time.time()
        ts, (lengths, returns) = vmap_train(ppo, keys)
        jax.block_until_ready(ts)
        elapsed = time.time() - start

        returns_np = np.array(returns)  # (seeds, evals, episodes)

        # First and final returns to show learning
        first_returns = returns_np[:, 0, :].mean(axis=-1)
        final_returns = returns_np[:, -1, :].mean(axis=-1)

        first_mean = float(first_returns.mean())
        final_mean = float(final_returns.mean())
        final_std = float(final_returns.std())

        # Check if learning happened
        improved = final_mean > first_mean * 1.5  # >50% improvement
        marker = "✓" if improved else "?"

        results[game_name] = {
            "first": first_mean,
            "final": final_mean,
            "std": final_std,
            "improved": improved,
            "time_s": elapsed,
        }

        print(f"{first_mean:5.1f} → {final_mean:6.1f} ± {final_std:4.1f}  ({elapsed:.1f}s) {marker}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Smoke test all baselines")
    parser.add_argument("--timesteps", type=int, default=5_000_000,
                        help="Steps per game (default: 5M for meaningful signal)")
    parser.add_argument("--num-seeds", type=int, default=2,
                        help="Seeds per game (default: 2)")
    parser.add_argument("--num-envs", type=int, default=2048,
                        help="Parallel envs (default: 2048, CNN needs this to avoid OOM)")
    parser.add_argument("--eval-freq", type=int, default=500_000,
                        help="Eval frequency (default: 500k)")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        choices=list(CONFIGS.keys()),
                        help="Configs to run")
    parser.add_argument("--output-dir", type=str, default="results/smoketest",
                        help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("SMOKE TEST: All Baselines on MinAtar (Single-Task)")
    print("=" * 70)
    print(f"Timesteps: {args.timesteps:,} (pgx uses 20M, we use {args.timesteps/20_000_000*100:.0f}%)")
    print(f"Seeds: {args.num_seeds}")
    print(f"Envs: {args.num_envs}")
    print(f"Eval freq: {args.eval_freq:,}")
    print(f"Configs: {args.configs}")
    print(f"Games: {list(GAMES.keys())}")
    print("=" * 70)
    print("Output format: first_eval → final_eval ± std  (time) ✓=learning")
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
            default_num_envs=args.num_envs,
            eval_freq=args.eval_freq,
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
            "eval_freq": args.eval_freq,
            "results": all_results,
            "total_time_s": total_time,
        }, f, indent=2)

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY (final returns)")
    print("=" * 100)

    # Header
    header = f"{'Config':<20}"
    for game in GAMES:
        header += f" {game:>12}"
    header += f" {'Mean':>10} {'Learn?':>8}"
    print(header)
    print("-" * 100)

    # Results
    for config_name in args.configs:
        row = f"{config_name:<20}"
        game_means = []
        num_improved = 0
        for game in GAMES:
            r = all_results[config_name][game]
            row += f" {r['final']:>12.1f}"
            game_means.append(r['final'])
            if r['improved']:
                num_improved += 1
        row += f" {np.mean(game_means):>10.1f}"
        row += f" {num_improved}/5"
        print(row)

    print("=" * 100)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {output_file}")
    print()
    print("pgx reference scores at 20M steps:")
    print("  Breakout ~50, Asterix ~25, SpaceInvaders ~150, Freeway ~60, Seaquest ~60")


if __name__ == "__main__":
    main()
