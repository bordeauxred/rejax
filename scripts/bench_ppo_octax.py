#!/usr/bin/env python3
"""
Benchmark PPOOctax (shared backbone) on Octax games.

This uses our rejax PPOOctax implementation which matches the paper architecture:
- Shared CNN backbone between actor and critic
- Combined loss with single backward pass
- optax.clip (element-wise) gradient clipping

Usage:
    # Quick test
    uv run python scripts/bench_ppo_octax.py --game brix --steps 100000 --seeds 1

    # Match paper config
    uv run python scripts/bench_ppo_octax.py --game brix --steps 5000000 --seeds 2 --mlp 256

    # Deep MLP for plasticity research
    uv run python scripts/bench_ppo_octax.py --game brix --steps 5000000 --seeds 2 --mlp 256x4

    # With reward normalization (for continual learning)
    uv run python scripts/bench_ppo_octax.py --game brix --steps 5000000 --seeds 2 --normalize-rewards
"""
import argparse
import time

import jax
import jax.numpy as jnp

from rejax import PPOOctax
from rejax.evaluate import evaluate
from rejax.compat.octax2gymnax import create_octax


def run_ppo_octax(
    game: str,
    num_seeds: int = 2,
    total_timesteps: int = 5_000_000,
    mlp_hidden_sizes: tuple = (256,),
    num_envs: int = 512,
    num_epochs: int = 4,
    normalize_rewards: bool = False,
    normalize_observations: bool = False,
):
    """Run our PPOOctax implementation."""
    print(f"\n{'='*60}")
    print(f"Running Rejax PPOOctax on {game}")
    print(f"Seeds: {num_seeds}, Steps: {total_timesteps:,}")
    print(f"MLP: {mlp_hidden_sizes}, Envs: {num_envs}, Epochs: {num_epochs}")
    print(f"Normalize rewards: {normalize_rewards}, obs: {normalize_observations}")
    print(f"{'='*60}")

    # Create environment
    env, env_params = create_octax(game)

    # Config matching paper
    config = {
        "env": env,
        "env_params": env_params,
        "num_envs": num_envs,
        "num_steps": 32,
        "num_epochs": num_epochs,
        "num_minibatches": 32,
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": total_timesteps,
        "eval_freq": 250_000,
        "skip_initial_evaluation": True,
        "normalize_rewards": normalize_rewards,
        "normalize_observations": normalize_observations,
        "agent_kwargs": {
            "mlp_hidden_sizes": mlp_hidden_sizes,
            "activation": "relu",
        },
    }

    # Create algorithm
    algo = PPOOctax.create(**config)

    # Train with vmap over seeds
    print("Compiling...")
    vmap_train = jax.jit(jax.vmap(algo.train))
    keys = jax.random.split(jax.random.PRNGKey(0), num_seeds)

    start = time.time()
    train_states, metrics = vmap_train(keys)
    jax.block_until_ready(train_states)
    elapsed = time.time() - start

    print(f"Done in {elapsed:.1f}s")
    print(f"Steps/sec: {total_timesteps * num_seeds / elapsed:,.0f}")

    # Evaluate
    all_returns = []
    for seed_idx in range(num_seeds):
        ts_for_seed = jax.tree.map(lambda x: x[seed_idx], train_states)
        lengths, returns = evaluate(
            algo.make_act(ts_for_seed),
            jax.random.PRNGKey(seed_idx + 1000),
            env,
            env_params,
            128,
        )
        mean_return = float(returns.mean())
        all_returns.append(mean_return)
        print(f"  Seed {seed_idx}: return = {mean_return:.1f}")

    mean_return = sum(all_returns) / len(all_returns)
    print(f"\nMean return: {mean_return:.1f}")
    return all_returns, mean_return


def main():
    parser = argparse.ArgumentParser(description="Benchmark PPOOctax on Octax games")
    parser.add_argument("--game", default="brix", help="Game name (brix, pong, tetris, tank)")
    parser.add_argument("--seeds", type=int, default=2, help="Number of seeds")
    parser.add_argument("--steps", type=int, default=5_000_000, help="Total timesteps")
    parser.add_argument("--mlp", default="256", help="MLP config: '256' or '256x4'")
    parser.add_argument("--envs", type=int, default=512, help="Number of parallel envs")
    parser.add_argument("--epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--normalize-rewards", action="store_true", help="Enable reward normalization")
    parser.add_argument("--normalize-observations", action="store_true", help="Enable observation normalization")
    args = parser.parse_args()

    # Parse MLP config
    if "x" in args.mlp:
        width, depth = args.mlp.split("x")
        mlp_hidden_sizes = tuple([int(width)] * int(depth))
    else:
        mlp_hidden_sizes = (int(args.mlp),)

    run_ppo_octax(
        game=args.game,
        num_seeds=args.seeds,
        total_timesteps=args.steps,
        mlp_hidden_sizes=mlp_hidden_sizes,
        num_envs=args.envs,
        num_epochs=args.epochs,
        normalize_rewards=args.normalize_rewards,
        normalize_observations=args.normalize_observations,
    )


if __name__ == "__main__":
    main()
