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

    # With unified action space (for continual learning compatibility)
    uv run python scripts/bench_ppo_octax.py --game brix --steps 5000000 --seeds 2 --unified
"""
import argparse
import time
import warnings
from copy import copy

import jax
import jax.numpy as jnp

from rejax import PPOOctax
from rejax.evaluate import evaluate
from rejax.compat.octax2gymnax import create_octax

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# All octax games with their native action counts
OCTAX_GAMES = {
    "airplane": 2,
    "blinky": 5,
    "brix": 3,
    "deep": 4,
    "filter": 3,
    "flight_runner": 5,
    "missile": 2,
    "pong": 3,
    "rocket": 2,
    "shooting_stars": 5,
    "spacejam": 5,
    "squash": 3,
    "submarine": 2,
    "tank": 6,
    "tetris": 5,
    "ufo": 4,
    "vertical_brix": 3,
    "wipe_off": 3,
    "worm": 5,
}
UNIFIED_ACTIONS = 6  # Max across all games


class UnifiedOctaxEnv:
    """Wrapper that unifies action space across Octax games for continual learning.

    Maps invalid actions (>= game's native actions) to no-op (action 0).
    """
    def __init__(self, env, game_name: str):
        object.__setattr__(self, '_env', env)
        object.__setattr__(self, 'game_name', game_name)
        object.__setattr__(self, 'native_actions', OCTAX_GAMES[game_name])

    def __getattr__(self, name):
        if name in ('_env', 'game_name', 'native_actions'):
            return object.__getattribute__(self, name)
        return getattr(self._env, name)

    @property
    def default_params(self):
        return self._env.default_params

    def observation_space(self, params):
        return self._env.observation_space(params)

    def action_space(self, params):
        from gymnax.environments import spaces
        return spaces.Discrete(UNIFIED_ACTIONS)

    def reset(self, key, params):
        return self._env.reset(key, params)

    def step(self, key, state, action, params):
        # Map invalid actions to no-op (action 0)
        valid_action = jnp.where(action < self.native_actions, action, 0)
        return self._env.step(key, state, valid_action, params)

    def __deepcopy__(self, memo):
        # Return self - octax envs don't need deepcopy, they're stateless
        return self

    def __copy__(self):
        return self


def run_ppo_octax(
    game: str,
    num_seeds: int = 2,
    total_timesteps: int = 5_000_000,
    mlp_hidden_sizes: tuple = (256,),
    num_envs: int = 512,
    num_epochs: int = 4,
    normalize_rewards: bool = False,
    unified: bool = False,
    use_wandb: bool = False,
):
    """Run our PPOOctax implementation."""
    # Format MLP string for logging
    mlp_str = f"{mlp_hidden_sizes[0]}x{len(mlp_hidden_sizes)}"

    print(f"\n{'='*60}")
    print(f"Running Rejax PPOOctax on {game}")
    print(f"Seeds: {num_seeds}, Steps: {total_timesteps:,}")
    print(f"MLP: {mlp_hidden_sizes}, Envs: {num_envs}, Epochs: {num_epochs}")
    if normalize_rewards:
        print("Reward normalization: ENABLED")
    if unified:
        print(f"Unified action space: {UNIFIED_ACTIONS} (native: {OCTAX_GAMES[game]})")
    print(f"{'='*60}")

    # Initialize wandb
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: wandb not installed, skipping logging")
            use_wandb = False
        else:
            wandb.init(
                project="octax-single-task",
                name=f"{game}_{mlp_str}_norm{int(normalize_rewards)}",
                config={
                    "game": game,
                    "mlp_hidden_sizes": mlp_hidden_sizes,
                    "mlp_str": mlp_str,
                    "num_seeds": num_seeds,
                    "total_timesteps": total_timesteps,
                    "num_envs": num_envs,
                    "num_epochs": num_epochs,
                    "normalize_rewards": normalize_rewards,
                    "unified": unified,
                    "algorithm": "PPOOctax",
                },
            )

    # Create environment
    env, env_params = create_octax(game)
    if unified:
        env = UnifiedOctaxEnv(env, game)

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

    steps_per_sec = total_timesteps * num_seeds / elapsed
    print(f"Done in {elapsed:.1f}s")
    print(f"Steps/sec: {steps_per_sec:,.0f}")

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
    std_return = (sum((r - mean_return) ** 2 for r in all_returns) / len(all_returns)) ** 0.5
    print(f"\nMean return: {mean_return:.1f} ± {std_return:.1f}")

    # Log to wandb
    if use_wandb:
        wandb.log({
            "eval/mean_return": mean_return,
            "eval/std_return": std_return,
            "eval/min_return": min(all_returns),
            "eval/max_return": max(all_returns),
            "throughput/steps_per_sec": steps_per_sec,
            "throughput/total_time_sec": elapsed,
            "throughput/time_per_1M_steps": elapsed / (total_timesteps * num_seeds / 1_000_000),
        })
        for i, ret in enumerate(all_returns):
            wandb.log({f"eval/seed_{i}_return": ret})
        wandb.finish()

    # Print throughput summary
    print(f"Throughput: {steps_per_sec:,.0f} steps/sec | {elapsed / (total_timesteps * num_seeds / 1_000_000):.1f}s per 1M steps")

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
    parser.add_argument("--unified", action="store_true", help="Use unified action space (6 actions)")
    parser.add_argument("--use-wandb", action="store_true", help="Log to Weights & Biases")
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
        unified=args.unified,
        use_wandb=args.use_wandb,
    )


if __name__ == "__main__":
    main()
