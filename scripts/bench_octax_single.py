"""
Single-task benchmark for Octax (CHIP-8 arcade games in JAX).

Tests PPO on individual Octax games with various MLP configurations.
Follows the Octax paper methodology (arXiv 2510.01764).

Usage:
    # Throughput test
    python scripts/bench_octax_single.py --mode throughput

    # Single game smoke test
    python scripts/bench_octax_single.py --game brix --steps 100000 --num-seeds 1

    # Full benchmark
    python scripts/bench_octax_single.py --mode full --steps 5000000 --use-wandb
"""
import argparse
import json
import time
import warnings
from copy import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from rejax import PPO
from rejax.compat.octax2gymnax import create_octax


# =============================================================================
# Octax Game Configurations
# =============================================================================

# Games selected for benchmark (representative of different categories)
OCTAX_GAMES = {
    # Puzzle
    "tetris": {"actions": 5},
    # Action/Arcade
    "brix": {"actions": 3},
    "pong": {"actions": 3},
    # Strategy
    "tank": {"actions": 6},
    # Exploration/Shooter
    "spacejam": {"actions": 5},
    "deep": {"actions": 4},
}

# Unified action space (max across games)
UNIFIED_ACTIONS = 6

# Default game order for benchmarks
GAME_ORDER = ["tetris", "brix", "tank", "spacejam", "deep"]


# =============================================================================
# Unified Octax Environment Wrapper
# =============================================================================

class UnifiedOctaxEnv:
    """Wrapper that unifies action space across Octax games.

    For continual learning, all tasks must have the same input/output dimensions.
    Octax observations are already unified to (64, 32, 4) by the HWCObsWrapper.
    This wrapper additionally:
    - Maps invalid actions to no-op (action 0)
    - Provides consistent action_space across games
    """

    def __init__(self, env, game_name: str, original_actions: int):
        self._env = env
        self.game_name = game_name
        self.original_actions = original_actions

    def __getattr__(self, name):
        if name in ["_env", "game_name", "original_actions",
                    "reset", "step", "observation_space", "action_space"]:
            return super().__getattribute__(name)
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
        valid_action = jnp.where(action < self.original_actions, action, 0)
        return self._env.step(key, state, valid_action, params)

    @property
    def num_actions(self) -> int:
        return UNIFIED_ACTIONS

    @property
    def name(self) -> str:
        return self._env.name

    def __deepcopy__(self, memo):
        warnings.warn(
            f"Trying to deepcopy {type(self).__name__}, which contains an octax env. "
            "Octax envs may throw an error when deepcopying, so a shallow copy is returned.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return copy(self)


def create_unified_env(game_name: str) -> Tuple[UnifiedOctaxEnv, Any]:
    """Create a unified Octax environment for the given game."""
    game_info = OCTAX_GAMES[game_name]
    env, env_params = create_octax(game_name)
    unified_env = UnifiedOctaxEnv(env, game_name, game_info["actions"])
    return unified_env, env_params


# =============================================================================
# PPO Configuration for Octax
# =============================================================================

def create_ppo_config(
    env,
    env_params,
    total_timesteps: int,
    num_envs: int = 2048,
    num_steps: int = 32,  # Octax paper default
    num_epochs: int = 4,
    num_minibatches: int = 32,
    learning_rate: float = 5e-4,  # Octax paper default
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    mlp_hidden_sizes: Tuple[int, ...] = (256, 256, 256, 256),
    activation: str = "relu",
    normalize_observations: bool = False,
    normalize_rewards: bool = False,
    eval_freq: int = 250_000,
    # AdaMo options
    ortho_mode: Optional[str] = None,
    ortho_coeff: float = 0.1,
    l2_init_coeff: Optional[float] = None,
    nap_enabled: bool = False,
    scale_enabled: bool = False,
    scale_reg_coeff: float = 0.01,
) -> Dict:
    """Create PPO config for Octax environments.

    Uses Octax paper defaults:
    - lr=5e-4
    - num_steps=32
    - 4 epochs, 32 minibatches
    - 3-conv CNN with configurable MLP head
    """
    config = {
        "env": env,
        "env_params": env_params,
        "agent_kwargs": {
            "mlp_hidden_sizes": mlp_hidden_sizes,
            "activation": activation,
            "use_bias": True if ortho_mode is None else False,
            "use_orthogonal_init": True,
        },
        "num_envs": num_envs,
        "num_steps": num_steps,
        "num_epochs": num_epochs,
        "num_minibatches": num_minibatches,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_eps": clip_eps,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "normalize_observations": normalize_observations,
        "normalize_rewards": normalize_rewards,
        "skip_initial_evaluation": True,
        "discrete_cnn_type": "octax",  # Use OctaxCNN architecture
    }

    # AdaMo regularization
    if ortho_mode and ortho_mode != "none":
        config["ortho_mode"] = ortho_mode
        config["ortho_coeff"] = ortho_coeff
        config["agent_kwargs"]["use_bias"] = False
        if activation == "relu":
            config["agent_kwargs"]["activation"] = "groupsort"

    if l2_init_coeff is not None:
        config["l2_init_coeff"] = l2_init_coeff

    if nap_enabled:
        config["nap_enabled"] = True

    if scale_enabled:
        config["scale_enabled"] = True
        config["scale_reg_coeff"] = scale_reg_coeff

    return config


# =============================================================================
# Experiment Configurations
# =============================================================================

EXPERIMENT_CONFIGS = {
    # Paper-style: 4x256 MLP head
    "256x4": {
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "activation": "relu",
    },
    # Smaller: 4x64 MLP head (faster, more plasticity stress)
    "64x4": {
        "mlp_hidden_sizes": (64, 64, 64, 64),
        "activation": "relu",
    },
    # AdaMo variants
    "adamo_256x4": {
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "activation": "groupsort",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
    },
    "adamo_64x4": {
        "mlp_hidden_sizes": (64, 64, 64, 64),
        "activation": "groupsort",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
    },
    # Plasticity baselines
    "l2_init_256x4": {
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "activation": "relu",
        "l2_init_coeff": 0.001,
    },
    "nap_256x4": {
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "activation": "relu",
        "nap_enabled": True,
    },
    "scale_adamo_256x4": {
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "activation": "groupsort",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "scale_enabled": True,
        "scale_reg_coeff": 0.01,
    },
}


# =============================================================================
# Training Functions
# =============================================================================

def run_throughput_test(
    games: list = None,
    num_envs_list: list = None,
    steps: int = 100_000,
    output_dir: Path = None,
):
    """Test throughput across different env counts."""
    games = games or ["brix"]
    num_envs_list = num_envs_list or [512, 1024, 2048, 4096, 8192]

    print("\n" + "=" * 60)
    print("THROUGHPUT TEST")
    print("=" * 60)

    results = []

    for game in games:
        print(f"\nGame: {game}")
        env, env_params = create_unified_env(game)

        for num_envs in num_envs_list:
            print(f"  num_envs={num_envs}...", end=" ", flush=True)

            config = create_ppo_config(
                env=env,
                env_params=env_params,
                total_timesteps=steps,
                num_envs=num_envs,
                eval_freq=steps,  # Only eval at end
                mlp_hidden_sizes=(64, 64, 64, 64),  # Small for speed
            )
            ppo = PPO.create(**config)

            # Compile
            rng = jax.random.PRNGKey(0)
            compile_start = time.time()
            ts, _ = PPO.train(ppo, rng)
            jax.block_until_ready(ts)
            compile_time = time.time() - compile_start

            # Benchmark
            start = time.time()
            ts, _ = PPO.train(ppo, rng)
            jax.block_until_ready(ts)
            runtime = time.time() - start

            steps_per_sec = steps / runtime
            print(f"{steps_per_sec:,.0f} steps/s (compile: {compile_time:.1f}s)")

            results.append({
                "game": game,
                "num_envs": num_envs,
                "steps": steps,
                "compile_time": compile_time,
                "runtime": runtime,
                "steps_per_sec": steps_per_sec,
            })

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "throughput.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def run_single_task(
    game_name: str,
    steps: int,
    num_seeds: int = 3,
    num_envs: int = 2048,
    config_name: str = "256x4",
    normalize_rewards: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "rejax-octax-single",
    output_dir: Path = None,
    seed: int = 0,
):
    """Run PPO on a single Octax game with multiple seeds."""
    experiment_config = EXPERIMENT_CONFIGS[config_name]

    print(f"\n{'='*60}")
    print(f"Single task: {game_name}")
    print(f"Config: {config_name}, Steps: {steps:,}, Seeds: {num_seeds}")
    print(f"Normalize rewards: {normalize_rewards}")
    print(f"{'='*60}")

    env, env_params = create_unified_env(game_name)

    eval_freq = max(steps // 20, 10000)
    ppo_config = create_ppo_config(
        env=env,
        env_params=env_params,
        total_timesteps=steps,
        num_envs=num_envs,
        eval_freq=eval_freq,
        normalize_rewards=normalize_rewards,
        **experiment_config,
    )
    ppo = PPO.create(**ppo_config)

    # Initialize wandb
    if use_wandb:
        import wandb
        norm_str = "norm" if normalize_rewards else "raw"
        wandb.init(
            project=wandb_project,
            name=f"{game_name}_{config_name}_{norm_str}",
            config={
                "game": game_name,
                "steps": steps,
                "num_seeds": num_seeds,
                "num_envs": num_envs,
                "config_name": config_name,
                "normalize_rewards": normalize_rewards,
                **experiment_config,
            },
            reinit=True,
        )

    # Compile with vmapped training
    print("Compiling...", end=" ", flush=True)
    compile_start = time.time()
    keys = jax.random.split(jax.random.PRNGKey(seed), num_seeds)
    vmap_train = jax.jit(jax.vmap(PPO.train, in_axes=(None, 0)))
    train_states, _ = vmap_train(ppo, keys)
    jax.block_until_ready(train_states)
    compile_time = time.time() - compile_start
    print(f"{compile_time:.1f}s")

    # Benchmark
    print("Running...", end=" ", flush=True)
    start = time.time()
    train_states, eval_data = vmap_train(ppo, keys)
    jax.block_until_ready(train_states)
    runtime = time.time() - start

    _, returns = eval_data
    # returns: (num_seeds, num_evals, num_episodes) or (num_seeds, num_episodes)
    if returns.ndim == 3:
        final_returns = returns[:, -1, :].mean(axis=-1)
    else:
        final_returns = returns.mean(axis=-1)

    steps_per_sec = steps * num_seeds / runtime
    steps_per_sec_per_seed = steps_per_sec / num_seeds

    print(f"done")
    print(f"\nResults:")
    print(f"  Steps/sec: {steps_per_sec:,.0f} ({steps_per_sec_per_seed:,.0f} per seed)")
    print(f"  Final returns: {final_returns.mean():.1f} +/- {final_returns.std():.1f}")
    print(f"  Per-seed: {[f'{r:.1f}' for r in final_returns]}")

    if use_wandb:
        import wandb
        wandb.log({
            "final/mean_return": float(final_returns.mean()),
            "final/std_return": float(final_returns.std()),
            "perf/steps_per_sec": steps_per_sec,
            "perf/compile_time": compile_time,
            "perf/runtime": runtime,
        })
        wandb.finish()

    result = {
        "game": game_name,
        "config": config_name,
        "steps": steps,
        "num_seeds": num_seeds,
        "normalize_rewards": normalize_rewards,
        "compile_time": compile_time,
        "runtime": runtime,
        "steps_per_sec": steps_per_sec,
        "final_returns": final_returns.tolist(),
        "mean_return": float(final_returns.mean()),
        "std_return": float(final_returns.std()),
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        norm_str = "norm" if normalize_rewards else "raw"
        with open(output_dir / f"{game_name}_{config_name}_{norm_str}.json", "w") as f:
            json.dump(result, f, indent=2)

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Octax single-task benchmark")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["throughput", "single", "full"],
                        help="Benchmark mode")
    parser.add_argument("--game", type=str, default="brix",
                        choices=list(OCTAX_GAMES.keys()),
                        help="Game to train on (for single mode)")
    parser.add_argument("--steps", type=int, default=5_000_000,
                        help="Training steps")
    parser.add_argument("--num-seeds", type=int, default=3,
                        help="Number of random seeds")
    parser.add_argument("--num-envs", type=int, default=2048,
                        help="Parallel environments")
    parser.add_argument("--config", type=str, default="256x4",
                        choices=list(EXPERIMENT_CONFIGS.keys()),
                        help="Experiment configuration")
    parser.add_argument("--normalize-rewards", action="store_true",
                        help="Enable reward normalization")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="rejax-octax-single",
                        help="W&B project name")
    parser.add_argument("--output-dir", type=str, default="results/octax_single",
                        help="Directory for results")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.mode == "throughput":
        run_throughput_test(output_dir=output_dir)

    elif args.mode == "single":
        run_single_task(
            game_name=args.game,
            steps=args.steps,
            num_seeds=args.num_seeds,
            num_envs=args.num_envs,
            config_name=args.config,
            normalize_rewards=args.normalize_rewards,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            output_dir=output_dir,
            seed=args.seed,
        )

    elif args.mode == "full":
        # Run all games with all configs
        games = GAME_ORDER
        configs = ["64x4", "256x4"]
        reward_modes = [False, True]  # raw, normalized

        all_results = []
        for game in games:
            for config in configs:
                for normalize in reward_modes:
                    result = run_single_task(
                        game_name=game,
                        steps=args.steps,
                        num_seeds=args.num_seeds,
                        num_envs=args.num_envs,
                        config_name=config,
                        normalize_rewards=normalize,
                        use_wandb=args.use_wandb,
                        wandb_project=args.wandb_project,
                        output_dir=output_dir,
                        seed=args.seed,
                    )
                    all_results.append(result)

        # Save combined results
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
