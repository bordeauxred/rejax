"""
Continual learning benchmark for MinAtar games following Lyle et al. NaP methodology.

Compares baseline PPO vs Ortho Optimizer on sequential game training without weight resets.

Usage:
    # Smoke test
    python scripts/bench_continual.py --steps-per-game 100000 --num-cycles 1 --num-seeds 1

    # Full experiment
    python scripts/bench_continual.py --steps-per-game 10000000 --num-cycles 2 --num-seeds 3 --use-wandb

    # Test action mapping
    python scripts/bench_continual.py --test-action-mapping

    # Resume from checkpoint
    python scripts/bench_continual.py --resume checkpoints/continual_baseline_game2.ckpt
"""
import argparse
import json
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization, struct
from flax.training.train_state import TrainState
from gymnax.environments.minatar.breakout import MinBreakout
from gymnax.environments.minatar.asterix import MinAsterix
from gymnax.environments.minatar.space_invaders import MinSpaceInvaders
from gymnax.environments.minatar.freeway import MinFreeway

from rejax import PPO


# MinAtar game configurations
# Note: Seaquest excluded - gymnax implementation is an incomplete stub with dict-style
# access on dataclasses and non-functional step functions (step_divers, step_e_subs, etc.)
MINATAR_GAMES = {
    "Breakout-MinAtar": {"channels": 4, "actions": 3, "env_cls": MinBreakout},
    "Asterix-MinAtar": {"channels": 4, "actions": 5, "env_cls": MinAsterix},
    "SpaceInvaders-MinAtar": {"channels": 6, "actions": 4, "env_cls": MinSpaceInvaders},
    "Freeway-MinAtar": {"channels": 7, "actions": 3, "env_cls": MinFreeway},
}

GAME_ORDER = [
    "Breakout-MinAtar",
    "Asterix-MinAtar",
    "SpaceInvaders-MinAtar",
    "Freeway-MinAtar",
]

# Unified observation space: 7 channels (max from Freeway)
UNIFIED_CHANNELS = 7
# Unified action space: 5 actions (max from Asterix)
UNIFIED_ACTIONS = 5


class PaddedMinAtarEnv:
    """Wrapper that pads observations and unifies action space for MinAtar games."""

    def __init__(self, env, original_channels: int, original_actions: int):
        self._env = env
        self.original_channels = original_channels
        self.original_actions = original_actions

    def __getattr__(self, name):
        if name in ["_env", "original_channels", "original_actions", "reset", "step",
                    "observation_space", "action_space"]:
            return super().__getattribute__(name)
        return getattr(self._env, name)

    @property
    def default_params(self):
        return self._env.default_params

    def observation_space(self, params):
        # Return padded observation space
        return gymnax.environments.spaces.Box(
            low=0.0, high=1.0, shape=(10, 10, UNIFIED_CHANNELS), dtype=jnp.float32
        )

    def action_space(self, params):
        # Return unified action space
        return gymnax.environments.spaces.Discrete(UNIFIED_ACTIONS)

    def reset(self, key, params):
        obs, state = self._env.reset(key, params)
        obs = self._pad_obs(obs)
        return obs, state

    def step(self, key, state, action, params):
        # Map invalid actions to no-op (action 0)
        valid_action = jnp.where(action < self.original_actions, action, 0)
        obs, state, reward, done, info = self._env.step(key, state, valid_action, params)
        obs = self._pad_obs(obs)
        return obs, state, reward, done, info

    def _pad_obs(self, obs):
        """Pad observations from (10, 10, C) to (10, 10, 10)."""
        if self.original_channels < UNIFIED_CHANNELS:
            pad_width = UNIFIED_CHANNELS - self.original_channels
            obs = jnp.pad(obs, ((0, 0), (0, 0), (0, pad_width)))
        return obs.astype(jnp.float32)


def create_padded_env(game_name: str) -> Tuple[PaddedMinAtarEnv, Any]:
    """Create a padded MinAtar environment for the given game."""
    game_info = MINATAR_GAMES[game_name]
    env = game_info["env_cls"]()
    padded_env = PaddedMinAtarEnv(
        env,
        original_channels=game_info["channels"],
        original_actions=game_info["actions"],
    )
    return padded_env, padded_env.default_params


def create_ppo_config(
    env,
    env_params,
    config_name: str,
    total_timesteps: int,
    ortho_mode: Optional[str] = None,
    ortho_coeff: float = 0.1,
    activation: str = "tanh",
    lr_schedule: str = "constant",
    learning_rate: float = 2.5e-4,
    num_envs: int = 2048,
    eval_freq: int = 500_000,
) -> Dict:
    """Create PPO configuration dict."""
    config = {
        "env": env,
        "env_params": env_params,
        "agent_kwargs": {
            "hidden_layer_sizes": (256, 256),
            "activation": activation,
        },
        "num_envs": num_envs,
        "num_steps": 128,
        "num_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": learning_rate,
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

    if ortho_mode and ortho_mode != "none":
        config["ortho_mode"] = ortho_mode
        config["ortho_coeff"] = ortho_coeff

    return config


# Experiment configurations
EXPERIMENT_CONFIGS = [
    {
        "name": "baseline",
        "ortho_mode": None,
        "activation": "tanh",
        "lr_schedule": "constant",
    },
    {
        "name": "ortho_opt",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "groupsort",
        "lr_schedule": "constant",
    },
    {
        "name": "ortho_opt_linear_lr",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "groupsort",
        "lr_schedule": "linear",
    },
]


def save_checkpoint(checkpoint_dir: Path, name: str, train_state, game_idx: int, cycle_idx: int, metadata: Dict):
    """Save checkpoint to disk."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}_cycle{cycle_idx}_game{game_idx}.ckpt"

    checkpoint = {
        "game_idx": game_idx,
        "cycle_idx": cycle_idx,
        "metadata": metadata,
    }

    # Serialize the train_state separately and save alongside metadata
    with open(checkpoint_path, "wb") as f:
        f.write(serialization.to_bytes((train_state, checkpoint)))

    print(f"  Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path, train_state_template):
    """Load checkpoint from disk."""
    with open(checkpoint_path, "rb") as f:
        train_state, checkpoint = serialization.from_bytes(
            (train_state_template, {"game_idx": 0, "cycle_idx": 0, "metadata": {}}),
            f.read()
        )
    return train_state, checkpoint


class ContinualTrainer:
    """Manages continual learning across multiple games."""

    def __init__(
        self,
        config_name: str,
        experiment_config: Dict,
        steps_per_game: int,
        num_cycles: int,
        num_envs: int = 2048,
        eval_freq: int = 500_000,
        checkpoint_dir: Optional[Path] = None,
        use_wandb: bool = False,
        wandb_project: str = "rejax-continual",
    ):
        self.config_name = config_name
        self.experiment_config = experiment_config
        self.steps_per_game = steps_per_game
        self.num_cycles = num_cycles
        self.num_envs = num_envs
        self.eval_freq = eval_freq
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        # Results storage
        self.results = {
            "config_name": config_name,
            "experiment_config": experiment_config,
            "steps_per_game": steps_per_game,
            "num_cycles": num_cycles,
            "games": GAME_ORDER,
            "per_game_results": [],
        }

    def _create_ppo_for_game(self, game_name: str) -> PPO:
        """Create PPO instance configured for a specific game."""
        env, env_params = create_padded_env(game_name)
        config = create_ppo_config(
            env=env,
            env_params=env_params,
            config_name=self.config_name,
            total_timesteps=self.steps_per_game,
            ortho_mode=self.experiment_config.get("ortho_mode"),
            ortho_coeff=self.experiment_config.get("ortho_coeff", 0.1),
            activation=self.experiment_config.get("activation", "tanh"),
            lr_schedule=self.experiment_config.get("lr_schedule", "constant"),
            num_envs=self.num_envs,
            eval_freq=self.eval_freq,
        )
        return PPO.create(**config)

    def _transfer_train_state(self, old_ts, new_ppo, rng):
        """Transfer network weights to a new environment's train state."""
        # Initialize a fresh train state for the new environment
        new_ts = new_ppo.init_state(rng)

        # Transfer actor and critic params (network weights)
        new_ts = new_ts.replace(
            actor_ts=new_ts.actor_ts.replace(
                params=old_ts.actor_ts.params,
                opt_state=old_ts.actor_ts.opt_state,
            ),
            critic_ts=new_ts.critic_ts.replace(
                params=old_ts.critic_ts.params,
                opt_state=old_ts.critic_ts.opt_state,
            ),
            # Preserve global step for tracking
            global_step=old_ts.global_step,
        )
        return new_ts

    def _make_progress_callback(self, game_name: str, cycle_idx: int, original_eval_callback):
        """Create a callback for logging progress."""
        start_time = [time.time()]

        def progress_callback(ppo, train_state, rng):
            lengths, returns = original_eval_callback(ppo, train_state, rng)

            def log(step, returns):
                elapsed = time.time() - start_time[0]
                steps_per_sec = step.item() / elapsed if elapsed > 0 else 0
                mean_ret = returns.mean().item()
                print(f"    [Cycle {cycle_idx} | {game_name}] step={step.item():,} "
                      f"| return={mean_ret:.1f} | {steps_per_sec:,.0f} steps/s")

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        f"cycle_{cycle_idx}/{game_name}/return": mean_ret,
                        f"cycle_{cycle_idx}/{game_name}/step": step.item(),
                        "global_step": train_state.global_step.item(),
                    })

            jax.experimental.io_callback(log, (), train_state.global_step, returns)
            return lengths, returns

        return progress_callback

    def train_single_game(self, game_name: str, train_state, rng, cycle_idx: int):
        """Train on a single game, optionally continuing from existing train_state."""
        ppo = self._create_ppo_for_game(game_name)

        # Add progress callback (wrap the original eval_callback)
        original_eval_callback = ppo.eval_callback
        progress_callback = self._make_progress_callback(game_name, cycle_idx, original_eval_callback)
        ppo = ppo.replace(eval_callback=progress_callback)

        if train_state is not None:
            # Transfer weights from previous game
            rng, init_rng = jax.random.split(rng)
            train_state = self._transfer_train_state(train_state, ppo, init_rng)
        else:
            # Fresh initialization
            rng, init_rng = jax.random.split(rng)
            train_state = ppo.init_state(init_rng)

        # Train
        print(f"  Training on {game_name}...")
        start_time = time.time()

        # Use train_iteration manually to allow step-by-step control
        iteration_steps = ppo.num_envs * ppo.num_steps
        num_iterations = int(np.ceil(self.steps_per_game / iteration_steps))
        eval_interval = int(np.ceil(self.eval_freq / iteration_steps))

        @jax.jit
        def train_chunk(ts, num_iters):
            def body(_, ts):
                return ppo.train_iteration(ts)
            return jax.lax.fori_loop(0, num_iters, body, ts)

        # Run training in chunks with periodic evaluation
        total_iters = 0
        while total_iters < num_iterations:
            chunk_size = min(eval_interval, num_iterations - total_iters)
            train_state = train_chunk(train_state, chunk_size)
            jax.block_until_ready(train_state)

            # Evaluation
            rng, eval_rng = jax.random.split(rng)
            lengths, returns = ppo.eval_callback(ppo, train_state, eval_rng)

            total_iters += chunk_size

        elapsed = time.time() - start_time
        steps_per_sec = self.steps_per_game / elapsed

        # Final evaluation
        rng, eval_rng = jax.random.split(rng)
        lengths, returns = ppo.eval_callback(ppo, train_state, eval_rng)
        final_return = float(returns.mean())

        print(f"  Completed {game_name}: final_return={final_return:.1f}, "
              f"elapsed={elapsed:.1f}s, {steps_per_sec:,.0f} steps/s")

        return train_state, rng, {
            "game": game_name,
            "cycle": cycle_idx,
            "final_return": final_return,
            "elapsed_s": elapsed,
            "steps_per_sec": steps_per_sec,
        }

    def evaluate_all_games(self, train_state, rng, cycle_idx: int, current_game_idx: int):
        """Evaluate current policy on all games."""
        print(f"  Evaluating on all games after game {current_game_idx}...")
        eval_results = {}

        for game_name in GAME_ORDER:
            ppo = self._create_ppo_for_game(game_name)

            # Transfer weights to this game's environment
            rng, eval_rng, init_rng = jax.random.split(rng, 3)
            eval_ts = self._transfer_train_state(train_state, ppo, init_rng)

            # Evaluate
            lengths, returns = ppo.eval_callback(ppo, eval_ts, eval_rng)
            mean_return = float(returns.mean())
            eval_results[game_name] = mean_return
            print(f"    {game_name}: {mean_return:.1f}")

        return eval_results, rng

    def run(self, rng, start_cycle: int = 0, start_game: int = 0, initial_train_state=None):
        """Run the full continual learning experiment."""
        train_state = initial_train_state

        for cycle_idx in range(start_cycle, self.num_cycles):
            print(f"\n{'='*60}")
            print(f"Cycle {cycle_idx + 1}/{self.num_cycles}")
            print(f"{'='*60}")

            game_start = start_game if cycle_idx == start_cycle else 0

            for game_idx, game_name in enumerate(GAME_ORDER[game_start:], start=game_start):
                print(f"\nGame {game_idx + 1}/{len(GAME_ORDER)}: {game_name}")

                # Train on this game
                train_state, rng, game_result = self.train_single_game(
                    game_name, train_state, rng, cycle_idx
                )
                self.results["per_game_results"].append(game_result)

                # Evaluate on all games at game boundaries
                eval_results, rng = self.evaluate_all_games(
                    train_state, rng, cycle_idx, game_idx
                )
                game_result["eval_all_games"] = eval_results

                # Save checkpoint
                if self.checkpoint_dir:
                    save_checkpoint(
                        self.checkpoint_dir,
                        self.config_name,
                        train_state,
                        game_idx,
                        cycle_idx,
                        metadata={"experiment_config": self.experiment_config},
                    )

                # Log to wandb
                if self.use_wandb:
                    import wandb
                    for eval_game, eval_return in eval_results.items():
                        wandb.log({
                            f"eval_after_{game_name}/{eval_game}": eval_return,
                            "game_idx": game_idx,
                            "cycle_idx": cycle_idx,
                        })

            # Reset start_game for subsequent cycles
            start_game = 0

        return self.results


def run_experiment(
    experiment_config: Dict,
    steps_per_game: int,
    num_cycles: int,
    num_seeds: int,
    num_envs: int,
    eval_freq: int,
    checkpoint_dir: Path,
    use_wandb: bool,
    wandb_project: str,
    seed: int = 0,
):
    """Run a single experiment configuration with multiple seeds."""
    config_name = experiment_config["name"]
    print(f"\n{'#'*70}")
    print(f"# Experiment: {config_name}")
    print(f"# Seeds: {num_seeds}, Steps per game: {steps_per_game:,}, Cycles: {num_cycles}")
    print(f"{'#'*70}")

    all_results = []

    for seed_idx in range(num_seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed_idx + 1}/{num_seeds}")
        print(f"{'='*60}")

        rng = jax.random.PRNGKey(seed + seed_idx)

        if use_wandb:
            import wandb
            run_name = f"{config_name}_seed{seed_idx}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "experiment_config": experiment_config,
                    "steps_per_game": steps_per_game,
                    "num_cycles": num_cycles,
                    "seed": seed + seed_idx,
                },
                reinit=True,
            )

        trainer = ContinualTrainer(
            config_name=f"{config_name}_seed{seed_idx}",
            experiment_config=experiment_config,
            steps_per_game=steps_per_game,
            num_cycles=num_cycles,
            num_envs=num_envs,
            eval_freq=eval_freq,
            checkpoint_dir=checkpoint_dir / config_name,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )

        results = trainer.run(rng)
        all_results.append(results)

        if use_wandb:
            import wandb
            wandb.finish()

    return all_results


def test_action_mapping():
    """Test that invalid actions map to no-op correctly."""
    print("Testing action mapping for each game...")

    for game_name in GAME_ORDER:
        game_info = MINATAR_GAMES[game_name]
        env, params = create_padded_env(game_name)

        print(f"\n{game_name}:")
        print(f"  Original actions: {game_info['actions']}")
        print(f"  Unified actions: {UNIFIED_ACTIONS}")

        # Test each action
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng, params)
        print(f"  Obs shape: {obs.shape}")

        for action in range(UNIFIED_ACTIONS):
            rng, step_rng = jax.random.split(rng)
            try:
                obs, state, reward, done, info = env.step(step_rng, state, action, params)
                valid = action < game_info['actions']
                print(f"  Action {action}: {'valid' if valid else 'mapped to no-op'}")
            except Exception as e:
                print(f"  Action {action}: ERROR - {e}")

    print("\nAction mapping test complete!")


def save_results(results: Dict, output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"continual_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Continual learning benchmark for MinAtar")
    parser.add_argument("--steps-per-game", type=int, default=10_000_000,
                        help="Training steps per game")
    parser.add_argument("--num-cycles", type=int, default=2,
                        help="Number of cycles through all games")
    parser.add_argument("--num-seeds", type=int, default=3,
                        help="Number of random seeds")
    parser.add_argument("--num-envs", type=int, default=2048,
                        help="Parallel environments")
    parser.add_argument("--eval-freq", type=int, default=500_000,
                        help="Evaluation frequency in steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/continual",
                        help="Directory for checkpoints")
    parser.add_argument("--output-dir", type=str, default="results/continual",
                        help="Directory for results JSON")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="rejax-continual",
                        help="W&B project name")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint for resuming")
    parser.add_argument("--test-action-mapping", action="store_true",
                        help="Test action mapping and exit")
    parser.add_argument("--configs", nargs="+",
                        default=["baseline", "ortho_opt", "ortho_opt_linear_lr"],
                        choices=["baseline", "ortho_opt", "ortho_opt_linear_lr"],
                        help="Experiment configurations to run")

    args = parser.parse_args()

    if args.test_action_mapping:
        test_action_mapping()
        return

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    # Filter configs
    configs_to_run = [c for c in EXPERIMENT_CONFIGS if c["name"] in args.configs]

    all_experiment_results = {}

    for experiment_config in configs_to_run:
        results = run_experiment(
            experiment_config=experiment_config,
            steps_per_game=args.steps_per_game,
            num_cycles=args.num_cycles,
            num_seeds=args.num_seeds,
            num_envs=args.num_envs,
            eval_freq=args.eval_freq,
            checkpoint_dir=checkpoint_dir,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            seed=args.seed,
        )
        all_experiment_results[experiment_config["name"]] = results

    # Save all results
    save_results(all_experiment_results, output_dir)

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    for config_name, results_list in all_experiment_results.items():
        print(f"\n{config_name}:")
        for seed_idx, results in enumerate(results_list):
            final_returns = [r["final_return"] for r in results["per_game_results"]]
            print(f"  Seed {seed_idx}: mean_return={np.mean(final_returns):.1f}")


if __name__ == "__main__":
    main()
