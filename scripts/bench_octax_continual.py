#!/usr/bin/env python3
"""
Continual learning benchmark for Octax (CHIP-8 arcade games in JAX).

Tests AdaMo and baselines on sequential Octax game training without weight resets.

Config: 8 games x 3 cycles, 64x4 MLP, 2 seeds
Games: brix, submarine, filter, tank, blinky, missile, ufo, wipe_off

Usage:
    # Smoke test (verifies full pipeline)
    python scripts/bench_octax_continual.py --smoke-test

    # Full experiment (all 4 configs, ~15 hours)
    python scripts/bench_octax_continual.py --output-dir results/octax_continual

    # Single config
    python scripts/bench_octax_continual.py --configs baseline --output-dir results/octax_continual
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from rejax import PPOOctax
from rejax.evaluate import evaluate
from rejax.compat.octax2gymnax import create_octax
from rejax.regularization import compute_gram_regularization_loss, compute_l2_init_loss


# =============================================================================
# Octax Game Configurations
# =============================================================================

# All learnable games (return > 5 in single-task experiments)
OCTAX_GAMES = {
    "brix": 3,
    "submarine": 2,
    "filter": 3,
    "tank": 6,
    "blinky": 5,
    "missile": 2,
    "ufo": 4,
    "wipe_off": 3,
}

# Task order for continual learning (8 games, diverse returns and action spaces)
TASK_ORDER = ["brix", "submarine", "filter", "tank", "blinky", "missile", "ufo", "wipe_off"]

# Unified action space (max across games = tank with 6 actions)
UNIFIED_ACTIONS = 6


# =============================================================================
# Experiment Configurations
# =============================================================================

EXPERIMENT_CONFIGS = {
    # Baseline: standard PPO with ReLU
    "baseline": {
        "agent_kwargs": {"mlp_hidden_sizes": (64, 64, 64, 64), "activation": "relu"},
        "ortho_mode": None,
        "normalize_rewards": False,
    },
    # AdaMo with GroupSort activation (1-Lipschitz)
    "adamo_groupsort": {
        "agent_kwargs": {"mlp_hidden_sizes": (64, 64, 64, 64), "activation": "groupsort"},
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "normalize_rewards": False,
    },
    # AdaMo with ReLU (ablation: ortho without 1-Lipschitz)
    "adamo_relu": {
        "agent_kwargs": {"mlp_hidden_sizes": (64, 64, 64, 64), "activation": "relu"},
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "normalize_rewards": False,
    },
    # AdaMo + GroupSort + reward normalization (ablation)
    "adamo_groupsort_norm": {
        "agent_kwargs": {"mlp_hidden_sizes": (64, 64, 64, 64), "activation": "groupsort"},
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "normalize_rewards": True,
    },
}


# =============================================================================
# Unified Octax Environment Wrapper
# =============================================================================

class UnifiedOctaxEnv:
    """Wrapper that unifies action space across Octax games for continual learning."""

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
        valid_action = jnp.where(action < self.native_actions, action, 0)
        return self._env.step(key, state, valid_action, params)

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self


# =============================================================================
# PPO Creation for Octax
# =============================================================================

def create_octax_ppo(
    game: str,
    agent_kwargs: Optional[Dict[str, Any]] = None,
    ortho_mode: Optional[str] = None,
    ortho_coeff: float = 0.1,
    normalize_rewards: bool = False,
    num_envs: int = 512,
    total_timesteps: int = 5_000_000,
    eval_freq: int = 250_000,
) -> PPOOctax:
    """Create PPOOctax configured for Octax with shared backbone."""
    env, env_params = create_octax(game)
    env = UnifiedOctaxEnv(env, game)

    # Default agent kwargs
    if agent_kwargs is None:
        agent_kwargs = {"mlp_hidden_sizes": (64, 64, 64, 64), "activation": "relu"}

    # Copy to avoid modifying original
    agent_kwargs = dict(agent_kwargs)

    # Use orthogonal init by default
    agent_kwargs.setdefault("use_orthogonal_init", True)

    config = {
        "env": env,
        "env_params": env_params,
        "agent_kwargs": agent_kwargs,
        # PPO hyperparameters (from Octax paper)
        "num_envs": num_envs,
        "num_steps": 32,
        "num_epochs": 4,
        "num_minibatches": 32,
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "skip_initial_evaluation": False,
        "normalize_rewards": normalize_rewards,
        # AdaMo settings
        "ortho_mode": ortho_mode,
        "ortho_coeff": ortho_coeff,
        "ortho_exclude_output": True,
    }

    return PPOOctax.create(**config)


# =============================================================================
# Continual Learning Trainer
# =============================================================================

class OctaxContinualTrainer:
    """Manages continual learning across Octax tasks."""

    def __init__(
        self,
        task_order: List[str],
        num_cycles: int,
        steps_per_task: int,
        config: Dict[str, Any],
        num_seeds: int = 2,
        eval_freq: int = 250_000,
        output_dir: Optional[Path] = None,
        num_envs: int = 512,
        use_wandb: bool = False,
    ):
        self.task_order = task_order
        self.num_cycles = num_cycles
        self.steps_per_task = steps_per_task
        self.config = config
        self.num_seeds = num_seeds
        self.eval_freq = eval_freq
        self.output_dir = Path(output_dir) if output_dir else None
        self.num_envs = num_envs
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Total tasks = games * cycles
        self.total_tasks = len(task_order) * num_cycles

    def _create_ppo_for_task(self, game: str) -> PPOOctax:
        """Create PPOOctax instance for a specific game."""
        return create_octax_ppo(
            game=game,
            agent_kwargs=self.config.get("agent_kwargs", None),
            ortho_mode=self.config.get("ortho_mode", None),
            ortho_coeff=self.config.get("ortho_coeff", 0.1),
            normalize_rewards=self.config.get("normalize_rewards", False),
            num_envs=self.num_envs,
            total_timesteps=self.steps_per_task,
            eval_freq=self.eval_freq,
        )

    def _transfer_params(self, old_ts, new_ppo, rng):
        """Transfer network parameters from old task to new task.

        PPOOctax uses a shared backbone (single agent_ts) instead of
        separate actor_ts/critic_ts.
        """
        # Initialize new train state
        new_ts = new_ppo.init_state(rng)

        # Transfer shared agent parameters (PPOOctax has single agent_ts)
        new_ts = new_ts.replace(
            agent_ts=new_ts.agent_ts.replace(params=old_ts.agent_ts.params),
        )

        # Transfer reward normalization state if enabled
        if hasattr(old_ts, 'rew_rms_state') and hasattr(new_ts, 'rew_rms_state'):
            new_ts = new_ts.replace(rew_rms_state=old_ts.rew_rms_state)

        return new_ts

    def train_single_seed(self, seed: int) -> Dict[str, Any]:
        """Train continual learning for a single seed."""
        rng = jax.random.PRNGKey(seed)

        # Storage for learning curves
        all_returns = []  # List of (task_idx, game, timestep, returns)
        task_boundaries = []  # (task_idx, game, global_step)

        train_state = None
        global_step = 0

        for cycle in range(self.num_cycles):
            for task_idx_in_cycle, game in enumerate(self.task_order):
                task_idx = cycle * len(self.task_order) + task_idx_in_cycle
                print(f"  Seed {seed} | Cycle {cycle+1}/{self.num_cycles} | "
                      f"Task {task_idx+1}/{self.total_tasks}: {game}")

                # Create PPO for this game
                rng, rng_ppo = jax.random.split(rng)
                ppo = self._create_ppo_for_task(game)

                # Transfer or initialize
                if train_state is None:
                    train_state = ppo.init_state(rng_ppo)
                else:
                    train_state = self._transfer_params(train_state, ppo, rng_ppo)

                # Record task boundary
                task_boundaries.append((task_idx, game, global_step))

                # Train on this task
                rng, rng_train = jax.random.split(rng)
                train_fn = jax.jit(ppo.train)
                train_state, metrics = train_fn(train_state=train_state)

                # metrics = (lengths, returns) with shape (num_evals, 128)
                eval_lengths, eval_returns = metrics
                num_evals = eval_returns.shape[0]

                # Compute diagnostic metrics (gram deviation)
                # PPOOctax uses shared backbone, so we compute metrics on agent_ts
                try:
                    _, agent_gram = compute_gram_regularization_loss(
                        train_state.agent_ts.params, lambda_coeff=1.0, exclude_output=True
                    )
                    gram_agent_val = float(agent_gram["ortho/total_loss"])
                except Exception as e:
                    print(f"    Warning: Could not compute gram metrics: {e}")
                    gram_agent_val = 0.0

                # Note: l2_init not available for PPOOctax (no init_params stored)
                # Could add this later if needed

                diagnostics = {
                    "gram/actor": gram_agent_val,  # Keep key for backward compat
                    "gram/critic": gram_agent_val,  # Same value (shared backbone)
                    "l2_init/actor": 0.0,
                    "l2_init/critic": 0.0,
                }

                # Store learning curves with global timestep
                for eval_idx in range(num_evals):
                    timestep = global_step + (eval_idx + 1) * self.eval_freq
                    mean_ret = float(eval_returns[eval_idx].mean())
                    std_ret = float(eval_returns[eval_idx].std())
                    all_returns.append({
                        "task_idx": task_idx,
                        "game": game,
                        "cycle": cycle,
                        "global_step": timestep,
                        "local_step": (eval_idx + 1) * self.eval_freq,
                        "mean_return": mean_ret,
                        "std_return": std_ret,
                        # Include diagnostics at final eval point of each task
                        **(diagnostics if eval_idx == num_evals - 1 else {}),
                    })

                    # Log to wandb
                    if self.use_wandb:
                        log_data = {
                            "global_step": timestep,
                            "mean_return": mean_ret,
                            "std_return": std_ret,
                            "task_idx": task_idx,
                            "game": game,
                            "cycle": cycle,
                        }
                        if eval_idx == num_evals - 1:
                            log_data.update(diagnostics)
                        wandb.log(log_data, step=timestep)

                global_step += self.steps_per_task

                # Print final return for this task with diagnostics
                final_return = float(eval_returns[-1].mean())
                print(f"    Final return: {final_return:.1f} | "
                      f"gram: actor={diagnostics['gram/actor']:.4f}, critic={diagnostics['gram/critic']:.4f} | "
                      f"l2_init: actor={diagnostics['l2_init/actor']:.2f}, critic={diagnostics['l2_init/critic']:.2f}")

        return {
            "seed": seed,
            "learning_curves": all_returns,
            "task_boundaries": task_boundaries,
            "total_steps": global_step,
        }

    def run(self, config_name: str) -> Dict[str, Any]:
        """Run full continual learning experiment."""
        print(f"\n{'='*70}")
        print(f"Octax Continual Learning: {config_name}")
        print(f"Tasks: {self.task_order}")
        print(f"Cycles: {self.num_cycles}, Steps/task: {self.steps_per_task:,}")
        print(f"Config: {self.config}")
        print(f"{'='*70}\n")

        start_time = time.time()
        all_seed_results = []

        for seed in range(self.num_seeds):
            seed_start = time.time()
            result = self.train_single_seed(seed)
            seed_time = time.time() - seed_start
            print(f"  Seed {seed} completed in {seed_time/60:.1f} min")
            all_seed_results.append(result)

        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")

        # Aggregate results
        results = {
            "config_name": config_name,
            "config": self.config,
            "task_order": self.task_order,
            "num_cycles": self.num_cycles,
            "steps_per_task": self.steps_per_task,
            "num_seeds": self.num_seeds,
            "eval_freq": self.eval_freq,
            "total_time_sec": total_time,
            "seed_results": all_seed_results,
        }

        # Save results
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filename = self.output_dir / f"continual_{config_name}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved: {filename}")

            # Also save as npz for easy plotting
            self._save_curves_npz(results, config_name)

        return results

    def _save_curves_npz(self, results: Dict, config_name: str):
        """Save learning curves in npz format for plotting."""
        # Aggregate across seeds
        num_evals_per_task = self.steps_per_task // self.eval_freq
        total_evals = self.total_tasks * num_evals_per_task

        # Shape: (num_seeds, total_evals)
        all_returns = np.zeros((self.num_seeds, total_evals))
        all_timesteps = np.zeros(total_evals)

        # Diagnostic metrics at task boundaries: (num_seeds, total_tasks)
        gram_actor = np.zeros((self.num_seeds, self.total_tasks))
        gram_critic = np.zeros((self.num_seeds, self.total_tasks))
        l2_init_actor = np.zeros((self.num_seeds, self.total_tasks))
        l2_init_critic = np.zeros((self.num_seeds, self.total_tasks))

        for seed_idx, seed_result in enumerate(results["seed_results"]):
            task_diag_idx = 0
            for eval_idx, curve_point in enumerate(seed_result["learning_curves"]):
                if eval_idx < total_evals:
                    all_returns[seed_idx, eval_idx] = curve_point["mean_return"]
                    all_timesteps[eval_idx] = curve_point["global_step"]

                # Capture diagnostics at task boundaries (last eval of each task)
                if "gram/actor" in curve_point and task_diag_idx < self.total_tasks:
                    gram_actor[seed_idx, task_diag_idx] = curve_point["gram/actor"]
                    gram_critic[seed_idx, task_diag_idx] = curve_point["gram/critic"]
                    l2_init_actor[seed_idx, task_diag_idx] = curve_point["l2_init/actor"]
                    l2_init_critic[seed_idx, task_diag_idx] = curve_point["l2_init/critic"]
                    task_diag_idx += 1

        # Task boundaries
        task_boundaries = []
        for task_idx in range(self.total_tasks):
            task_boundaries.append(task_idx * self.steps_per_task)

        filename = self.output_dir / f"continual_{config_name}_curves.npz"
        np.savez(
            filename,
            timesteps=all_timesteps,
            returns=all_returns,  # (num_seeds, total_evals)
            mean_returns=all_returns.mean(axis=0),
            std_returns=all_returns.std(axis=0),
            task_boundaries=np.array(task_boundaries),
            task_order=self.task_order * self.num_cycles,
            config=self.config,
            # Diagnostic metrics at task boundaries
            gram_actor=gram_actor,  # (num_seeds, total_tasks)
            gram_critic=gram_critic,
            l2_init_actor=l2_init_actor,
            l2_init_critic=l2_init_critic,
        )
        print(f"Saved curves: {filename}")


# =============================================================================
# Smoke Test
# =============================================================================

def run_smoke_test(output_dir: Path, num_envs: int = 128, steps_per_task: int = 50_000, use_wandb: bool = False, wandb_project: str = "octax-continual"):
    """Quick smoke test to verify the full pipeline.

    Note: On CPU this will be slow due to JAX compilation.
    Run on GPU for faster testing, or use --smoke-test-tiny for minimal CPU test.
    """
    print("\n" + "="*70)
    print("SMOKE TEST: Verifying full pipeline")
    print("="*70 + "\n")

    test_config = {
        "agent_kwargs": {"mlp_hidden_sizes": (64, 64, 64, 64), "activation": "groupsort"},
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "normalize_rewards": False,
    }

    # Initialize wandb for smoke test
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name="smoke_test_adamo_groupsort",
            config={
                "experiment": "smoke_test",
                "config": test_config,
                "steps_per_task": steps_per_task,
                "num_envs": num_envs,
            },
        )

    eval_freq = max(steps_per_task // 20, 25_000)  # ~20 eval points per task

    trainer = OctaxContinualTrainer(
        task_order=["brix", "submarine"],  # Just 2 games from OCTAX_GAMES
        num_cycles=1,
        steps_per_task=steps_per_task,
        config=test_config,
        num_seeds=1,
        eval_freq=eval_freq,
        output_dir=output_dir,
        num_envs=num_envs,
        use_wandb=use_wandb,
    )

    start = time.time()
    results = trainer.run("smoke_test")
    elapsed = time.time() - start

    # Verify outputs
    print("\n" + "-"*50)
    print("SMOKE TEST RESULTS:")
    print("-"*50)

    # Check learning curves exist
    curves = results["seed_results"][0]["learning_curves"]
    print(f"  Learning curve points: {len(curves)}")
    assert len(curves) > 0, "No learning curves recorded!"

    # Check task boundaries
    boundaries = results["seed_results"][0]["task_boundaries"]
    print(f"  Task boundaries: {len(boundaries)}")
    assert len(boundaries) == 2, "Wrong number of task boundaries!"

    # Check files saved
    json_file = output_dir / "continual_smoke_test.json"
    npz_file = output_dir / "continual_smoke_test_curves.npz"
    assert json_file.exists(), f"JSON file not saved: {json_file}"
    assert npz_file.exists(), f"NPZ file not saved: {npz_file}"
    print(f"  JSON file: {json_file} ({json_file.stat().st_size} bytes)")
    print(f"  NPZ file: {npz_file} ({npz_file.stat().st_size} bytes)")

    # Load and verify npz
    data = np.load(npz_file, allow_pickle=True)
    print(f"  NPZ keys: {list(data.keys())}")
    print(f"  Returns shape: {data['returns'].shape}")
    print(f"  Timesteps: {data['timesteps'][:5]}...")

    # Print sample learning curve
    print(f"\n  Sample learning curve (first 4 points):")
    for i, pt in enumerate(curves[:4]):
        print(f"    Step {pt['global_step']:,}: {pt['game']} -> {pt['mean_return']:.2f}")

    print(f"\n  Elapsed time: {elapsed:.1f}s")
    print("\n" + "="*70)
    print("SMOKE TEST PASSED!")
    print("="*70 + "\n")

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return True


# =============================================================================
# Plotting
# =============================================================================

def plot_continual_results(output_dir: Path):
    """Plot continual learning results from saved npz files."""
    import matplotlib.pyplot as plt

    npz_files = list(output_dir.glob("continual_*_curves.npz"))
    if not npz_files:
        print(f"No curve files found in {output_dir}")
        return

    # Filter out smoke test
    npz_files = [f for f in npz_files if "smoke_test" not in f.name]

    if not npz_files:
        print("No non-smoke-test curve files found")
        return

    print(f"Plotting {len(npz_files)} experiments...")

    # =========================================================================
    # Plot 1: Learning curves
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        config_name = npz_file.stem.replace("continual_", "").replace("_curves", "")

        timesteps = data["timesteps"] / 1e6  # Convert to millions
        mean_returns = data["mean_returns"]
        std_returns = data["std_returns"]

        ax.plot(timesteps, mean_returns, label=config_name, linewidth=1.5)
        ax.fill_between(timesteps, mean_returns - std_returns, mean_returns + std_returns, alpha=0.2)

        # Add task boundaries
        task_boundaries = data["task_boundaries"] / 1e6
        for i, boundary in enumerate(task_boundaries[1:], 1):  # Skip first (0)
            if i == 1:  # Only label once
                ax.axvline(boundary, color='gray', linestyle='--', alpha=0.3, label='Task boundary')
            else:
                ax.axvline(boundary, color='gray', linestyle='--', alpha=0.3)

    ax.set_xlabel("Total Timesteps (M)")
    ax.set_ylabel("Mean Return")
    ax.set_title("Octax Continual Learning: 8 games x 3 cycles")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Add task labels at bottom
    if len(npz_files) > 0:
        data = np.load(npz_files[0], allow_pickle=True)
        task_order = list(data["task_order"])
        task_boundaries = list(data["task_boundaries"] / 1e6)

        for i, (game, boundary) in enumerate(zip(task_order, task_boundaries)):
            if i < len(task_boundaries) - 1:
                mid = (boundary + task_boundaries[i + 1]) / 2
            else:
                mid = boundary + (task_boundaries[1] - task_boundaries[0]) / 2
            ax.text(mid, ax.get_ylim()[0] + 5, game[:4], ha='center', fontsize=6, rotation=45)

    plt.tight_layout()
    plot_file = output_dir / "continual_learning_curves.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {plot_file}")
    plt.close()

    # =========================================================================
    # Plot 2: Gram deviation (orthogonality diagnostic)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        config_name = npz_file.stem.replace("continual_", "").replace("_curves", "")

        if "gram_actor" not in data:
            continue

        task_indices = np.arange(len(data["gram_actor"].mean(axis=0)))

        # Actor gram
        gram_actor_mean = data["gram_actor"].mean(axis=0)
        gram_actor_std = data["gram_actor"].std(axis=0)
        axes[0].plot(task_indices, gram_actor_mean, label=config_name, marker='o', markersize=4)
        axes[0].fill_between(task_indices, gram_actor_mean - gram_actor_std,
                             gram_actor_mean + gram_actor_std, alpha=0.2)

        # Critic gram
        gram_critic_mean = data["gram_critic"].mean(axis=0)
        gram_critic_std = data["gram_critic"].std(axis=0)
        axes[1].plot(task_indices, gram_critic_mean, label=config_name, marker='o', markersize=4)
        axes[1].fill_between(task_indices, gram_critic_mean - gram_critic_std,
                             gram_critic_mean + gram_critic_std, alpha=0.2)

    axes[0].set_xlabel("Task Index")
    axes[0].set_ylabel("Gram Deviation (||W'W - I||)")
    axes[0].set_title("Actor Gram Deviation")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_yscale("log")

    axes[1].set_xlabel("Task Index")
    axes[1].set_ylabel("Gram Deviation (||W'W - I||)")
    axes[1].set_title("Critic Gram Deviation")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale("log")

    plt.suptitle("Orthogonality: Lower = Weights Closer to Orthonormal", fontsize=12)
    plt.tight_layout()
    plot_file = output_dir / "continual_gram_deviation.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {plot_file}")
    plt.close()

    # =========================================================================
    # Plot 3: L2-init distance (weight drift from initialization)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        config_name = npz_file.stem.replace("continual_", "").replace("_curves", "")

        if "l2_init_actor" not in data:
            continue

        task_indices = np.arange(len(data["l2_init_actor"].mean(axis=0)))

        # Actor l2_init
        l2_actor_mean = data["l2_init_actor"].mean(axis=0)
        l2_actor_std = data["l2_init_actor"].std(axis=0)
        axes[0].plot(task_indices, l2_actor_mean, label=config_name, marker='o', markersize=4)
        axes[0].fill_between(task_indices, l2_actor_mean - l2_actor_std,
                             l2_actor_mean + l2_actor_std, alpha=0.2)

        # Critic l2_init
        l2_critic_mean = data["l2_init_critic"].mean(axis=0)
        l2_critic_std = data["l2_init_critic"].std(axis=0)
        axes[1].plot(task_indices, l2_critic_mean, label=config_name, marker='o', markersize=4)
        axes[1].fill_between(task_indices, l2_critic_mean - l2_critic_std,
                             l2_critic_mean + l2_critic_std, alpha=0.2)

    axes[0].set_xlabel("Task Index")
    axes[0].set_ylabel("L2 Distance from Init")
    axes[0].set_title("Actor Weight Drift")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Task Index")
    axes[1].set_ylabel("L2 Distance from Init")
    axes[1].set_title("Critic Weight Drift")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Weight Drift: How Far Weights Have Moved from Initialization", fontsize=12)
    plt.tight_layout()
    plot_file = output_dir / "continual_l2_init.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {plot_file}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Octax Continual Learning Benchmark")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run quick smoke test to verify pipeline")
    parser.add_argument("--steps-per-task", type=int, default=5_000_000,
                        help="Training steps per task (default: 5M)")
    parser.add_argument("--num-cycles", type=int, default=3,
                        help="Number of cycles through all tasks (default: 3)")
    parser.add_argument("--num-seeds", type=int, default=2,
                        help="Number of random seeds (default: 2)")
    parser.add_argument("--eval-freq", type=int, default=250_000,
                        help="Evaluation frequency in steps (default: 250k)")
    parser.add_argument("--num-envs", type=int, default=512,
                        help="Number of parallel environments (default: 512)")
    parser.add_argument("--configs", nargs="+",
                        default=["baseline", "adamo_groupsort", "adamo_relu", "adamo_groupsort_norm"],
                        choices=list(EXPERIMENT_CONFIGS.keys()),
                        help="Configs to run")
    parser.add_argument("--output-dir", type=Path, default=Path("results/octax_continual"),
                        help="Output directory for results")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only plot existing results, don't train")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="octax-continual",
                        help="Wandb project name")
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Plot only mode
    if args.plot_only:
        plot_continual_results(args.output_dir)
        return

    # Smoke test mode
    if args.smoke_test:
        smoke_dir = args.output_dir / "smoke_test"
        success = run_smoke_test(
            smoke_dir,
            num_envs=args.num_envs,
            steps_per_task=args.steps_per_task,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
        )
        if success:
            print("\nYou can now run the full experiment with:")
            print(f"  python scripts/bench_octax_continual.py --output-dir {args.output_dir}")
        return

    # Full experiment
    print(f"\nRunning {len(args.configs)} configs: {args.configs}")
    print(f"Output directory: {args.output_dir}")

    for config_name in args.configs:
        config = EXPERIMENT_CONFIGS[config_name]

        # Initialize wandb for each config
        if args.wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                name=f"continual_{config_name}",
                config={
                    "experiment": config_name,
                    "config": config,
                    "task_order": TASK_ORDER,
                    "num_cycles": args.num_cycles,
                    "steps_per_task": args.steps_per_task,
                    "num_seeds": args.num_seeds,
                    "num_envs": args.num_envs,
                },
                reinit=True,
            )

        trainer = OctaxContinualTrainer(
            task_order=TASK_ORDER,
            num_cycles=args.num_cycles,
            steps_per_task=args.steps_per_task,
            config=config,
            num_seeds=args.num_seeds,
            eval_freq=args.eval_freq,
            output_dir=args.output_dir,
            num_envs=args.num_envs,
            use_wandb=args.wandb,
        )

        trainer.run(config_name)

        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

    # Plot results
    print("\nGenerating plots...")
    plot_continual_results(args.output_dir)

    print("\nDone! Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
