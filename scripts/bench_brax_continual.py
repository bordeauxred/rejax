"""
Continual learning benchmark for Brax locomotion tasks with PPO.

Tests AdaMo and baselines on sequential Brax task training without weight resets.
Similar to MinAtar continual benchmark but for continuous control.

Usage:
    # Quick smoke test (single backend)
    python scripts/bench_brax_continual.py --steps-per-task 100000 --num-cycles 1 --num-seeds 1

    # Compare backends
    python scripts/bench_brax_continual.py --steps-per-task 1000000 --compare-backends

    # Full experiment with wandb
    python scripts/bench_brax_continual.py --steps-per-task 5000000 --num-cycles 2 --use-wandb

    # Test single task (no continual)
    python scripts/bench_brax_continual.py --single-task halfcheetah --steps-per-task 1000000
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from rejax import PPO
from rejax.compat.brax2gymnax import create_brax, Brax2GymnaxEnv
from brax.envs import create as brax_create


# =============================================================================
# Brax Task Configurations
# =============================================================================

def get_brax_task_info(task_name: str, backend: str = "spring") -> Dict[str, int]:
    """Get observation and action sizes for a Brax task (dynamically)."""
    from brax.envs import create
    env = create(task_name, backend=backend)
    return {
        "obs_size": env.observation_size,
        "action_size": env.action_size,
        "episode_length": env.episode_length,
    }


def get_unified_sizes(task_list: List[str], backend: str = "spring") -> Tuple[int, int]:
    """Compute unified obs/action sizes as max across all tasks."""
    max_obs = 0
    max_act = 0
    for task in task_list:
        info = get_brax_task_info(task, backend)
        max_obs = max(max_obs, info["obs_size"])
        max_act = max(max_act, info["action_size"])
    return max_obs, max_act


# Default task order for continual learning (ordered by complexity)
TASK_ORDER = ["hopper", "halfcheetah", "walker2d", "ant"]

# Cache for task info to avoid repeated env creation
_TASK_INFO_CACHE = {}


# =============================================================================
# Unified Brax Environment Wrapper
# =============================================================================

class UnifiedBraxEnv:
    """Wrapper that unifies observation and action spaces across Brax tasks.

    For continual learning, all tasks must have the same input/output dimensions.
    This wrapper:
    - Pads observations with zeros to unified_obs_size
    - Pads actions (extra dims ignored) to unified_action_size
    - Scales rewards by reward_scaling (Brax default: 10)
    - Preserves the underlying Brax environment behavior
    """

    def __init__(self, env: Brax2GymnaxEnv, task_name: str,
                 unified_obs_size: int, unified_action_size: int,
                 reward_scaling: float = 10.0):
        self._env = env
        self.task_name = task_name
        self.original_obs_size = env.env.observation_size
        self.original_action_size = env.env.action_size
        self.unified_obs_size = unified_obs_size
        self.unified_action_size = unified_action_size
        self.reward_scaling = reward_scaling

    def __getattr__(self, name):
        if name in ["_env", "task_name", "original_obs_size", "original_action_size",
                    "unified_obs_size", "unified_action_size",
                    "reset", "step", "observation_space", "action_space"]:
            return super().__getattribute__(name)
        return getattr(self._env, name)

    @property
    def default_params(self):
        return self._env.default_params

    def observation_space(self, params):
        from gymnax.environments import spaces
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(self.unified_obs_size,))

    def action_space(self, params):
        from gymnax.environments import spaces
        return spaces.Box(low=-1.0, high=1.0, shape=(self.unified_action_size,))

    def reset(self, key, params):
        obs, state = self._env.reset(key, params)
        obs = self._pad_obs(obs)
        return obs, state

    def step(self, key, state, action, params):
        # Truncate action to original size
        action = action[:self.original_action_size]
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = self._pad_obs(obs)
        reward = reward * self.reward_scaling
        return obs, state, reward, done, info

    def _pad_obs(self, obs):
        """Pad observation to unified size."""
        if self.original_obs_size < self.unified_obs_size:
            pad_width = self.unified_obs_size - self.original_obs_size
            obs = jnp.pad(obs, (0, pad_width))
        return obs


def create_unified_brax_env(
    task_name: str,
    backend: str = "spring",
    unified_obs_size: Optional[int] = None,
    unified_action_size: Optional[int] = None,
    task_list: Optional[List[str]] = None,
    reward_scaling: float = 10.0,
) -> Tuple[UnifiedBraxEnv, Any]:
    """Create a unified Brax environment for continual learning.

    Args:
        task_name: Name of the Brax task (hopper, halfcheetah, walker2d, ant)
        backend: Physics backend ("spring" fast, "mjx" accurate, "generalized" deprecated)
        unified_obs_size: Unified observation size (if None, computed from task_list)
        unified_action_size: Unified action size (if None, computed from task_list)
        task_list: List of tasks to compute unified sizes from (default: TASK_ORDER)
        reward_scaling: Multiply rewards by this factor (Brax default: 10)

    Returns:
        Tuple of (UnifiedBraxEnv, env_params)
    """
    # Compute unified sizes if not provided
    if unified_obs_size is None or unified_action_size is None:
        tasks = task_list or TASK_ORDER
        computed_obs, computed_act = get_unified_sizes(tasks, backend)
        unified_obs_size = unified_obs_size or computed_obs
        unified_action_size = unified_action_size or computed_act

    env, env_params = create_brax(task_name, backend=backend)
    unified_env = UnifiedBraxEnv(env, task_name, unified_obs_size, unified_action_size, reward_scaling)
    return unified_env, env_params


# =============================================================================
# PPO Configuration for Brax
# =============================================================================

def create_brax_ppo_config(
    env,
    env_params,
    total_timesteps: int,
    num_envs: int = 2048,
    num_steps: int = 20,  # Brax default unroll_length
    num_epochs: int = 4,
    num_minibatches: int = 64,  # Higher for more gradient steps (like MinAtar)
    learning_rate: float = 3e-4,  # Brax default
    gamma: float = 0.95,
    ent_coef: float = 0.001,  # Brax default entropy_cost
    clip_eps: float = 0.3,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    activation: str = "swish",
    normalize_observations: bool = True,
    normalize_rewards: bool = False,  # Brax uses reward_scaling instead
    eval_freq: int = 100_000,
    # AdaMo options
    ortho_mode: Optional[str] = None,
    ortho_coeff: float = 0.1,
    l2_init_coeff: Optional[float] = None,
    nap_enabled: bool = False,
    scale_enabled: bool = False,
    scale_reg_coeff: float = 0.01,
) -> Dict:
    """Create PPO config for Brax environments.

    Uses Brax paper defaults:
    - gamma=0.95 (shorter horizon than typical 0.99)
    - lr=3e-4 (Brax default)
    - ent_coef=0.001 (Brax default entropy_cost)
    - clip_eps=0.3 (larger than typical 0.2)
    - num_steps=20 (unroll_length)
    """
    config = {
        "env": env,
        "env_params": env_params,
        "agent_kwargs": {
            "hidden_layer_sizes": hidden_layer_sizes,
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
        "gae_lambda": 0.95,
        "clip_eps": clip_eps,
        "ent_coef": ent_coef,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "normalize_observations": normalize_observations,
        "normalize_rewards": normalize_rewards,
        "skip_initial_evaluation": True,
    }

    # AdaMo regularization
    if ortho_mode and ortho_mode != "none":
        config["ortho_mode"] = ortho_mode
        config["ortho_coeff"] = ortho_coeff
        config["agent_kwargs"]["use_bias"] = False
        if activation == "swish":
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

# Experiment configs for Brax PPO
# Note: All use Brax-tuned hyperparams (gamma=0.95, lr=1e-4, etc.)
#
# UTD (Update-To-Data) ratio = grad_steps / env_steps
# With num_envs=2048, num_steps=10: B = 20,480 steps per iteration
# UTD = (num_epochs × num_minibatches) / B
#
EXPERIMENT_CONFIGS = {
    # ==========================================================================
    # Brax paper settings (low UTD): 4×16=64 grad steps, UTD≈0.003
    # ==========================================================================
    "baseline_64x4_brax": {
        "hidden_layer_sizes": (64, 64, 64, 64),
        "activation": "swish",
        "ortho_mode": None,
        "num_epochs": 4,
        "num_minibatches": 16,
    },
    "baseline_256x4_brax": {
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "swish",
        "ortho_mode": None,
        "num_epochs": 4,
        "num_minibatches": 16,
    },
    # ==========================================================================
    # MinAtar-like settings (high UTD): 4×128=512 grad steps, UTD≈0.025
    # ==========================================================================
    "baseline_64x4_minatar": {
        "hidden_layer_sizes": (64, 64, 64, 64),
        "activation": "swish",
        "ortho_mode": None,
        "num_epochs": 4,
        "num_minibatches": 128,
    },
    "baseline_256x4_minatar": {
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "swish",
        "ortho_mode": None,
        "num_epochs": 4,
        "num_minibatches": 128,
    },
    # ==========================================================================
    # AdaMo variants
    # ==========================================================================
    "adamo_256x4": {
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "groupsort",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
    },
    "adamo_64x4": {
        "hidden_layer_sizes": (64, 64, 64, 64),
        "activation": "groupsort",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
    },
    "adamo_scale_256x4": {
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "groupsort",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "scale_enabled": True,
        "scale_reg_coeff": 0.01,
    },
    # ==========================================================================
    # Other plasticity methods
    # ==========================================================================
    "l2_init_256x4": {
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "swish",
        "l2_init_coeff": 0.001,
    },
    "nap_256x4": {
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "swish",
        "nap_enabled": True,
    },
    # ==========================================================================
    # Legacy aliases (for backward compatibility)
    # ==========================================================================
    "baseline": {
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "swish",
        "ortho_mode": None,
    },
    "baseline_deep": {
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "swish",
        "ortho_mode": None,
    },
    "adamo": {
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "groupsort",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
    },
}


# =============================================================================
# Training Functions
# =============================================================================

def create_cached_eval_fn(ppo):
    """Create a cached evaluation function that doesn't recompile."""
    env = ppo.env
    env_params = ppo.env_params
    actor = ppo.actor
    max_steps = env_params.max_steps_in_episode

    @jax.jit
    def cached_eval(actor_params, obs_rms_state, rng, num_episodes=32):
        """Evaluate actor with given params."""
        normalize_obs = getattr(ppo, "normalize_observations", False)

        def eval_episode(rng):
            rng_reset, rng_ep = jax.random.split(rng)
            obs, env_state = env.reset(rng_reset, env_params)

            def step_fn(carry, _):
                obs, env_state, rng, done, ret, length = carry
                rng, rng_act, rng_step = jax.random.split(rng, 3)

                # Normalize observation if needed (and obs_rms_state is provided)
                if normalize_obs and obs_rms_state is not None:
                    obs_normalized = ppo.normalize_obs(obs_rms_state, obs)
                else:
                    obs_normalized = obs

                action = actor.apply(actor_params, jnp.expand_dims(obs_normalized, 0),
                                    rng_act, method="act")
                action = jnp.squeeze(action)

                next_obs, next_state, reward, next_done, _ = env.step(
                    rng_step, env_state, action, env_params)

                ret = ret + reward * (1 - done.astype(jnp.float32))
                length = length + (1 - done.astype(jnp.int32))

                return (next_obs, next_state, rng, done | next_done, ret, length), None

            init_carry = (obs, env_state, rng_ep, jnp.array(False), 0.0, 0)
            (_, _, _, _, final_return, final_length), _ = jax.lax.scan(
                step_fn, init_carry, None, length=max_steps)

            return final_length, final_return

        rngs = jax.random.split(rng, num_episodes)
        lengths, returns = jax.vmap(eval_episode)(rngs)
        return lengths, returns

    return cached_eval


class BraxContinualTrainer:
    """Manages continual learning across Brax tasks."""

    def __init__(
        self,
        config_name: str,
        experiment_config: Dict,
        steps_per_task: int,
        num_cycles: int,
        backend: str = "spring",
        num_envs: int = 2048,
        eval_freq: int = 100_000,
        use_wandb: bool = False,
        wandb_project: str = "rejax-brax-continual",
        seed: int = 0,
        task_list: Optional[List[str]] = None,
    ):
        self.config_name = config_name
        self.experiment_config = experiment_config
        self.steps_per_task = steps_per_task
        self.num_cycles = num_cycles
        self.backend = backend
        self.num_envs = num_envs
        self.eval_freq = eval_freq
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.seed = seed
        self.task_list = task_list or TASK_ORDER

        # Compute unified sizes once for all tasks
        self.unified_obs_size, self.unified_action_size = get_unified_sizes(
            self.task_list, backend
        )
        print(f"Unified sizes: obs={self.unified_obs_size}, action={self.unified_action_size}")

        self.results = {
            "config_name": config_name,
            "experiment_config": experiment_config,
            "steps_per_task": steps_per_task,
            "num_cycles": num_cycles,
            "backend": backend,
            "tasks": self.task_list,
            "per_task_results": [],
        }

        # Cache PPO instances and compiled functions per task
        self._ppo_cache = {}

    def _print_grad_step_diagnostics(self):
        """Print gradient step diagnostics to verify we have enough updates."""
        # Get PPO config values (use defaults from create_brax_ppo_config)
        num_envs = self.num_envs
        num_steps = self.experiment_config.get("num_steps", 10)
        num_epochs = self.experiment_config.get("num_epochs", 4)
        num_minibatches = self.experiment_config.get("num_minibatches", 64)

        B = num_envs * num_steps
        num_updates = self.steps_per_task // B
        grad_steps_per_update = num_epochs * num_minibatches
        total_grad_steps = num_updates * grad_steps_per_update

        print(f"\nGradient step diagnostics:")
        print(f"  B (batch) = {num_envs} × {num_steps} = {B:,}")
        print(f"  num_updates = {num_updates:,}")
        print(f"  grad_steps/update = {num_epochs} × {num_minibatches} = {grad_steps_per_update}")
        print(f"  total_grad_steps = {total_grad_steps:,}")
        if total_grad_steps < 10_000:
            print(f"  ⚠️  WARNING: Low gradient steps! May not learn well.")
        else:
            print(f"  ✓ Gradient steps OK")

    def _get_ppo_for_task(self, task_name: str):
        """Get cached PPO and compiled functions for a task."""
        if task_name not in self._ppo_cache:
            ppo = self._create_ppo_for_task(task_name)

            @jax.jit
            def train_chunk(ts, num_iters):
                def body(_, ts):
                    return ppo.train_iteration(ts)
                return jax.lax.fori_loop(0, num_iters, body, ts)

            cached_eval = create_cached_eval_fn(ppo)
            self._ppo_cache[task_name] = (ppo, train_chunk, cached_eval)

        return self._ppo_cache[task_name]

    def _create_ppo_for_task(self, task_name: str) -> PPO:
        """Create PPO instance for a specific task."""
        env, env_params = create_unified_brax_env(
            task_name,
            backend=self.backend,
            unified_obs_size=self.unified_obs_size,
            unified_action_size=self.unified_action_size,
        )

        config = create_brax_ppo_config(
            env=env,
            env_params=env_params,
            total_timesteps=self.steps_per_task,
            num_envs=self.num_envs,
            eval_freq=self.eval_freq,
            **self.experiment_config,
        )
        return PPO.create(**config)

    def _transfer_train_state(self, old_ts, new_ppo, rng):
        """Transfer weights to new task, reset optimizer state."""
        new_ts = new_ppo.init_state(rng)

        # Transfer network weights, reset optimizer
        new_ts = new_ts.replace(
            actor_ts=new_ts.actor_ts.replace(params=old_ts.actor_ts.params),
            critic_ts=new_ts.critic_ts.replace(params=old_ts.critic_ts.params),
            global_step=jnp.array(0),
            # Preserve init params for L2-init
            actor_init_params=old_ts.actor_init_params,
            critic_init_params=old_ts.critic_init_params,
            actor_init_norms=old_ts.actor_init_norms,
            critic_init_norms=old_ts.critic_init_norms,
        )
        return new_ts

    def train_single_task(self, task_name: str, train_state, rng, cycle_idx: int,
                          task_idx: int = 0):
        """Train on a single Brax task."""
        ppo, train_chunk, cached_eval = self._get_ppo_for_task(task_name)

        if train_state is not None:
            rng, init_rng = jax.random.split(rng)
            train_state = self._transfer_train_state(train_state, ppo, init_rng)
        else:
            rng, init_rng = jax.random.split(rng)
            train_state = ppo.init_state(init_rng)

        print(f"  Training on {task_name}...", flush=True)
        start_time = time.time()

        iteration_steps = ppo.num_envs * ppo.num_steps
        num_iterations = int(np.ceil(self.steps_per_task / iteration_steps))
        eval_interval = int(np.ceil(self.eval_freq / iteration_steps))

        print(f"    Compiling...", flush=True)
        compile_start = time.time()

        total_iters = 0
        first_chunk = True

        while total_iters < num_iterations:
            chunk_size = min(eval_interval, num_iterations - total_iters)
            train_state = train_chunk(train_state, chunk_size)
            jax.block_until_ready(train_state)

            if first_chunk:
                print(f"    Compiled in {time.time() - compile_start:.1f}s", flush=True)
                first_chunk = False

            # Evaluation
            rng, eval_rng = jax.random.split(rng)
            obs_rms_state = getattr(train_state, "obs_rms_state", None)
            lengths, returns = cached_eval(train_state.actor_ts.params, obs_rms_state, eval_rng)

            current_steps = (total_iters + chunk_size) * iteration_steps
            elapsed = time.time() - start_time
            steps_per_sec = current_steps / elapsed if elapsed > 0 else 0
            mean_return = float(returns.mean())
            pct = 100 * current_steps / self.steps_per_task

            print(f"    [{task_name}] {current_steps:,}/{self.steps_per_task:,} ({pct:.0f}%) "
                  f"| return={mean_return:.1f} | {steps_per_sec:,.0f} steps/s", flush=True)

            if self.use_wandb:
                import wandb
                cumulative_step = (
                    cycle_idx * len(self.task_list) * self.steps_per_task +
                    task_idx * self.steps_per_task +
                    current_steps
                )
                wandb.log({
                    f"train/{task_name}/return": mean_return,
                    "return": mean_return,
                    "cycle": cycle_idx,
                    "task_idx": task_idx,
                    "task": task_name,
                }, step=cumulative_step)

            total_iters += chunk_size

        elapsed = time.time() - start_time
        steps_per_sec = self.steps_per_task / elapsed

        # Final evaluation
        rng, eval_rng = jax.random.split(rng)
        lengths, returns = cached_eval(train_state.actor_ts.params, obs_rms_state, eval_rng)
        final_return = float(returns.mean())

        print(f"  Completed {task_name}: final_return={final_return:.1f}, "
              f"{steps_per_sec:,.0f} steps/s", flush=True)

        return train_state, rng, {
            "task": task_name,
            "cycle": cycle_idx,
            "final_return": final_return,
            "elapsed_s": elapsed,
            "steps_per_sec": steps_per_sec,
        }

    def evaluate_all_tasks(self, train_state, rng, cycle_idx: int, current_task_idx: int):
        """Evaluate current policy on all tasks.

        Note: obs_rms_state is only valid for the current task (statistics are task-specific).
        For other tasks, we evaluate without observation normalization.
        """
        print(f"  Evaluating on all tasks...")
        eval_results = {}
        current_task = self.task_list[current_task_idx]

        for task_name in self.task_list:
            ppo, _, cached_eval = self._get_ppo_for_task(task_name)
            rng, eval_rng = jax.random.split(rng)
            # Only use obs_rms_state for current task (stats are task-specific)
            if task_name == current_task:
                obs_rms_state = getattr(train_state, "obs_rms_state", None)
            else:
                obs_rms_state = None  # Skip normalization for other tasks
            lengths, returns = cached_eval(train_state.actor_ts.params, obs_rms_state, eval_rng)
            mean_return = float(returns.mean())
            eval_results[task_name] = mean_return
            print(f"    {task_name}: {mean_return:.1f}")

        return eval_results, rng

    def run(self, rng):
        """Run the full continual learning experiment."""
        train_state = None

        print(f"\nConfig: {self.config_name}")
        print(f"Backend: {self.backend}")
        print(f"Tasks: {' -> '.join(self.task_list)}")
        print(f"Steps per task: {self.steps_per_task:,}")
        print(f"Cycles: {self.num_cycles}")

        # Print gradient step diagnostics
        self._print_grad_step_diagnostics()

        for cycle_idx in range(self.num_cycles):
            print(f"\n{'='*60}")
            print(f"Cycle {cycle_idx + 1}/{self.num_cycles}")
            print(f"{'='*60}")

            for task_idx, task_name in enumerate(self.task_list):
                print(f"\nTask {task_idx + 1}/{len(self.task_list)}: {task_name}")

                train_state, rng, task_result = self.train_single_task(
                    task_name, train_state, rng, cycle_idx, task_idx
                )
                self.results["per_task_results"].append(task_result)

                # Evaluate on all tasks
                eval_results, rng = self.evaluate_all_tasks(
                    train_state, rng, cycle_idx, task_idx
                )
                task_result["eval_all_tasks"] = eval_results

                if self.use_wandb:
                    import wandb
                    for eval_task, eval_return in eval_results.items():
                        wandb.log({
                            f"eval_after_{task_name}/{eval_task}": eval_return,
                        })

        return self.results


# =============================================================================
# Backend Comparison
# =============================================================================

def compare_backends(task_name: str, steps: int, num_envs: int = 2048):
    """Compare throughput and scores across different Brax backends."""
    backends = ["mjx", "spring", "generalized"]
    results = {}

    print(f"\nComparing backends on {task_name} ({steps:,} steps)")
    print("=" * 60)

    for backend in backends:
        print(f"\n{backend.upper()}:")
        try:
            # For single-task comparison, use native env (no padding needed)
            env, env_params = create_brax(task_name, backend=backend)
            print(f"  obs={env.env.observation_size}, action={env.env.action_size}")
            config = create_brax_ppo_config(
                env=env,
                env_params=env_params,
                total_timesteps=steps,
                num_envs=num_envs,
                eval_freq=steps,  # Only eval at end
            )
            ppo = PPO.create(**config)

            # Compile
            print("  Compiling...", end=" ", flush=True)
            rng = jax.random.PRNGKey(0)
            compile_start = time.time()
            train_state, _ = PPO.train(ppo, rng)
            jax.block_until_ready(train_state)
            compile_time = time.time() - compile_start
            print(f"{compile_time:.1f}s")

            # Benchmark
            print("  Running benchmark...", end=" ", flush=True)
            rng = jax.random.PRNGKey(1)
            start = time.time()
            train_state, eval_data = PPO.train(ppo, rng)
            jax.block_until_ready(train_state)
            runtime = time.time() - start

            steps_per_sec = steps / runtime
            _, returns = eval_data
            final_return = float(returns.mean())

            print(f"done")
            print(f"  Steps/sec: {steps_per_sec:,.0f}")
            print(f"  Final return: {final_return:.1f}")

            results[backend] = {
                "compile_time": compile_time,
                "runtime": runtime,
                "steps_per_sec": steps_per_sec,
                "final_return": final_return,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[backend] = {"error": str(e)}

    return results


# =============================================================================
# Single Task Baseline
# =============================================================================

def run_single_task(
    task_name: str,
    steps: int,
    backend: str = "spring",
    num_seeds: int = 3,
    num_envs: int = 2048,
    experiment_config: Optional[Dict] = None,
    config_name: str = "baseline",
    use_wandb: bool = False,
    wandb_project: str = "rejax-brax-single",
):
    """Run PPO on a single Brax task with multiple seeds."""
    config = experiment_config or EXPERIMENT_CONFIGS["baseline"]

    print(f"\nSingle task: {task_name}")
    print(f"Backend: {backend}, Steps: {steps:,}, Seeds: {num_seeds}")
    print(f"Config: {config}")
    print("=" * 60)

    # Use unified env for fair comparison with continual learning
    unified_obs, unified_act = get_unified_sizes(TASK_ORDER, backend)
    env, env_params = create_unified_brax_env(
        task_name, backend=backend,
        unified_obs_size=unified_obs, unified_action_size=unified_act
    )
    print(f"Env: obs={env.original_obs_size}→{unified_obs}, action={env.original_action_size}→{unified_act}")

    eval_freq = steps // 10
    ppo_config = create_brax_ppo_config(
        env=env,
        env_params=env_params,
        total_timesteps=steps,
        num_envs=num_envs,
        eval_freq=eval_freq,
        **config,
    )
    ppo = PPO.create(**ppo_config)

    # Initialize wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=f"{task_name}_{config_name}",
            config={
                "task": task_name,
                "backend": backend,
                "steps": steps,
                "num_seeds": num_seeds,
                "num_envs": num_envs,
                "config_name": config_name,
                "eval_freq": eval_freq,
                **config,
                # PPO defaults
                "learning_rate": ppo.learning_rate,
                "gamma": ppo.gamma,
                "num_steps": ppo.num_steps,
                "num_epochs": ppo.num_epochs,
                "num_minibatches": ppo.num_minibatches,
                "ent_coef": ppo.ent_coef,
                "clip_eps": ppo.clip_eps,
            },
            reinit=True,
        )

    # Compile
    print("Compiling...", end=" ", flush=True)
    compile_start = time.time()
    keys = jax.random.split(jax.random.PRNGKey(0), num_seeds)
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
        # Log training curves to wandb
        num_evals = returns.shape[1]
        for eval_idx in range(num_evals):
            eval_step = (eval_idx + 1) * eval_freq
            eval_returns = returns[:, eval_idx, :].mean(axis=-1)  # mean over episodes per seed
            mean_return = float(eval_returns.mean())
            std_return = float(eval_returns.std())

            if use_wandb:
                wandb.log({
                    f"train/{task_name}/return": mean_return,
                    f"train/{task_name}/return_std": std_return,
                    f"train/{task_name}/return_min": float(eval_returns.min()),
                    f"train/{task_name}/return_max": float(eval_returns.max()),
                    "step": eval_step,
                })

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

    # Log final results to wandb
    if use_wandb:
        wandb.log({
            "final/mean_return": float(final_returns.mean()),
            "final/std_return": float(final_returns.std()),
            "final/per_seed_returns": final_returns.tolist(),
            "perf/steps_per_sec": steps_per_sec,
            "perf/steps_per_sec_per_seed": steps_per_sec_per_seed,
            "perf/compile_time": compile_time,
            "perf/runtime": runtime,
        })
        wandb.finish()

    return {
        "task": task_name,
        "backend": backend,
        "steps": steps,
        "num_seeds": num_seeds,
        "compile_time": compile_time,
        "runtime": runtime,
        "steps_per_sec": steps_per_sec,
        "steps_per_sec_per_seed": steps_per_sec_per_seed,
        "final_returns": final_returns.tolist(),
        "mean_return": float(final_returns.mean()),
        "std_return": float(final_returns.std()),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Brax continual learning benchmark")
    parser.add_argument("--steps-per-task", type=int, default=5_000_000)
    parser.add_argument("--num-cycles", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--backend", type=str, default="spring",
                        choices=["mjx", "spring", "generalized"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="rejax-brax-continual")
    parser.add_argument("--output-dir", type=str, default="results/brax_continual")

    # Mode selection
    parser.add_argument("--compare-backends", action="store_true",
                        help="Compare throughput across backends")
    parser.add_argument("--single-task", type=str, default=None,
                        choices=TASK_ORDER,
                        help="Run single task baseline (no continual)")
    parser.add_argument("--configs", nargs="+", default=["baseline"],
                        choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--tasks", nargs="+", default=None,
                        choices=TASK_ORDER,
                        help="Tasks to include (default: all)")

    args = parser.parse_args()

    # Backend comparison mode
    if args.compare_backends:
        task = args.single_task or "halfcheetah"
        results = compare_backends(task, args.steps_per_task, args.num_envs)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "backend_comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # Single task mode
    if args.single_task:
        for config_name in args.configs:
            config = EXPERIMENT_CONFIGS[config_name]
            results = run_single_task(
                args.single_task,
                args.steps_per_task,
                args.backend,
                args.num_seeds,
                args.num_envs,
                config,
                config_name=config_name,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
            )
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"single_{args.single_task}_{config_name}.json", "w") as f:
                json.dump(results, f, indent=2)
        return

    # Continual learning mode
    task_list = args.tasks or TASK_ORDER
    all_results = {}

    for seed_idx in range(args.num_seeds):
        print(f"\n{'#'*70}")
        print(f"# SEED {seed_idx + 1}/{args.num_seeds}")
        print(f"{'#'*70}")

        for config_name in args.configs:
            config = EXPERIMENT_CONFIGS[config_name]

            if args.use_wandb:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=f"{config_name}_seed{seed_idx}",
                    config={
                        "config_name": config_name,
                        "experiment_config": config,
                        "steps_per_task": args.steps_per_task,
                        "num_cycles": args.num_cycles,
                        "backend": args.backend,
                        "seed": args.seed + seed_idx,
                    },
                    reinit=True,
                )

            trainer = BraxContinualTrainer(
                config_name=f"{config_name}_seed{seed_idx}",
                experiment_config=config,
                steps_per_task=args.steps_per_task,
                num_cycles=args.num_cycles,
                backend=args.backend,
                num_envs=args.num_envs,
                eval_freq=args.eval_freq,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                seed=args.seed + seed_idx,
                task_list=task_list,
            )

            rng = jax.random.PRNGKey(args.seed + seed_idx)
            results = trainer.run(rng)

            if config_name not in all_results:
                all_results[config_name] = []
            all_results[config_name].append(results)

            if args.use_wandb:
                import wandb
                wandb.finish()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"continual_results_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
