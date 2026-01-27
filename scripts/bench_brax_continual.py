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

# Tasks ordered roughly by complexity (observation/action size)
BRAX_TASKS = {
    "hopper": {"obs_size": 11, "action_size": 3},
    "halfcheetah": {"obs_size": 17, "action_size": 6},
    "walker2d": {"obs_size": 17, "action_size": 6},
    "ant": {"obs_size": 27, "action_size": 8},
}

# Default task order for continual learning
TASK_ORDER = ["hopper", "halfcheetah", "walker2d", "ant"]

# Unified space sizes (max across all tasks)
UNIFIED_OBS_SIZE = 27  # max from ant
UNIFIED_ACTION_SIZE = 8  # max from ant


# =============================================================================
# Unified Brax Environment Wrapper
# =============================================================================

class UnifiedBraxEnv:
    """Wrapper that unifies observation and action spaces across Brax tasks.

    For continual learning, all tasks must have the same input/output dimensions.
    This wrapper:
    - Pads observations with zeros to UNIFIED_OBS_SIZE
    - Pads actions (extra dims ignored) to UNIFIED_ACTION_SIZE
    - Preserves the underlying Brax environment behavior
    """

    def __init__(self, env: Brax2GymnaxEnv, task_name: str):
        self._env = env
        self.task_name = task_name
        self.original_obs_size = BRAX_TASKS[task_name]["obs_size"]
        self.original_action_size = BRAX_TASKS[task_name]["action_size"]

    def __getattr__(self, name):
        if name in ["_env", "task_name", "original_obs_size", "original_action_size",
                    "reset", "step", "observation_space", "action_space"]:
            return super().__getattribute__(name)
        return getattr(self._env, name)

    @property
    def default_params(self):
        return self._env.default_params

    def observation_space(self, params):
        from gymnax.environments import spaces
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(UNIFIED_OBS_SIZE,))

    def action_space(self, params):
        from gymnax.environments import spaces
        return spaces.Box(low=-1.0, high=1.0, shape=(UNIFIED_ACTION_SIZE,))

    def reset(self, key, params):
        obs, state = self._env.reset(key, params)
        obs = self._pad_obs(obs)
        return obs, state

    def step(self, key, state, action, params):
        # Truncate action to original size
        action = action[:self.original_action_size]
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = self._pad_obs(obs)
        return obs, state, reward, done, info

    def _pad_obs(self, obs):
        """Pad observation to unified size."""
        if self.original_obs_size < UNIFIED_OBS_SIZE:
            pad_width = UNIFIED_OBS_SIZE - self.original_obs_size
            obs = jnp.pad(obs, (0, pad_width))
        return obs


def create_unified_brax_env(task_name: str, backend: str = "mjx") -> Tuple[UnifiedBraxEnv, Any]:
    """Create a unified Brax environment for continual learning.

    Args:
        task_name: Name of the Brax task (hopper, halfcheetah, walker2d, ant)
        backend: Physics backend ("mjx" recommended, "spring" or "generalized" also available)

    Returns:
        Tuple of (UnifiedBraxEnv, env_params)
    """
    env, env_params = create_brax(task_name, backend=backend)
    unified_env = UnifiedBraxEnv(env, task_name)
    return unified_env, env_params


# =============================================================================
# PPO Configuration for Brax
# =============================================================================

def create_brax_ppo_config(
    env,
    env_params,
    total_timesteps: int,
    num_envs: int = 2048,
    num_steps: int = 10,
    num_epochs: int = 4,
    num_minibatches: int = 32,
    learning_rate: float = 3e-4,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    activation: str = "swish",
    normalize_observations: bool = True,
    normalize_rewards: bool = True,
    reward_scaling: float = 1.0,
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

    Uses settings similar to PureJaxRL/Brax defaults.
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
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.0,  # Brax typically uses 0 entropy for continuous control
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

EXPERIMENT_CONFIGS = {
    "baseline": {
        "hidden_layer_sizes": (256, 256),
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
    "adamo_scale": {
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "groupsort",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "scale_enabled": True,
        "scale_reg_coeff": 0.01,
    },
    "l2_init": {
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "swish",
        "l2_init_coeff": 0.001,
    },
    "nap": {
        "hidden_layer_sizes": (256, 256, 256, 256),
        "activation": "swish",
        "nap_enabled": True,
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

                # Normalize observation if needed
                if normalize_obs:
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
        backend: str = "mjx",
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
        env, env_params = create_unified_brax_env(task_name, backend=self.backend)

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
        """Evaluate current policy on all tasks."""
        print(f"  Evaluating on all tasks...")
        eval_results = {}

        for task_name in self.task_list:
            ppo, _, cached_eval = self._get_ppo_for_task(task_name)
            rng, eval_rng = jax.random.split(rng)
            obs_rms_state = getattr(train_state, "obs_rms_state", None)
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
            env, env_params = create_unified_brax_env(task_name, backend=backend)
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
    backend: str = "mjx",
    num_seeds: int = 3,
    num_envs: int = 2048,
    experiment_config: Optional[Dict] = None,
):
    """Run PPO on a single Brax task with multiple seeds."""
    config = experiment_config or EXPERIMENT_CONFIGS["baseline"]

    print(f"\nSingle task: {task_name}")
    print(f"Backend: {backend}, Steps: {steps:,}, Seeds: {num_seeds}")
    print(f"Config: {config}")
    print("=" * 60)

    env, env_params = create_unified_brax_env(task_name, backend=backend)
    ppo_config = create_brax_ppo_config(
        env=env,
        env_params=env_params,
        total_timesteps=steps,
        num_envs=num_envs,
        eval_freq=steps // 10,
        **config,
    )
    ppo = PPO.create(**ppo_config)

    # Compile
    print("Compiling...", end=" ", flush=True)
    compile_start = time.time()
    keys = jax.random.split(jax.random.PRNGKey(0), num_seeds)
    vmap_train = jax.jit(jax.vmap(PPO.train, in_axes=(None, 0)))
    train_states, _ = vmap_train(ppo, keys)
    jax.block_until_ready(train_states)
    print(f"{time.time() - compile_start:.1f}s")

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

    print(f"done")
    print(f"\nResults:")
    print(f"  Steps/sec: {steps_per_sec:,.0f} ({steps_per_sec/num_seeds:,.0f} per seed)")
    print(f"  Final returns: {final_returns.mean():.1f} +/- {final_returns.std():.1f}")
    print(f"  Per-seed: {[f'{r:.1f}' for r in final_returns]}")

    return {
        "task": task_name,
        "backend": backend,
        "steps": steps,
        "num_seeds": num_seeds,
        "runtime": runtime,
        "steps_per_sec": steps_per_sec,
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
    parser.add_argument("--backend", type=str, default="mjx",
                        choices=["mjx", "spring", "generalized"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="rejax-brax-continual")
    parser.add_argument("--output-dir", type=str, default="results/brax_continual")

    # Mode selection
    parser.add_argument("--compare-backends", action="store_true",
                        help="Compare throughput across backends")
    parser.add_argument("--single-task", type=str, default=None,
                        choices=list(BRAX_TASKS.keys()),
                        help="Run single task baseline (no continual)")
    parser.add_argument("--configs", nargs="+", default=["baseline"],
                        choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--tasks", nargs="+", default=None,
                        choices=list(BRAX_TASKS.keys()),
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
