"""
Continual learning benchmark for Octax (CHIP-8 arcade games in JAX).

Tests AdaMo and baselines on sequential Octax game training without weight resets.
Follows the MinAtar and Brax continual learning benchmark patterns.

Reference: arXiv 2510.01764 (Octax paper)

Usage:
    # Quick smoke test
    python scripts/bench_octax_continual.py --steps-per-task 100000 --num-cycles 1 --num-seeds 1

    # Full experiment with wandb
    python scripts/bench_octax_continual.py --steps-per-task 5000000 --num-cycles 2 --use-wandb

    # Single task baseline (no continual)
    python scripts/bench_octax_continual.py --single-task brix --steps-per-task 5000000
"""
import argparse
import json
import time
import warnings
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from rejax import PPO
from rejax.compat.octax2gymnax import create_octax


# =============================================================================
# Octax Game Configurations
# =============================================================================

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

# Task order for continual learning (5 representative games)
TASK_ORDER = ["brix", "tetris", "tank", "spacejam", "deep"]

# Unified action space (max across games = tank with 6 actions)
UNIFIED_ACTIONS = 6


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
        # Octax envs may throw error on deepcopy, return shallow copy
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

def create_octax_ppo_config(
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
    # ==========================================================================
    # Baseline configurations (no plasticity interventions)
    # ==========================================================================
    "baseline_64x4": {
        "mlp_hidden_sizes": (64, 64, 64, 64),
        "activation": "relu",
        "ortho_mode": None,
    },
    "baseline_256x4": {
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "activation": "relu",
        "ortho_mode": None,
    },
    # ==========================================================================
    # AdaMo configurations (orthogonal optimizer)
    # ==========================================================================
    "adamo_64x4": {
        "mlp_hidden_sizes": (64, 64, 64, 64),
        "activation": "groupsort",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
    },
    "adamo_256x4": {
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "activation": "groupsort",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
    },
    # ==========================================================================
    # Scale-AdaMo (per-layer learnable scaling)
    # ==========================================================================
    "scale_adamo_64x4": {
        "mlp_hidden_sizes": (64, 64, 64, 64),
        "activation": "groupsort",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "scale_enabled": True,
        "scale_reg_coeff": 0.01,
    },
    "scale_adamo_256x4": {
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "activation": "groupsort",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "scale_enabled": True,
        "scale_reg_coeff": 0.01,
    },
    # ==========================================================================
    # L2-Init (regenerative regularization toward init)
    # ==========================================================================
    "l2_init_64x4": {
        "mlp_hidden_sizes": (64, 64, 64, 64),
        "activation": "relu",
        "l2_init_coeff": 0.001,
    },
    "l2_init_256x4": {
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "activation": "relu",
        "l2_init_coeff": 0.001,
    },
    # ==========================================================================
    # NaP (Normalize-and-Project)
    # ==========================================================================
    "nap_64x4": {
        "mlp_hidden_sizes": (64, 64, 64, 64),
        "activation": "relu",
        "nap_enabled": True,
    },
    "nap_256x4": {
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "activation": "relu",
        "nap_enabled": True,
    },
}


# =============================================================================
# Cached Evaluation Function
# =============================================================================

def create_cached_eval_fn(ppo):
    """Create a cached evaluation function that doesn't recompile on each call."""
    env = ppo.env
    env_params = ppo.env_params
    actor = ppo.actor
    max_steps = env_params.max_steps_in_episode

    @jax.jit
    def cached_eval(actor_params, rng, num_episodes=32):
        """Evaluate actor with given params (no recompilation)."""

        def eval_episode(rng):
            """Run single episode, return (length, return)."""
            rng_reset, rng_ep = jax.random.split(rng)
            obs, env_state = env.reset(rng_reset, env_params)

            def step_fn(carry, _):
                obs, env_state, rng, done, ret, length = carry
                rng, rng_act, rng_step = jax.random.split(rng, 3)

                # Use actor directly with params
                action = actor.apply(actor_params, jnp.expand_dims(obs, 0),
                                    rng_act, method="act")
                action = jnp.squeeze(action)

                next_obs, next_state, reward, next_done, _ = env.step(
                    rng_step, env_state, action, env_params)

                # Only update if not already done
                ret = ret + reward * (1 - done)
                length = length + (1 - done).astype(jnp.int32)

                return (next_obs, next_state, rng, done | next_done, ret, length), None

            init_carry = (obs, env_state, rng_ep, False, 0.0, 0)
            (_, _, _, _, final_return, final_length), _ = jax.lax.scan(
                step_fn, init_carry, None, length=max_steps)

            return final_length, final_return

        rngs = jax.random.split(rng, num_episodes)
        lengths, returns = jax.vmap(eval_episode)(rngs)
        return lengths, returns

    return cached_eval


# =============================================================================
# Continual Learning Trainer
# =============================================================================

class OctaxContinualTrainer:
    """Manages continual learning across Octax tasks."""

    def __init__(
        self,
        config_name: str,
        experiment_config: Dict,
        steps_per_task: int,
        num_cycles: int,
        num_envs: int = 2048,
        eval_freq: int = 250_000,
        use_wandb: bool = False,
        wandb_project: str = "rejax-octax-continual",
        seed: int = 0,
        task_list: Optional[List[str]] = None,
        normalize_rewards: bool = False,
    ):
        self.config_name = config_name
        self.experiment_config = experiment_config
        self.steps_per_task = steps_per_task
        self.num_cycles = num_cycles
        self.num_envs = num_envs
        self.eval_freq = eval_freq
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.seed = seed
        self.task_list = task_list or TASK_ORDER
        self.normalize_rewards = normalize_rewards

        self.results = {
            "config_name": config_name,
            "experiment_config": experiment_config,
            "steps_per_task": steps_per_task,
            "num_cycles": num_cycles,
            "tasks": self.task_list,
            "normalize_rewards": normalize_rewards,
            "per_task_results": [],
        }

        # Cache PPO instances and compiled functions per task
        self._ppo_cache = {}

    def _print_grad_step_diagnostics(self):
        """Print gradient step diagnostics."""
        num_envs = self.num_envs
        num_steps = self.experiment_config.get("num_steps", 32)
        num_epochs = self.experiment_config.get("num_epochs", 4)
        num_minibatches = self.experiment_config.get("num_minibatches", 32)

        B = num_envs * num_steps
        num_updates = self.steps_per_task // B
        grad_steps_per_update = num_epochs * num_minibatches
        total_grad_steps = num_updates * grad_steps_per_update

        print(f"\nGradient step diagnostics:")
        print(f"  B (batch) = {num_envs} x {num_steps} = {B:,}")
        print(f"  num_updates = {num_updates:,}")
        print(f"  grad_steps/update = {num_epochs} x {num_minibatches} = {grad_steps_per_update}")
        print(f"  total_grad_steps = {total_grad_steps:,}")

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
        env, env_params = create_unified_env(task_name)

        config = create_octax_ppo_config(
            env=env,
            env_params=env_params,
            total_timesteps=self.steps_per_task,
            num_envs=self.num_envs,
            eval_freq=self.eval_freq,
            normalize_rewards=self.normalize_rewards,
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
        """Train on a single Octax task."""
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
            lengths, returns = cached_eval(train_state.actor_ts.params, eval_rng)

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
        lengths, returns = cached_eval(train_state.actor_ts.params, eval_rng)
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
            lengths, returns = cached_eval(train_state.actor_ts.params, eval_rng)
            mean_return = float(returns.mean())
            eval_results[task_name] = mean_return
            print(f"    {task_name}: {mean_return:.1f}")

        return eval_results, rng

    def run(self, rng):
        """Run the full continual learning experiment."""
        train_state = None

        print(f"\nConfig: {self.config_name}")
        print(f"Tasks: {' -> '.join(self.task_list)}")
        print(f"Steps per task: {self.steps_per_task:,}")
        print(f"Cycles: {self.num_cycles}")
        print(f"Normalize rewards: {self.normalize_rewards}")

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
# Single Task Baseline
# =============================================================================

def run_single_task(
    task_name: str,
    steps: int,
    num_seeds: int = 3,
    num_envs: int = 2048,
    experiment_config: Optional[Dict] = None,
    config_name: str = "baseline_256x4",
    normalize_rewards: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "rejax-octax-single",
):
    """Run PPO on a single Octax task with multiple seeds."""
    config = experiment_config or EXPERIMENT_CONFIGS[config_name]

    print(f"\nSingle task: {task_name}")
    print(f"Steps: {steps:,}, Seeds: {num_seeds}")
    print(f"Config: {config}")
    print("=" * 60)

    env, env_params = create_unified_env(task_name)

    eval_freq = max(steps // 20, 10000)
    ppo_config = create_octax_ppo_config(
        env=env,
        env_params=env_params,
        total_timesteps=steps,
        num_envs=num_envs,
        eval_freq=eval_freq,
        normalize_rewards=normalize_rewards,
        **config,
    )
    ppo = PPO.create(**ppo_config)

    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=f"{task_name}_{config_name}",
            config={
                "task": task_name,
                "steps": steps,
                "num_seeds": num_seeds,
                "num_envs": num_envs,
                "config_name": config_name,
                "normalize_rewards": normalize_rewards,
                **config,
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
    if returns.ndim == 3:
        final_returns = returns[:, -1, :].mean(axis=-1)
    else:
        final_returns = returns.mean(axis=-1)

    steps_per_sec = steps * num_seeds / runtime

    print(f"done")
    print(f"\nResults:")
    print(f"  Steps/sec: {steps_per_sec:,.0f}")
    print(f"  Final returns: {final_returns.mean():.1f} +/- {final_returns.std():.1f}")

    if use_wandb:
        import wandb
        wandb.log({
            "final/mean_return": float(final_returns.mean()),
            "final/std_return": float(final_returns.std()),
            "perf/steps_per_sec": steps_per_sec,
        })
        wandb.finish()

    return {
        "task": task_name,
        "config_name": config_name,
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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Octax continual learning benchmark")
    parser.add_argument("--steps-per-task", type=int, default=5_000_000)
    parser.add_argument("--num-cycles", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--eval-freq", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize-rewards", action="store_true",
                        help="Enable reward normalization")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="rejax-octax-continual")
    parser.add_argument("--output-dir", type=str, default="results/octax_continual")

    # Mode selection
    parser.add_argument("--single-task", type=str, default=None,
                        choices=list(OCTAX_GAMES.keys()),
                        help="Run single task baseline (no continual)")
    parser.add_argument("--configs", nargs="+", default=["baseline_64x4"],
                        choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--tasks", nargs="+", default=None,
                        choices=list(OCTAX_GAMES.keys()),
                        help="Tasks to include (default: TASK_ORDER)")

    args = parser.parse_args()

    # Single task mode
    if args.single_task:
        for config_name in args.configs:
            config = EXPERIMENT_CONFIGS[config_name]
            results = run_single_task(
                args.single_task,
                args.steps_per_task,
                args.num_seeds,
                args.num_envs,
                config,
                config_name=config_name,
                normalize_rewards=args.normalize_rewards,
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
                        "seed": args.seed + seed_idx,
                        "normalize_rewards": args.normalize_rewards,
                    },
                    reinit=True,
                )

            trainer = OctaxContinualTrainer(
                config_name=f"{config_name}_seed{seed_idx}",
                experiment_config=config,
                steps_per_task=args.steps_per_task,
                num_cycles=args.num_cycles,
                num_envs=args.num_envs,
                eval_freq=args.eval_freq,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                seed=args.seed + seed_idx,
                task_list=task_list,
                normalize_rewards=args.normalize_rewards,
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

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for config_name, results_list in all_results.items():
        print(f"\n{config_name}:")
        for seed_idx, results in enumerate(results_list):
            final_returns = [r["final_return"] for r in results["per_task_results"]]
            mean_return = np.mean(final_returns)
            print(f"  Seed {seed_idx}: mean_return={mean_return:.1f}")


if __name__ == "__main__":
    main()
