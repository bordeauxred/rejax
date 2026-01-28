"""
Quick validation benchmark for PPO with PopArt on Octax.

Tests different MLP configurations:
- 256x1: Paper-exact single layer
- 64x4: Deep narrow (for plasticity research)
- 256x4: Deep wide

Usage:
    # Quick smoke test (default)
    python scripts/bench_octax_popart.py

    # Full single-task validation
    python scripts/bench_octax_popart.py --steps 5000000 --game brix

    # Continual learning test
    python scripts/bench_octax_popart.py --continual --steps-per-task 1000000
"""
import argparse
import json
import time
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rejax.algos.ppo_octax_popart import PPOOctaxPopArt, PopArtState
from rejax.compat.octax2gymnax import create_octax


# =============================================================================
# Octax Configuration (same as bench_octax_continual.py)
# =============================================================================

OCTAX_GAMES = {
    # Puzzle
    "tetris": {"actions": 5},
    # Action/Arcade
    "brix": {"actions": 3},
    "pong": {"actions": 3},
    "blinky": {"actions": 4},
    "worm": {"actions": 4},
    # Strategy
    "tank": {"actions": 6},
    # Exploration/Shooter
    "spacejam": {"actions": 5},
    "deep": {"actions": 4},
    "flight_runner": {"actions": 3},
}

TASK_ORDER = ["brix", "tetris", "tank", "spacejam", "deep"]
UNIFIED_ACTIONS = 6


class UnifiedOctaxEnv:
    """Wrapper that unifies action space across Octax games."""

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
            f"Shallow copy returned for {type(self).__name__}",
            category=RuntimeWarning, stacklevel=2,
        )
        return copy(self)


def create_unified_env(game_name: str) -> Tuple[UnifiedOctaxEnv, Any]:
    """Create a unified Octax environment."""
    game_info = OCTAX_GAMES[game_name]
    env, env_params = create_octax(game_name)
    unified_env = UnifiedOctaxEnv(env, game_name, game_info["actions"])
    return unified_env, env_params


# =============================================================================
# Architecture Configurations
# =============================================================================

ARCH_CONFIGS = {
    "256x1": {
        "mlp_hidden_sizes": (256,),
        "description": "Paper-exact single hidden layer",
    },
    "64x4": {
        "mlp_hidden_sizes": (64, 64, 64, 64),
        "description": "Deep narrow (plasticity research)",
    },
    "256x4": {
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "description": "Deep wide",
    },
}


def create_popart_ppo(
    env,
    env_params,
    total_timesteps: int,
    mlp_hidden_sizes: Tuple[int, ...] = (256,),
    num_envs: int = 512,
    num_steps: int = 32,
    num_epochs: int = 8,
    num_minibatches: int = 32,
    learning_rate: float = 5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    popart_beta: float = 0.0001,
    eval_freq: int = 250_000,
) -> PPOOctaxPopArt:
    """Create PPOOctaxPopArt instance."""
    config = {
        "env": env,
        "env_params": env_params,
        "agent_kwargs": {
            "mlp_hidden_sizes": mlp_hidden_sizes,
            "activation": "relu",
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
        "popart_beta": popart_beta,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "skip_initial_evaluation": True,
    }
    return PPOOctaxPopArt.create(**config)


# =============================================================================
# Single Task Benchmark
# =============================================================================

def run_single_task(
    game: str,
    arch_name: str,
    total_steps: int,
    num_envs: int = 512,
    num_seeds: int = 1,
    popart_beta: float = 0.0001,
    eval_freq: int = None,
    verbose: bool = True,
):
    """Run single-task benchmark with full learning curves.

    Returns dict with:
    - Learning curves: returns at each eval point (seeds x evals x episodes)
    - Final statistics
    - PopArt state evolution
    """
    arch = ARCH_CONFIGS[arch_name]

    # IMPORTANT: Use UnifiedOctaxEnv for consistent action space (6 actions)
    env, env_params = create_unified_env(game)

    # ~20 eval points for smooth curves
    if eval_freq is None:
        eval_freq = max(total_steps // 20, 10000)

    ppo = create_popart_ppo(
        env=env,
        env_params=env_params,
        total_timesteps=total_steps,
        mlp_hidden_sizes=arch["mlp_hidden_sizes"],
        num_envs=num_envs,
        popart_beta=popart_beta,
        eval_freq=eval_freq,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Single Task: {game} | Arch: {arch_name} ({arch['description']})")
        print(f"Steps: {total_steps:,} | Seeds: {num_seeds} | PopArt beta: {popart_beta}")
        print(f"Eval freq: {eval_freq:,} | Unified actions: {UNIFIED_ACTIONS}")
        print(f"{'='*60}")

    # Compile
    if verbose:
        print("Compiling...", end=" ", flush=True)
    compile_start = time.time()

    keys = jax.random.split(jax.random.PRNGKey(0), num_seeds)
    vmap_train = jax.jit(jax.vmap(PPOOctaxPopArt.train, in_axes=(None, 0)))
    train_states, evals = vmap_train(ppo, keys)
    jax.block_until_ready(train_states)

    compile_time = time.time() - compile_start
    if verbose:
        print(f"{compile_time:.1f}s")

    # Benchmark
    if verbose:
        print("Running...", end=" ", flush=True)
    start = time.time()
    train_states, evals = vmap_train(ppo, keys)
    jax.block_until_ready(train_states)
    runtime = time.time() - start

    # Extract learning curves
    # evals = (lengths, returns) where each is (seeds, num_evals, num_eval_episodes)
    lengths, returns = evals

    # Compute mean return per eval point per seed: (seeds, num_evals)
    mean_returns_per_eval = returns.mean(axis=-1)  # average over episodes

    # Steps at each eval point
    num_evals = mean_returns_per_eval.shape[1]
    eval_steps = [(i + 1) * eval_freq for i in range(num_evals)]

    # Final returns (last eval point)
    final_returns = mean_returns_per_eval[:, -1]

    # PopArt final stats (per seed, then averaged)
    final_mu = float(train_states.popart_state.mu.mean())
    final_sigma = float(train_states.popart_state.sigma.mean())

    steps_per_sec = total_steps * num_seeds / runtime

    if verbose:
        print(f"done")
        print(f"\nResults:")
        print(f"  Steps/sec: {steps_per_sec:,.0f}")
        print(f"  Final return: {final_returns.mean():.1f} +/- {final_returns.std():.1f}")
        print(f"  PopArt final: mu={final_mu:.2f}, sigma={final_sigma:.2f}")
        # Show learning curve summary
        print(f"  Learning curve: {mean_returns_per_eval.mean(axis=0)[0]:.1f} -> {mean_returns_per_eval.mean(axis=0)[-1]:.1f}")

    return {
        "game": game,
        "arch": arch_name,
        "mlp_hidden_sizes": list(arch["mlp_hidden_sizes"]),
        "steps": total_steps,
        "num_seeds": num_seeds,
        "num_envs": num_envs,
        "popart_beta": popart_beta,
        "eval_freq": eval_freq,
        "unified_actions": UNIFIED_ACTIONS,
        "compile_time": compile_time,
        "runtime": runtime,
        "steps_per_sec": steps_per_sec,
        # Learning curves (for plotting)
        "eval_steps": eval_steps,
        "returns_per_seed": mean_returns_per_eval.tolist(),  # (seeds, num_evals)
        "returns_mean": mean_returns_per_eval.mean(axis=0).tolist(),  # (num_evals,)
        "returns_std": mean_returns_per_eval.std(axis=0).tolist(),  # (num_evals,)
        # Final stats
        "final_return_mean": float(final_returns.mean()),
        "final_return_std": float(final_returns.std()),
        "popart_mu": final_mu,
        "popart_sigma": final_sigma,
    }


# =============================================================================
# Continual Learning Benchmark
# =============================================================================

def run_continual(
    arch_name: str,
    steps_per_task: int,
    num_cycles: int = 1,
    num_envs: int = 512,
    popart_beta: float = 0.0001,
    reset_popart_at_boundary: bool = True,
    task_list: Optional[List[str]] = None,
    verbose: bool = True,
):
    """Run continual learning benchmark with PopArt."""
    arch = ARCH_CONFIGS[arch_name]
    task_list = task_list or TASK_ORDER

    if verbose:
        print(f"\n{'='*60}")
        print(f"Continual Learning | Arch: {arch_name}")
        print(f"Tasks: {' -> '.join(task_list)}")
        print(f"Steps/task: {steps_per_task:,} | Cycles: {num_cycles}")
        print(f"PopArt beta: {popart_beta} | Reset at boundary: {reset_popart_at_boundary}")
        print(f"{'='*60}")

    results = {
        "arch": arch_name,
        "steps_per_task": steps_per_task,
        "num_cycles": num_cycles,
        "reset_popart": reset_popart_at_boundary,
        "task_results": [],
    }

    # Cache PPO instances per task
    ppo_cache = {}
    train_chunk_cache = {}

    def get_ppo_and_train(task_name):
        if task_name not in ppo_cache:
            env, env_params = create_unified_env(task_name)
            ppo = create_popart_ppo(
                env=env,
                env_params=env_params,
                total_timesteps=steps_per_task,
                mlp_hidden_sizes=arch["mlp_hidden_sizes"],
                num_envs=num_envs,
                popart_beta=popart_beta,
                eval_freq=steps_per_task // 4,
            )

            @jax.jit
            def train_chunk(ts, num_iters):
                def body(_, ts):
                    return ppo.train_iteration(ts)
                return jax.lax.fori_loop(0, num_iters, body, ts)

            ppo_cache[task_name] = ppo
            train_chunk_cache[task_name] = train_chunk

        return ppo_cache[task_name], train_chunk_cache[task_name]

    rng = jax.random.PRNGKey(42)
    train_state = None

    for cycle_idx in range(num_cycles):
        if verbose:
            print(f"\n--- Cycle {cycle_idx + 1}/{num_cycles} ---")

        for task_idx, task_name in enumerate(task_list):
            ppo, train_chunk = get_ppo_and_train(task_name)

            # Transfer or initialize state
            if train_state is None:
                rng, init_rng = jax.random.split(rng)
                train_state = ppo.init_state(init_rng)
            else:
                # Transfer weights, reset optimizer and optionally PopArt
                rng, init_rng = jax.random.split(rng)
                new_ts = ppo.init_state(init_rng)

                # Reset PopArt at task boundary (optional)
                if reset_popart_at_boundary:
                    new_popart = PopArtState.create()
                else:
                    new_popart = train_state.popart_state

                train_state = new_ts.replace(
                    agent_ts=new_ts.agent_ts.replace(
                        params=train_state.agent_ts.params,
                    ),
                    popart_state=new_popart,
                )

            if verbose:
                print(f"\nTask: {task_name}", flush=True)
                print(f"  PopArt start: mu={train_state.popart_state.mu:.2f}, "
                      f"sigma={train_state.popart_state.sigma:.2f}")

            # Train
            iteration_steps = ppo.num_envs * ppo.num_steps
            num_iterations = int(np.ceil(steps_per_task / iteration_steps))

            start_time = time.time()
            train_state = train_chunk(train_state, num_iterations)
            jax.block_until_ready(train_state)
            elapsed = time.time() - start_time

            steps_per_sec = steps_per_task / elapsed

            if verbose:
                print(f"  PopArt end: mu={train_state.popart_state.mu:.2f}, "
                      f"sigma={train_state.popart_state.sigma:.2f}")
                print(f"  Time: {elapsed:.1f}s ({steps_per_sec:,.0f} steps/s)")

            results["task_results"].append({
                "cycle": cycle_idx,
                "task": task_name,
                "popart_mu": float(train_state.popart_state.mu),
                "popart_sigma": float(train_state.popart_state.sigma),
                "elapsed": elapsed,
                "steps_per_sec": steps_per_sec,
            })

    return results


# =============================================================================
# Quick Validation
# =============================================================================

def quick_validate(num_envs: int = 256, steps: int = 100_000):
    """Quick validation of all architectures on Brix."""
    print("\n" + "="*70)
    print("QUICK VALIDATION: PopArt PPO on Octax")
    print("="*70)

    results = []
    for arch_name in ["256x1", "64x4", "256x4"]:
        result = run_single_task(
            game="brix",
            arch_name=arch_name,
            total_steps=steps,
            num_envs=num_envs,
            num_seeds=1,
            popart_beta=0.001,  # Faster for quick test
            verbose=True,
        )
        results.append(result)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        print(f"{r['arch']:8s}: return={r['final_return_mean']:6.1f}, "
              f"mu={r['popart_mu']:.2f}, sigma={r['popart_sigma']:.2f}, "
              f"{r['steps_per_sec']:,.0f} steps/s")

    return results


# =============================================================================
# Main
# =============================================================================

def fast_validate():
    """Ultra-fast validation for local testing (no JIT overhead)."""
    from rejax.compat.octax2gymnax import create_octax
    import time

    print("=== Fast Local Validation (no vmap, minimal JIT) ===\n")

    env, env_params = create_octax('brix')

    for arch_name, hidden_sizes in [('256x1', (256,)), ('64x4', (64,64,64,64)), ('256x4', (256,256,256,256))]:
        print(f'{arch_name}:', end=' ', flush=True)

        ppo = PPOOctaxPopArt.create(
            env=env,
            env_params=env_params,
            total_timesteps=1000,
            num_envs=4,
            num_steps=8,
            num_minibatches=2,
            popart_beta=0.01,
            eval_freq=500,
            skip_initial_evaluation=True,
            agent_kwargs={'mlp_hidden_sizes': hidden_sizes},
        )

        rng = jax.random.PRNGKey(0)
        ts = ppo.init_state(rng)

        start = time.time()
        for _ in range(3):
            ts = ppo.train_iteration(ts)
        jax.block_until_ready(ts)
        elapsed = time.time() - start

        print(f'OK (mu={ts.popart_state.mu:.3f}, sigma={ts.popart_state.sigma:.3f}, {elapsed:.1f}s)')

    print("\n=== All architectures validated! ===")


def main():
    parser = argparse.ArgumentParser(description="PopArt PPO validation on Octax")
    parser.add_argument("--fast", action="store_true",
                        help="Ultra-fast validation (local, no vmap)")
    parser.add_argument("--quick", action="store_true", default=False,
                        help="Quick validation with small vmap")
    parser.add_argument("--game", type=str, default="brix",
                        choices=list(OCTAX_GAMES.keys()))
    parser.add_argument("--arch", type=str, default=None,
                        choices=list(ARCH_CONFIGS.keys()),
                        help="Architecture (default: all)")
    parser.add_argument("--steps", type=int, default=100_000,
                        help="Total steps for single task")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--popart-beta", type=float, default=0.001)
    parser.add_argument("--eval-freq", type=int, default=None,
                        help="Evaluation frequency (default: steps/20)")

    # Continual mode
    parser.add_argument("--continual", action="store_true",
                        help="Run continual learning benchmark")
    parser.add_argument("--steps-per-task", type=int, default=500_000)
    parser.add_argument("--num-cycles", type=int, default=1)
    parser.add_argument("--no-reset-popart", action="store_true",
                        help="Don't reset PopArt at task boundaries")

    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")

    args = parser.parse_args()

    if args.fast:
        fast_validate()
        return

    if args.continual:
        # Continual learning mode
        archs = [args.arch] if args.arch else ["256x1", "64x4", "256x4"]
        all_results = {}

        for arch_name in archs:
            result = run_continual(
                arch_name=arch_name,
                steps_per_task=args.steps_per_task,
                num_cycles=args.num_cycles,
                num_envs=args.num_envs,
                popart_beta=args.popart_beta,
                reset_popart_at_boundary=not args.no_reset_popart,
            )
            all_results[arch_name] = result

        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    elif args.arch:
        # Single architecture test
        result = run_single_task(
            game=args.game,
            arch_name=args.arch,
            total_steps=args.steps,
            num_envs=args.num_envs,
            num_seeds=args.num_seeds,
            popart_beta=args.popart_beta,
            eval_freq=args.eval_freq,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)

    else:
        # Quick validation (default)
        results = quick_validate(num_envs=args.num_envs, steps=args.steps)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
