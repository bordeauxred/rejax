"""
Continual learning pipeline with OOM fix.

This script fixes the OOM issue in continual learning by caching the evaluation
function per game. The original bench_continual.py creates a new `act` closure
on every eval_callback call, and since `evaluate()` uses `static_argnames=("act", ...)`,
each new closure triggers a JAX recompilation. With ~40 evals per game and 25 games,
that's ~1000 compilations -> OOM.

FIX: Create a cached eval function per game that takes actor_params as an explicit
pytree argument (not via closure). The function identity stays constant, so JAX
reuses the compilation.

Usage:
    # Smoke test (fast, should complete ~3-5 min)
    python scripts/continual_pipeline.py --steps-per-game 500000 --num-cycles 5 --num-seeds 1

    # Full experiment
    python scripts/continual_pipeline.py --steps-per-game 20000000 --num-cycles 5 --num-seeds 3 --use-wandb
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Import from bench_continual to reuse configs and utilities
from bench_continual import (
    EXPERIMENT_CONFIGS,
    EXPERIMENT_CONFIGS_LEGACY,
    GAME_ORDER,
    ContinualTrainer,
    create_padded_env,
    create_ppo_config,
    print_update_diagnostics,
    save_checkpoint,
    save_results,
)
from rejax import PPO


def create_cached_eval_fn(ppo: PPO):
    """Create a cached evaluation function that doesn't recompile on each call.

    The key insight: the act function in evaluate() is marked as static_argnames,
    so each new act closure triggers recompilation. Instead, we create ONE
    JIT-compiled function that takes params as a regular pytree argument.

    Args:
        ppo: The PPO instance (used to get env, actor network structure)

    Returns:
        A JIT-compiled function: cached_eval(actor_params, rng) -> (lengths, returns)
    """
    env = ppo.env
    env_params = ppo.env_params
    actor = ppo.actor
    max_steps = env_params.max_steps_in_episode

    @jax.jit
    def cached_eval(actor_params, rng, num_episodes=128):
        """Evaluate actor with given params (no recompilation on new params)."""

        def eval_episode(rng):
            """Run single episode, return (length, return)."""
            rng_reset, rng_ep = jax.random.split(rng)
            obs, env_state = env.reset(rng_reset, env_params)

            def step_fn(carry, _):
                obs, env_state, rng, done, ret, length = carry
                rng, rng_act, rng_step = jax.random.split(rng, 3)

                # Use actor directly with params (not via closure)
                obs_batch = jnp.expand_dims(obs, 0)
                action = actor.apply(actor_params, obs_batch, rng_act, method="act")
                action = jnp.squeeze(action)

                next_obs, next_state, reward, next_done, _ = env.step(
                    rng_step, env_state, action, env_params
                )

                # Only update if not already done
                not_done = 1.0 - done.astype(jnp.float32)
                ret = ret + reward * not_done
                length = length + not_done.astype(jnp.int32)

                return (next_obs, next_state, rng, done | next_done, ret, length), None

            init_carry = (obs, env_state, rng_ep, jnp.array(False), 0.0, 0)
            (_, _, _, _, final_return, final_length), _ = jax.lax.scan(
                step_fn, init_carry, None, length=max_steps
            )

            return final_length, final_return

        rngs = jax.random.split(rng, num_episodes)
        lengths, returns = jax.vmap(eval_episode)(rngs)
        return lengths, returns

    return cached_eval


class ContinualTrainerFixed(ContinualTrainer):
    """ContinualTrainer with OOM fix: cached evaluation functions.

    Overrides _get_ppo_for_game to also cache a custom eval function
    that doesn't recompile on each call.
    """

    def _get_ppo_for_game(self, game_name: str) -> Tuple[PPO, Any, Any]:
        """Get cached (PPO, train_chunk, cached_eval) for a game.

        The cached_eval function takes actor_params as an explicit argument
        instead of via closure, avoiding JAX recompilation.
        """
        if game_name not in self._ppo_cache:
            ppo = self._create_ppo_for_game(game_name)

            # Cached train_chunk (same as parent)
            @jax.jit
            def train_chunk(ts, num_iters):
                def body(_, ts):
                    return ppo.train_iteration(ts)
                return jax.lax.fori_loop(0, num_iters, body, ts)

            # NEW: Cached eval function that doesn't recompile
            cached_eval = create_cached_eval_fn(ppo)

            self._ppo_cache[game_name] = (ppo, train_chunk, cached_eval)

        return self._ppo_cache[game_name]

    def train_single_game(self, game_name: str, train_state, rng, cycle_idx: int):
        """Train on a single game using cached eval (OOM fix)."""
        # Get cached PPO, train_chunk, AND cached_eval
        ppo, train_chunk, cached_eval = self._get_ppo_for_game(game_name)

        if train_state is not None:
            rng, init_rng = jax.random.split(rng)
            train_state = self._transfer_train_state(train_state, ppo, init_rng)
        else:
            rng, init_rng = jax.random.split(rng)
            train_state = ppo.init_state(init_rng)

        print(f"  Training on {game_name}...", flush=True)
        start_time = time.time()

        iteration_steps = ppo.num_envs * ppo.num_steps
        num_iterations = int(np.ceil(self.steps_per_game / iteration_steps))
        eval_interval = int(np.ceil(self.eval_freq / iteration_steps))

        # Compile on first call
        print(f"    Compiling...", flush=True)
        compile_start = time.time()

        total_iters = 0
        first_chunk = True
        while total_iters < num_iterations:
            chunk_size = min(eval_interval, num_iterations - total_iters)
            train_state = train_chunk(train_state, chunk_size)
            jax.block_until_ready(train_state)

            if first_chunk:
                compile_time = time.time() - compile_start
                print(f"    Compiled in {compile_time:.1f}s", flush=True)
                first_chunk = False

            # Use CACHED eval instead of ppo.eval_callback (OOM FIX!)
            rng, eval_rng = jax.random.split(rng)
            lengths, returns = cached_eval(train_state.actor_ts.params, eval_rng)

            # Progress logging
            current_steps = (total_iters + chunk_size) * iteration_steps
            elapsed = time.time() - start_time
            steps_per_sec = current_steps / elapsed if elapsed > 0 else 0
            mean_return = float(returns.mean())
            pct = 100 * current_steps / self.steps_per_game
            print(
                f"    [{game_name}] {current_steps:,}/{self.steps_per_game:,} ({pct:.0f}%) "
                f"| return={mean_return:.1f} | {steps_per_sec:,.0f} steps/s",
                flush=True,
            )

            if self.use_wandb:
                import wandb

                game_idx = GAME_ORDER.index(game_name)
                cumulative_step = (
                    cycle_idx * len(GAME_ORDER) * self.steps_per_game
                    + game_idx * self.steps_per_game
                    + current_steps
                )
                wandb.log(
                    {
                        f"train/{game_name}/return": mean_return,
                        f"train/{game_name}/step": current_steps,
                        f"cycle_{cycle_idx}/return": mean_return,
                        f"cycle_{cycle_idx}/game": game_name,
                        "return": mean_return,
                        "cycle": cycle_idx,
                        "game_idx": game_idx,
                        "game": game_name,
                    },
                    step=cumulative_step,
                )

            total_iters += chunk_size

        elapsed = time.time() - start_time
        steps_per_sec = self.steps_per_game / elapsed

        # Final evaluation with cached eval
        rng, eval_rng = jax.random.split(rng)
        lengths, returns = cached_eval(train_state.actor_ts.params, eval_rng)
        final_return = float(returns.mean())

        print(
            f"  Completed {game_name}: final_return={final_return:.1f}, "
            f"elapsed={elapsed:.1f}s, {steps_per_sec:,.0f} steps/s",
            flush=True,
        )

        return train_state, rng, {
            "game": game_name,
            "cycle": cycle_idx,
            "final_return": final_return,
            "elapsed_s": elapsed,
            "steps_per_sec": steps_per_sec,
        }

    def evaluate_all_games(self, train_state, rng, cycle_idx: int, current_game_idx: int):
        """Evaluate current policy on all games using cached eval (OOM fix)."""
        print(f"  Evaluating on all games after game {current_game_idx}...")
        eval_results = {}

        for game_name in GAME_ORDER:
            # Get cached eval for this game
            ppo, _, cached_eval = self._get_ppo_for_game(game_name)

            # Transfer weights to this game's environment
            rng, eval_rng, init_rng = jax.random.split(rng, 3)
            eval_ts = self._transfer_train_state(train_state, ppo, init_rng)

            # Use cached eval (OOM FIX!)
            lengths, returns = cached_eval(eval_ts.actor_ts.params, eval_rng)
            mean_return = float(returns.mean())
            eval_results[game_name] = mean_return
            print(f"    {game_name}: {mean_return:.1f}")

        return eval_results, rng


def main():
    parser = argparse.ArgumentParser(
        description="Continual learning pipeline with OOM fix"
    )
    parser.add_argument(
        "--steps-per-game",
        type=int,
        default=10_000_000,
        help="Training steps per game",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=2,
        help="Number of cycles through all games",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=3, help="Number of random seeds"
    )
    parser.add_argument(
        "--num-envs", type=int, default=2048, help="Parallel environments"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=500_000,
        help="Evaluation frequency in steps",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/continual_fixed",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/continual_fixed",
        help="Directory for results JSON",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="rejax-continual",
        help="W&B project name",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")

    # Get all config names
    all_config_names = [c["name"] for c in EXPERIMENT_CONFIGS] + [
        c["name"] for c in EXPERIMENT_CONFIGS_LEGACY
    ]
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["pgx_baseline"],
        choices=all_config_names,
        help="Experiment configurations to run",
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    # Filter configs
    all_configs = EXPERIMENT_CONFIGS + EXPERIMENT_CONFIGS_LEGACY
    configs_to_run = [c for c in all_configs if c["name"] in args.configs]

    all_experiment_results = {c["name"]: [] for c in configs_to_run}

    for seed_idx in range(args.num_seeds):
        print(f"\n{'='*70}", flush=True)
        print(f"SEED {seed_idx + 1}/{args.num_seeds}", flush=True)
        print(f"{'='*70}", flush=True)

        for experiment_config in configs_to_run:
            config_name = experiment_config["name"]
            print(f"\n{'#'*60}", flush=True)
            print(f"# Config: {config_name} | Seed: {seed_idx}", flush=True)
            print(f"{'#'*60}", flush=True)

            rng = jax.random.PRNGKey(args.seed + seed_idx)

            if args.use_wandb:
                import wandb

                run_name = f"{config_name}_seed{seed_idx}_fixed"
                wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config={
                        "experiment_config": experiment_config,
                        "steps_per_game": args.steps_per_game,
                        "num_cycles": args.num_cycles,
                        "seed": args.seed + seed_idx,
                        "oom_fix": True,
                    },
                    reinit=True,
                )

            # Use the FIXED trainer with cached eval
            trainer = ContinualTrainerFixed(
                config_name=f"{config_name}_seed{seed_idx}",
                experiment_config=experiment_config,
                steps_per_game=args.steps_per_game,
                num_cycles=args.num_cycles,
                num_envs=args.num_envs,
                eval_freq=args.eval_freq,
                checkpoint_dir=checkpoint_dir / config_name,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
            )

            results = trainer.run(rng)
            all_experiment_results[config_name].append(results)

            # Save intermediate results
            save_results(all_experiment_results, output_dir)

            if args.use_wandb:
                import wandb

                wandb.finish()

    # Final save
    save_results(all_experiment_results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    for config_name, results_list in all_experiment_results.items():
        print(f"\n{config_name}:")
        for seed_idx, results in enumerate(results_list):
            final_returns = [r["final_return"] for r in results["per_game_results"]]
            print(f"  Seed {seed_idx}: mean_return={np.mean(final_returns):.1f}")


if __name__ == "__main__":
    main()
