"""
Strong single-task baseline for MinAtar games matching PureJaxRL performance.

Collects full learning curves (5 seeds) for all 5 MinAtar games with:
- Learning rate annealing (linear decay to 0)
- Orthogonal weight initialization
- PureJaxRL-matching hyperparameters

Supports both PADDED (for continual learning comparison) and NATIVE (for pure single-task) modes.

Usage:
    # Smoke test (quick)
    python scripts/bench_single_strong.py --timesteps 500000 --num-seeds 2 --eval-freq 50000

    # Full benchmark - PADDED (matches continual learning setup)
    python scripts/bench_single_strong.py --timesteps 10000000 --num-seeds 5 --padded --use-wandb

    # Full benchmark - NATIVE (pure single-task)
    python scripts/bench_single_strong.py --timesteps 10000000 --num-seeds 5 --use-wandb

    # Single game test
    python scripts/bench_single_strong.py --games Breakout-MinAtar --timesteps 1000000 --num-seeds 3
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnax
import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from rejax import PPO
from gymnax.environments.minatar.breakout import MinBreakout
from gymnax.environments.minatar.asterix import MinAsterix
from gymnax.environments.minatar.space_invaders import MinSpaceInvaders
from gymnax.environments.minatar.freeway import MinFreeway
from minatar_seaquest_fixed import MinSeaquestFixed
from bench_continual import (
    create_padded_env, UNIFIED_CHANNELS, UNIFIED_ACTIONS,
    EXPERIMENT_CONFIGS_BASELINE, EXPERIMENT_CONFIGS, EXPERIMENT_CONFIGS_LEGACY,
    print_update_diagnostics,
)


# MinAtar game configurations (native, not padded for single-task)
MINATAR_GAMES = {
    "Breakout-MinAtar": {"channels": 4, "actions": 3, "env_cls": MinBreakout},
    "Asterix-MinAtar": {"channels": 4, "actions": 5, "env_cls": MinAsterix},
    "SpaceInvaders-MinAtar": {"channels": 6, "actions": 4, "env_cls": MinSpaceInvaders},
    "Freeway-MinAtar": {"channels": 7, "actions": 3, "env_cls": MinFreeway},
    "Seaquest-MinAtar": {"channels": 10, "actions": 6, "env_cls": MinSeaquestFixed},
}

GAME_ORDER = [
    "Breakout-MinAtar",
    "Asterix-MinAtar",
    "SpaceInvaders-MinAtar",
    "Freeway-MinAtar",
    "Seaquest-MinAtar",
]


def create_native_env(game_name: str) -> Tuple[Any, Any]:
    """Create a native MinAtar environment (no padding)."""
    game_info = MINATAR_GAMES[game_name]
    env = game_info["env_cls"]()
    return env, env.default_params


def create_env(game_name: str, padded: bool = False) -> Tuple[Any, Any]:
    """Create MinAtar environment, optionally padded for continual learning."""
    if padded:
        return create_padded_env(game_name)
    return create_native_env(game_name)


def get_experiment_config(config_name: str) -> Dict:
    """Get experiment config by name from bench_continual configs."""
    all_configs = EXPERIMENT_CONFIGS + EXPERIMENT_CONFIGS_LEGACY
    for c in all_configs:
        if c["name"] == config_name:
            return c
    raise ValueError(f"Unknown config: {config_name}. Available: {[c['name'] for c in all_configs]}")


def create_ppo_config_from_experiment(
    env,
    env_params,
    experiment_config: Dict,
    total_timesteps: int,
    eval_freq: int = 100_000,
    num_envs_override: Optional[int] = None,
) -> Dict:
    """
    Create PPO config from an experiment config (from bench_continual.py).

    This uses the same hyperparameters as the continual learning experiments
    to validate single-task performance before running continual.
    """
    network_type = experiment_config.get("network_type", "cnn")
    num_envs = num_envs_override or experiment_config.get("num_envs", 2048)

    if network_type == "cnn":
        agent_kwargs = {
            "network_type": "cnn",
            "conv_channels": experiment_config.get("conv_channels", 32),
            "mlp_hidden_sizes": experiment_config.get("mlp_hidden_sizes", (256, 256, 256, 256)),
            "kernel_size": 3,
            "activation": experiment_config.get("activation", "relu"),
            "use_orthogonal_init": experiment_config.get("use_orthogonal_init", True),
            "use_bias": experiment_config.get("use_bias", True),
        }
    else:
        agent_kwargs = {
            "network_type": "mlp",
            "hidden_layer_sizes": experiment_config.get("hidden_layer_sizes", (256, 256, 256, 256)),
            "activation": experiment_config.get("activation", "relu"),
            "use_orthogonal_init": experiment_config.get("use_orthogonal_init", True),
            "use_bias": experiment_config.get("use_bias", True),
        }

    return {
        "env": env,
        "env_params": env_params,
        "agent_kwargs": agent_kwargs,
        "num_envs": num_envs,
        "num_steps": experiment_config.get("num_steps", 128),
        "num_epochs": experiment_config.get("num_epochs", 4),
        "num_minibatches": experiment_config.get("num_minibatches", 128),
        "learning_rate": experiment_config.get("learning_rate", 3e-4),
        "anneal_lr": experiment_config.get("anneal_lr", True),
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "skip_initial_evaluation": False,
    }


def create_strong_ppo_config(
    env,
    env_params,
    total_timesteps: int,
    eval_freq: int = 100_000,
    num_envs: int = 4096,
    network_type: str = "cnn",
) -> Dict:
    """
    Create PPO config matching CleanRL MinAtar baseline.

    Key features:
    - network_type: "cnn" (default for MinAtar) or "mlp" (for comparison)
    - adam eps=1e-5 (handled in PPO)
    - clip_by_global_norm (handled in PPO)

    CNN architecture (CleanRL MinAtar):
    - Conv: 16 filters, 3x3 kernel, VALID padding -> 8x8x16
    - Flatten -> 1024
    - Dense: 128
    - Activation: ReLU, orthogonal init

    Reference: benchmark/cleanRL/ppo_minatar.py
    """
    if network_type == "cnn":
        # CNN for MinAtar (CleanRL style)
        # Conv(16, k=3, VALID) -> Flatten -> Dense(128)
        agent_kwargs = {
            "network_type": "cnn",
            "conv_channels": 16,
            "mlp_hidden_sizes": (128,),  # Single layer for CleanRL baseline
            "kernel_size": 3,
            "activation": "relu",
            "use_orthogonal_init": True,
        }
    else:
        # MLP for comparison (flattened obs)
        agent_kwargs = {
            "network_type": "mlp",
            "hidden_layer_sizes": (64, 64),
            "activation": "tanh",
            "use_orthogonal_init": True,
        }

    return {
        "env": env,
        "env_params": env_params,
        "agent_kwargs": agent_kwargs,
        # CleanRL MinAtar PPO hyperparameters
        "num_envs": num_envs,
        "num_steps": 128,
        "num_epochs": 4,  # CleanRL default
        "num_minibatches": 4,  # CleanRL default
        "learning_rate": 2.5e-4,  # CleanRL default
        "anneal_lr": True,  # CleanRL default
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.1,  # CleanRL MinAtar uses 0.1
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "skip_initial_evaluation": False,  # Want initial point for curves
    }


def make_progress_callback(game_name: str, seed_idx: int, total_timesteps: int, original_eval_callback):
    """Create a callback that prints progress during training."""
    start_time = [time.time()]

    def progress_callback(ppo, train_state, rng):
        lengths, returns = original_eval_callback(ppo, train_state, rng)

        def log(step, returns):
            elapsed = time.time() - start_time[0]
            pct = 100 * step.item() / total_timesteps
            steps_per_sec = step.item() / elapsed if elapsed > 0 else 0
            mean_ret = returns.mean().item()
            print(f"    [{game_name}|seed{seed_idx}] {step.item():,}/{total_timesteps:,} ({pct:.0f}%) "
                  f"| return={mean_ret:.1f} | {steps_per_sec:,.0f} steps/s")

        jax.experimental.io_callback(log, (), train_state.global_step, returns)
        return lengths, returns

    return progress_callback


@dataclass
class LearningCurve:
    """Store learning curve data for a single seed."""
    seed: int
    steps: np.ndarray  # Timesteps at each eval point
    returns_mean: np.ndarray  # Mean return at each eval point
    returns_std: np.ndarray  # Std of returns at each eval point
    returns_all: np.ndarray  # All eval episode returns (num_evals, num_episodes)


@dataclass
class GameResult:
    """Store results for a single game across all seeds."""
    game: str
    config: Dict
    curves: List[LearningCurve]
    compile_time_s: float
    runtime_s: float
    steps_per_second: float

    @property
    def final_return_mean(self) -> float:
        """Mean final return across seeds."""
        finals = [c.returns_mean[-1] for c in self.curves]
        return float(np.mean(finals))

    @property
    def final_return_std(self) -> float:
        """Std of final returns across seeds."""
        finals = [c.returns_mean[-1] for c in self.curves]
        return float(np.std(finals))

    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
        return {
            "game": self.game,
            "config": {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v
                      for k, v in self.config.items()},
            "num_seeds": len(self.curves),
            "compile_time_s": self.compile_time_s,
            "runtime_s": self.runtime_s,
            "steps_per_second": self.steps_per_second,
            "final_return_mean": self.final_return_mean,
            "final_return_std": self.final_return_std,
            "curves": [
                {
                    "seed": c.seed,
                    "steps": c.steps.tolist(),
                    "returns_mean": c.returns_mean.tolist(),
                    "returns_std": c.returns_std.tolist(),
                }
                for c in self.curves
            ],
        }


def benchmark_single_game(
    game_name: str,
    total_timesteps: int,
    num_seeds: int,
    num_envs: int = 2048,
    eval_freq: int = 100_000,
    use_wandb: bool = False,
    wandb_run = None,
    padded: bool = False,
    network_type: str = "cnn",
    experiment_config: Optional[Dict] = None,
) -> GameResult:
    """Benchmark a single game and collect learning curves."""
    mode_str = "PADDED" if padded else "NATIVE"
    config_name = experiment_config["name"] if experiment_config else network_type.upper()
    print(f"\n{'='*70}")
    print(f"Game: {game_name} ({mode_str}, {config_name})")
    print(f"{'='*70}")

    # Create environment
    env, env_params = create_env(game_name, padded=padded)

    # Create PPO config - use experiment config if provided
    if experiment_config:
        config = create_ppo_config_from_experiment(
            env=env,
            env_params=env_params,
            experiment_config=experiment_config,
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            num_envs_override=num_envs if num_envs != 2048 else None,  # Only override if explicitly set
        )
        # Print update diagnostics
        print_update_diagnostics(experiment_config, total_timesteps)
    else:
        config = create_strong_ppo_config(
            env=env,
            env_params=env_params,
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            num_envs=num_envs,
            network_type=network_type,
        )

    ppo = PPO.create(**config)

    # Print config summary
    print(f"  Config: anneal_lr={ppo.anneal_lr}, clip_eps={ppo.clip_eps}, "
          f"lr={ppo.learning_rate}, num_envs={ppo.num_envs}, "
          f"epochs={ppo.num_epochs}, minibatches={ppo.num_minibatches}")
    if hasattr(ppo.actor, 'conv_channels'):
        print(f"  Network: CNN conv={ppo.actor.conv_channels}, mlp={ppo.actor.mlp_hidden_sizes}, "
              f"ortho_init={ppo.actor.use_orthogonal_init}")
    else:
        print(f"  Network: MLP hidden={ppo.actor.hidden_layer_sizes}, "
              f"ortho_init={ppo.actor.use_orthogonal_init}")

    # Prepare seeds
    keys = jax.random.split(jax.random.PRNGKey(42), num_seeds)

    # Vectorized training across seeds
    vmap_train = jax.jit(jax.vmap(PPO.train, in_axes=(None, 0)))

    # Compile
    print(f"  Compiling...")
    start = time.time()
    _ = vmap_train(ppo, keys)
    jax.block_until_ready(_)
    compile_time = time.time() - start
    print(f"  Compiled in {compile_time:.1f}s")

    # Train
    print(f"  Training {num_seeds} seeds...")
    start = time.time()
    ts, eval_results = vmap_train(ppo, keys)
    jax.block_until_ready(ts)
    runtime = time.time() - start

    steps_per_second = (total_timesteps * num_seeds) / runtime
    print(f"  Completed in {runtime:.1f}s ({steps_per_second:,.0f} steps/s)")

    # Extract learning curves
    lengths, returns = eval_results
    returns_np = np.array(returns)  # (num_seeds, num_evals, num_episodes)

    # Compute eval steps
    num_evals = returns_np.shape[1]
    # First eval is at step 0 (initial), then every eval_freq
    eval_steps = np.arange(num_evals) * eval_freq

    curves = []
    for seed_idx in range(num_seeds):
        seed_returns = returns_np[seed_idx]  # (num_evals, num_episodes)
        curves.append(LearningCurve(
            seed=seed_idx,
            steps=eval_steps,
            returns_mean=seed_returns.mean(axis=-1),
            returns_std=seed_returns.std(axis=-1),
            returns_all=seed_returns,
        ))

    result = GameResult(
        game=game_name,
        config=config,
        curves=curves,
        compile_time_s=compile_time,
        runtime_s=runtime,
        steps_per_second=steps_per_second,
    )

    print(f"  Final return: {result.final_return_mean:.1f} ± {result.final_return_std:.1f}")

    # Log to wandb
    if use_wandb and wandb_run:
        import wandb
        # Log final metrics
        wandb.log({
            f"{game_name}/final_return_mean": result.final_return_mean,
            f"{game_name}/final_return_std": result.final_return_std,
            f"{game_name}/runtime_s": runtime,
            f"{game_name}/steps_per_second": steps_per_second,
        })

        # Log learning curves (mean across seeds)
        mean_curve = np.mean([c.returns_mean for c in curves], axis=0)
        std_curve = np.std([c.returns_mean for c in curves], axis=0)
        for i, (step, mean_ret, std_ret) in enumerate(zip(eval_steps, mean_curve, std_curve)):
            wandb.log({
                f"{game_name}/return_mean": mean_ret,
                f"{game_name}/return_std": std_ret,
                f"{game_name}/step": step,
                "eval_idx": i,
            })

    return result


def run_all_games(
    games: List[str],
    total_timesteps: int,
    num_seeds: int,
    num_envs: int,
    eval_freq: int,
    use_wandb: bool,
    wandb_project: str,
    padded: bool = False,
    network_type: str = "cnn",
    experiment_config: Optional[Dict] = None,
) -> Dict:
    """Run benchmark on all specified games."""
    mode_str = "padded" if padded else "native"
    config_name = experiment_config["name"] if experiment_config else network_type
    results = {
        "experiment": f"single_task_{config_name}_{mode_str}",
        "description": f"PPO with {config_name} config ({mode_str} envs)",
        "padded": padded,
        "config_name": config_name,
        "experiment_config": experiment_config,
        "total_timesteps": total_timesteps,
        "num_seeds": num_seeds,
        "num_envs": num_envs,
        "eval_freq": eval_freq,
        "games": games,
        "per_game_results": [],
    }

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=wandb_project,
            name=f"single_{config_name}_{mode_str}_{time.strftime('%Y%m%d_%H%M%S')}",
            config={
                "experiment": f"single_task_{config_name}_{mode_str}",
                "padded": padded,
                "config_name": config_name,
                "experiment_config": experiment_config,
                "total_timesteps": total_timesteps,
                "num_seeds": num_seeds,
                "num_envs": num_envs,
                "eval_freq": eval_freq,
                "games": games,
            },
        )

    for game_name in games:
        game_result = benchmark_single_game(
            game_name=game_name,
            total_timesteps=total_timesteps,
            num_seeds=num_seeds,
            num_envs=num_envs,
            eval_freq=eval_freq,
            use_wandb=use_wandb,
            wandb_run=wandb_run,
            padded=padded,
            network_type=network_type,
            experiment_config=experiment_config,
        )
        results["per_game_results"].append(game_result.to_dict())

    if use_wandb:
        import wandb
        wandb.finish()

    return results


def save_results(results: Dict, output_dir: Path) -> Path:
    """Save results to JSON with learning curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"strong_baseline_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return output_file


def print_summary(results: Dict):
    """Print summary table."""
    print("\n" + "=" * 90)
    print(f"{'Game':<25} {'Final Return':>20} {'Runtime':>12} {'Steps/s':>15}")
    print("=" * 90)

    for r in results["per_game_results"]:
        ret_str = f"{r['final_return_mean']:.1f} ± {r['final_return_std']:.1f}"
        print(f"{r['game']:<25} {ret_str:>20} {r['runtime_s']:>10.1f}s {r['steps_per_second']:>15,.0f}")

    print("=" * 90)

    # Summary statistics
    all_returns = [r['final_return_mean'] for r in results["per_game_results"]]
    print(f"\nMean across games: {np.mean(all_returns):.1f}")
    print(f"Total runtime: {sum(r['runtime_s'] for r in results['per_game_results']):.1f}s")


def plot_learning_curves(results: Dict, output_dir: Path):
    """Generate learning curve plots (optional, requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, game_result in enumerate(results["per_game_results"]):
        if idx >= 5:
            break

        ax = axes[idx]
        game_name = game_result["game"]
        curves = game_result["curves"]

        # Aggregate across seeds
        all_means = np.array([c["returns_mean"] for c in curves])
        steps = np.array(curves[0]["steps"])

        mean_curve = all_means.mean(axis=0)
        std_curve = all_means.std(axis=0)

        ax.plot(steps / 1e6, mean_curve, label="Mean", linewidth=2)
        ax.fill_between(
            steps / 1e6,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.3,
        )

        # Plot individual seeds faintly
        for seed_idx, curve in enumerate(curves):
            ax.plot(steps / 1e6, curve["returns_mean"], alpha=0.3, linewidth=0.5)

        ax.set_title(game_name.replace("-MinAtar", ""))
        ax.set_xlabel("Timesteps (M)")
        ax.set_ylabel("Return")
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    if len(results["per_game_results"]) < 6:
        axes[5].axis("off")

    plt.suptitle("Strong Baseline: PPO with LR Annealing + Orthogonal Init", fontsize=14)
    plt.tight_layout()

    plot_file = output_dir / f"learning_curves_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_file}")
    plt.close()


def main():
    # Get all available config names
    all_configs = EXPERIMENT_CONFIGS + EXPERIMENT_CONFIGS_LEGACY
    all_config_names = [c["name"] for c in all_configs]

    parser = argparse.ArgumentParser(
        description="Single-task MinAtar baseline for validating configs before continual learning"
    )
    parser.add_argument(
        "--games", nargs="+", default=GAME_ORDER,
        choices=GAME_ORDER, help="Games to benchmark"
    )
    parser.add_argument(
        "--timesteps", type=int, default=10_000_000,
        help="Training timesteps per game (default: 10M, pgx uses 20M)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=3,
        help="Number of seeds per game"
    )
    parser.add_argument(
        "--num-envs", type=int, default=2048,
        help="Parallel environments (override config default)"
    )
    parser.add_argument(
        "--config", type=str, default="pgx_baseline",
        choices=all_config_names,
        help="Experiment config from bench_continual.py (recommended: pgx_baseline, mlp_baseline)"
    )
    parser.add_argument(
        "--network-type", type=str, default="cnn",
        choices=["cnn", "mlp"],
        help="Network type (only used if --config is not specified)"
    )
    parser.add_argument(
        "--legacy-mode", action="store_true",
        help="Use legacy CleanRL config instead of experiment configs"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=100_000,
        help="Evaluation frequency (timesteps)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for results"
    )
    parser.add_argument(
        "--use-wandb", action="store_true",
        help="Enable W&B logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="rejax-minatar",
        help="W&B project name"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate learning curve plots"
    )
    parser.add_argument(
        "--padded", action="store_true",
        help="Use padded environments (unified obs/action space for continual learning comparison)"
    )

    args = parser.parse_args()

    # Get experiment config if not in legacy mode
    experiment_config = None
    if not args.legacy_mode:
        experiment_config = get_experiment_config(args.config)

    mode_str = "PADDED" if args.padded else "NATIVE"
    config_name = experiment_config["name"] if experiment_config else args.network_type.upper()

    print("=" * 70)
    print(f"MinAtar Single-Task Baseline - {mode_str}")
    print("=" * 70)
    print(f"Config: {config_name}")
    print(f"Games: {args.games}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Seeds: {args.num_seeds}")
    print(f"Eval freq: {args.eval_freq:,}")
    print(f"Padded: {args.padded}")
    if experiment_config:
        print(f"Network: {experiment_config.get('network_type', 'cnn').upper()}")
        print(f"num_envs: {experiment_config.get('num_envs', args.num_envs)}")
        print(f"num_epochs: {experiment_config.get('num_epochs', 4)}")
        print(f"num_minibatches: {experiment_config.get('num_minibatches', 128)}")
        print(f"learning_rate: {experiment_config.get('learning_rate', 3e-4)}")
        print(f"anneal_lr: {experiment_config.get('anneal_lr', True)}")
    print("=" * 70)

    results = run_all_games(
        games=args.games,
        total_timesteps=args.timesteps,
        num_seeds=args.num_seeds,
        num_envs=args.num_envs,
        eval_freq=args.eval_freq,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        padded=args.padded,
        network_type=args.network_type,
        experiment_config=experiment_config,
    )

    # Default output dir based on config
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        mode_suffix = "padded" if args.padded else "native"
        output_dir = Path(f"results/single_task_{config_name}_{mode_suffix}")

    save_results(results, output_dir)
    print_summary(results)

    if args.plot:
        plot_learning_curves(results, output_dir)


if __name__ == "__main__":
    main()
