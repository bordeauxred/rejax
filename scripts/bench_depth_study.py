"""
Depth Study: PPO MLP Network Depth Ablation

Studies the effect of network depth on PPO performance with MLP networks.
Compares baseline (ReLU) vs AdaMO (GroupSort + orthogonal optimizer) across depths.

Usage:
    # Quick test
    python scripts/bench_depth_study.py --timesteps 1000000 --num-seeds 1 --depths 2 4

    # Full depth study on Breakout
    python scripts/bench_depth_study.py --timesteps 10000000 --num-seeds 3 --depths 2 4 8 16 32

    # With WandB logging
    python scripts/bench_depth_study.py --timesteps 10000000 --num-seeds 3 --use-wandb
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

import sys
sys.path.insert(0, str(Path(__file__).parent))

from rejax import PPO
from gymnax.environments.minatar.breakout import MinBreakout
from gymnax.environments.minatar.asterix import MinAsterix
from gymnax.environments.minatar.space_invaders import MinSpaceInvaders
from gymnax.environments.minatar.freeway import MinFreeway
from minatar_seaquest_fixed import MinSeaquestFixed
from bench_continual import create_padded_env


# MinAtar game configurations
MINATAR_GAMES = {
    "Breakout-MinAtar": {"channels": 4, "actions": 3, "env_cls": MinBreakout},
    "Asterix-MinAtar": {"channels": 4, "actions": 5, "env_cls": MinAsterix},
    "SpaceInvaders-MinAtar": {"channels": 6, "actions": 4, "env_cls": MinSpaceInvaders},
    "Freeway-MinAtar": {"channels": 7, "actions": 3, "env_cls": MinFreeway},
    "Seaquest-MinAtar": {"channels": 10, "actions": 6, "env_cls": MinSeaquestFixed},
}


def create_native_env(game_name: str) -> Tuple[Any, Any]:
    """Create a native MinAtar environment (no padding)."""
    game_info = MINATAR_GAMES[game_name]
    env = game_info["env_cls"]()
    return env, env.default_params


def create_depth_config(
    depth: int,
    method: str,  # "baseline" or "adamo"
    width: int = 256,
) -> Dict:
    """
    Create experiment config for a given depth and method.

    Args:
        depth: Number of hidden layers
        method: "baseline" (ReLU + standard Adam) or "adamo" (GroupSort + ortho optimizer)
        width: Width of each hidden layer (default 256)

    Returns:
        Experiment config dict compatible with bench_single_strong.py
    """
    hidden_layer_sizes = tuple([width] * depth)

    if method == "baseline":
        return {
            "name": f"mlp_baseline_depth{depth}",
            "network_type": "mlp",
            "hidden_layer_sizes": hidden_layer_sizes,
            "ortho_mode": None,
            "activation": "relu",
            "lr_schedule": "constant",
            "learning_rate": 2.5e-4,
            "num_minibatches": 128,
            "num_epochs": 4,
            "num_steps": 128,
            "num_envs": 2048,
            "use_bias": True,
            "use_orthogonal_init": True,
        }
    elif method == "adamo":
        return {
            "name": f"mlp_adamo_depth{depth}",
            "network_type": "mlp",
            "hidden_layer_sizes": hidden_layer_sizes,
            "ortho_mode": "optimizer",
            "ortho_coeff": 0.1,
            "activation": "groupsort",
            "lr_schedule": "constant",
            "learning_rate": 2.5e-4,
            "num_minibatches": 128,
            "num_epochs": 4,
            "num_steps": 128,
            "num_envs": 2048,
            "use_bias": False,  # No bias for ortho experiments
            "use_orthogonal_init": True,
        }
    else:
        raise ValueError(f"Unknown method: {method}. Use 'baseline' or 'adamo'.")


def create_ppo_from_config(
    env,
    env_params,
    experiment_config: Dict,
    total_timesteps: int,
    eval_freq: int = 100_000,
    num_envs_override: Optional[int] = None,
) -> PPO:
    """Create PPO instance from experiment config."""
    num_envs = num_envs_override or experiment_config.get("num_envs", 2048)

    agent_kwargs = {
        "network_type": "mlp",
        "hidden_layer_sizes": experiment_config["hidden_layer_sizes"],
        "activation": experiment_config.get("activation", "relu"),
        "use_orthogonal_init": experiment_config.get("use_orthogonal_init", True),
        "use_bias": experiment_config.get("use_bias", True),
    }

    # Add ortho mode if specified
    ortho_mode = experiment_config.get("ortho_mode")
    ortho_coeff = experiment_config.get("ortho_coeff", 0.1)

    ppo_config = {
        "env": env,
        "env_params": env_params,
        "agent_kwargs": agent_kwargs,
        "num_envs": num_envs,
        "num_steps": experiment_config.get("num_steps", 128),
        "num_epochs": experiment_config.get("num_epochs", 4),
        "num_minibatches": experiment_config.get("num_minibatches", 128),
        "learning_rate": experiment_config.get("learning_rate", 2.5e-4),
        "anneal_lr": experiment_config.get("anneal_lr", False),
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

    # Add ortho mode
    if ortho_mode:
        ppo_config["ortho_mode"] = ortho_mode
        ppo_config["ortho_coeff"] = ortho_coeff

    return PPO.create(**ppo_config)


@dataclass
class LearningCurve:
    """Store learning curve data for a single seed."""
    seed: int
    steps: np.ndarray
    returns_mean: np.ndarray
    returns_std: np.ndarray
    returns_all: np.ndarray


@dataclass
class DepthResult:
    """Store results for a single depth/method combination."""
    depth: int
    method: str
    config: Dict
    curves: List[LearningCurve]
    compile_time_s: float
    runtime_s: float
    steps_per_second: float

    @property
    def final_return_mean(self) -> float:
        finals = [c.returns_mean[-1] for c in self.curves]
        return float(np.mean(finals))

    @property
    def final_return_std(self) -> float:
        finals = [c.returns_mean[-1] for c in self.curves]
        return float(np.std(finals))

    def to_dict(self) -> Dict:
        return {
            "depth": self.depth,
            "method": self.method,
            "config_name": self.config["name"],
            "hidden_layer_sizes": list(self.config["hidden_layer_sizes"]),
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


def benchmark_depth(
    depth: int,
    method: str,
    game_name: str,
    total_timesteps: int,
    num_seeds: int,
    num_envs: int,
    eval_freq: int,
    use_wandb: bool = False,
    wandb_run = None,
    width: int = 256,
) -> DepthResult:
    """Benchmark a single depth/method combination."""
    config = create_depth_config(depth, method, width)
    config_name = config["name"]

    print(f"\n{'='*70}")
    print(f"Depth: {depth} layers | Method: {method.upper()} | Config: {config_name}")
    print(f"{'='*70}")

    # Create environment
    env, env_params = create_native_env(game_name)

    # Create PPO
    ppo = create_ppo_from_config(
        env=env,
        env_params=env_params,
        experiment_config=config,
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
        num_envs_override=num_envs,
    )

    # Print config
    print(f"  Hidden layers: {config['hidden_layer_sizes']}")
    print(f"  Activation: {config['activation']}")
    print(f"  Ortho mode: {config.get('ortho_mode', 'None')}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Num envs: {num_envs}")

    # Prepare seeds
    keys = jax.random.split(jax.random.PRNGKey(42), num_seeds)

    # Vectorized training
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
    returns_np = np.array(returns)

    num_evals = returns_np.shape[1]
    eval_steps = np.arange(num_evals) * eval_freq

    curves = []
    for seed_idx in range(num_seeds):
        seed_returns = returns_np[seed_idx]
        curves.append(LearningCurve(
            seed=seed_idx,
            steps=eval_steps,
            returns_mean=seed_returns.mean(axis=-1),
            returns_std=seed_returns.std(axis=-1),
            returns_all=seed_returns,
        ))

    result = DepthResult(
        depth=depth,
        method=method,
        config=config,
        curves=curves,
        compile_time_s=compile_time,
        runtime_s=runtime,
        steps_per_second=steps_per_second,
    )

    print(f"  Final return: {result.final_return_mean:.1f} +/- {result.final_return_std:.1f}")

    # Log to wandb
    if use_wandb and wandb_run:
        import wandb

        wandb.log({
            f"{config_name}/final_return_mean": result.final_return_mean,
            f"{config_name}/final_return_std": result.final_return_std,
            f"{config_name}/runtime_s": runtime,
            f"{config_name}/steps_per_second": steps_per_second,
            f"{config_name}/depth": depth,
        })

        # Log learning curves
        mean_curve = np.mean([c.returns_mean for c in curves], axis=0)
        std_curve = np.std([c.returns_mean for c in curves], axis=0)
        for i, (step, mean_ret, std_ret) in enumerate(zip(eval_steps, mean_curve, std_curve)):
            wandb.log({
                f"{config_name}/return_mean": mean_ret,
                f"{config_name}/return_std": std_ret,
                f"{config_name}/step": step,
            })

    return result


def run_depth_study(
    depths: List[int],
    methods: List[str],
    game_name: str,
    total_timesteps: int,
    num_seeds: int,
    num_envs: int,
    eval_freq: int,
    use_wandb: bool,
    wandb_project: str,
    width: int = 256,
) -> Dict:
    """Run full depth study across all depths and methods."""
    results = {
        "experiment": "depth_study_ppo_mlp",
        "description": f"PPO MLP depth ablation on {game_name}",
        "game": game_name,
        "depths": depths,
        "methods": methods,
        "width": width,
        "total_timesteps": total_timesteps,
        "num_seeds": num_seeds,
        "num_envs": num_envs,
        "eval_freq": eval_freq,
        "depth_results": [],
    }

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=wandb_project,
            name=f"depth_study_{game_name}_{time.strftime('%Y%m%d_%H%M%S')}",
            config={
                "experiment": "depth_study_ppo_mlp",
                "game": game_name,
                "depths": depths,
                "methods": methods,
                "width": width,
                "total_timesteps": total_timesteps,
                "num_seeds": num_seeds,
                "num_envs": num_envs,
                "eval_freq": eval_freq,
            },
        )

    for depth in depths:
        for method in methods:
            depth_result = benchmark_depth(
                depth=depth,
                method=method,
                game_name=game_name,
                total_timesteps=total_timesteps,
                num_seeds=num_seeds,
                num_envs=num_envs,
                eval_freq=eval_freq,
                use_wandb=use_wandb,
                wandb_run=wandb_run,
                width=width,
            )
            results["depth_results"].append(depth_result.to_dict())

    if use_wandb:
        import wandb
        wandb.finish()

    return results


def save_results(results: Dict, output_dir: Path) -> Path:
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"depth_study_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return output_file


def print_summary(results: Dict):
    """Print summary table."""
    print("\n" + "=" * 100)
    print(f"{'Depth':<8} {'Method':<12} {'Final Return':>20} {'Runtime':>12} {'Steps/s':>15}")
    print("=" * 100)

    for r in results["depth_results"]:
        ret_str = f"{r['final_return_mean']:.1f} +/- {r['final_return_std']:.1f}"
        print(f"{r['depth']:<8} {r['method']:<12} {ret_str:>20} {r['runtime_s']:>10.1f}s {r['steps_per_second']:>15,.0f}")

    print("=" * 100)

    # Summary by method
    print("\nSummary by method:")
    for method in results["methods"]:
        method_results = [r for r in results["depth_results"] if r["method"] == method]
        mean_return = np.mean([r["final_return_mean"] for r in method_results])
        print(f"  {method.upper()}: mean final return = {mean_return:.1f}")


def plot_depth_study(results: Dict, output_dir: Path):
    """Generate depth study plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Final return vs depth for each method
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"baseline": "tab:blue", "adamo": "tab:orange"}
    markers = {"baseline": "o", "adamo": "s"}

    for method in results["methods"]:
        method_results = [r for r in results["depth_results"] if r["method"] == method]
        depths = [r["depth"] for r in method_results]
        means = [r["final_return_mean"] for r in method_results]
        stds = [r["final_return_std"] for r in method_results]

        ax.errorbar(
            depths, means, yerr=stds,
            label=method.upper(),
            color=colors.get(method, "gray"),
            marker=markers.get(method, "o"),
            capsize=5,
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Network Depth (# hidden layers)", fontsize=12)
    ax.set_ylabel("Final Return", fontsize=12)
    ax.set_title(f"PPO MLP Depth Study on {results['game']}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(results["depths"])
    ax.set_xticklabels(results["depths"])

    plt.tight_layout()
    plot_file = output_dir / f"depth_vs_return_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_file}")
    plt.close()

    # Plot 2: Learning curves for each depth
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    depth_results_by_depth = {}
    for r in results["depth_results"]:
        depth = r["depth"]
        if depth not in depth_results_by_depth:
            depth_results_by_depth[depth] = {}
        depth_results_by_depth[depth][r["method"]] = r

    for idx, depth in enumerate(results["depths"]):
        if idx >= 6:
            break
        ax = axes[idx]

        for method in results["methods"]:
            if depth in depth_results_by_depth and method in depth_results_by_depth[depth]:
                r = depth_results_by_depth[depth][method]
                curves = r["curves"]

                # Aggregate across seeds
                all_means = np.array([c["returns_mean"] for c in curves])
                steps = np.array(curves[0]["steps"])

                mean_curve = all_means.mean(axis=0)
                std_curve = all_means.std(axis=0)

                ax.plot(steps / 1e6, mean_curve,
                       label=method.upper(),
                       color=colors.get(method, "gray"),
                       linewidth=2)
                ax.fill_between(
                    steps / 1e6,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    alpha=0.3,
                    color=colors.get(method, "gray"),
                )

        ax.set_title(f"Depth {depth}")
        ax.set_xlabel("Timesteps (M)")
        ax.set_ylabel("Return")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(results["depths"]), 6):
        axes[i].axis("off")

    plt.suptitle(f"Learning Curves by Depth - {results['game']}", fontsize=14)
    plt.tight_layout()

    plot_file = output_dir / f"learning_curves_by_depth_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="PPO MLP Depth Study - ablation on network depth"
    )
    parser.add_argument(
        "--game", type=str, default="Breakout-MinAtar",
        choices=list(MINATAR_GAMES.keys()),
        help="MinAtar game to benchmark"
    )
    parser.add_argument(
        "--timesteps", type=int, default=25_000_000,
        help="Training timesteps per configuration (default: 25M)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=3,
        help="Number of seeds per configuration"
    )
    parser.add_argument(
        "--num-envs", type=int, default=2048,
        help="Parallel environments"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=100_000,
        help="Evaluation frequency (timesteps)"
    )
    parser.add_argument(
        "--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64],
        help="Network depths to test"
    )
    parser.add_argument(
        "--width", type=int, default=256,
        help="Width of each hidden layer"
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=["baseline", "adamo"],
        choices=["baseline", "adamo"],
        help="Methods to compare"
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
        "--wandb-project", type=str, default="rejax-depth-study-ppo-mlp",
        help="W&B project name"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate plots"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PPO MLP Depth Study")
    print("=" * 70)
    print(f"Game: {args.game}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Seeds: {args.num_seeds}")
    print(f"Depths: {args.depths}")
    print(f"Width: {args.width}")
    print(f"Methods: {args.methods}")
    print(f"Eval freq: {args.eval_freq:,}")
    print("=" * 70)

    results = run_depth_study(
        depths=args.depths,
        methods=args.methods,
        game_name=args.game,
        total_timesteps=args.timesteps,
        num_seeds=args.num_seeds,
        num_envs=args.num_envs,
        eval_freq=args.eval_freq,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        width=args.width,
    )

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"results/depth_study_{args.game.replace('-', '_').lower()}")

    save_results(results, output_dir)
    print_summary(results)

    if args.plot:
        plot_depth_study(results, output_dir)


if __name__ == "__main__":
    main()
