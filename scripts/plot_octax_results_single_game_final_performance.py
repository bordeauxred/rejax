#!/usr/bin/env python3
"""
Plot Octax single-task results from wandb.

Usage:
    python scripts/plot_octax_results.py
    python scripts/plot_octax_results.py --from-wandb  # Fetch from wandb API
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_local_results(results_dir: Path) -> pd.DataFrame:
    """Load results from synced wandb folders."""
    records = []

    for run_dir in results_dir.glob("run-*"):
        config_file = run_dir / "files" / "config.yaml"
        summary_file = run_dir / "files" / "wandb-summary.json"

        if not config_file.exists() or not summary_file.exists():
            continue

        with open(config_file) as f:
            config = yaml.safe_load(f)

        with open(summary_file) as f:
            summary = json.load(f)

        # Extract config values
        game = config.get("game", {}).get("value", "unknown")
        mlp_str = config.get("mlp_str", {}).get("value", "unknown")
        normalize_rewards = config.get("normalize_rewards", {}).get("value", False)
        num_seeds = config.get("num_seeds", {}).get("value", 1)
        total_timesteps = config.get("total_timesteps", {}).get("value", 0)

        # Skip test runs (< 1M steps)
        if total_timesteps < 1_000_000:
            continue

        # Extract results
        mean_return = summary.get("eval/mean_return", 0)
        std_return = summary.get("eval/std_return", 0)
        steps_per_sec = summary.get("throughput/steps_per_sec", 0)

        records.append({
            "game": game,
            "mlp": mlp_str,
            "norm": "norm" if normalize_rewards else "no_norm",
            "mean_return": mean_return,
            "std_return": std_return,
            "steps_per_sec": steps_per_sec,
            "num_seeds": num_seeds,
            "total_timesteps": total_timesteps,
        })

    return pd.DataFrame(records)


def load_wandb_results(project: str = "octax-single-task") -> pd.DataFrame:
    """Load results from wandb API."""
    import wandb

    api = wandb.Api()
    runs = api.runs(project)

    records = []
    for run in runs:
        config = run.config
        summary = run.summary._json_dict

        # Skip test runs
        total_timesteps = config.get("total_timesteps", 0)
        if total_timesteps < 1_000_000:
            continue

        records.append({
            "game": config.get("game", "unknown"),
            "mlp": config.get("mlp_str", "unknown"),
            "norm": "norm" if config.get("normalize_rewards", False) else "no_norm",
            "mean_return": summary.get("eval/mean_return", 0),
            "std_return": summary.get("eval/std_return", 0),
            "steps_per_sec": summary.get("throughput/steps_per_sec", 0),
            "num_seeds": config.get("num_seeds", 1),
            "total_timesteps": total_timesteps,
        })

    return pd.DataFrame(records)


def plot_returns_by_game(df: pd.DataFrame, output_dir: Path, log_scale: bool = False):
    """Create bar plot of returns by game, grouped by MLP and norm."""
    output_dir.mkdir(parents=True, exist_ok=True)

    games = sorted(df["game"].unique())
    mlps = sorted(df["mlp"].unique())
    norms = ["norm", "no_norm"]

    # Create figure with subplots for each MLP config
    fig, axes = plt.subplots(1, len(mlps), figsize=(8 * len(mlps), 10), sharey=True)
    if len(mlps) == 1:
        axes = [axes]

    for ax, mlp in zip(axes, mlps):
        mlp_df = df[df["mlp"] == mlp]

        x = np.arange(len(games))
        width = 0.35

        for i, norm in enumerate(norms):
            norm_df = mlp_df[mlp_df["norm"] == norm]
            returns = []
            stds = []
            for game in games:
                game_df = norm_df[norm_df["game"] == game]
                if len(game_df) > 0:
                    returns.append(game_df["mean_return"].values[0])
                    stds.append(game_df["std_return"].values[0])
                else:
                    returns.append(0)
                    stds.append(0)

            offset = (i - 0.5) * width
            label = "Reward Norm" if norm == "norm" else "No Norm"
            # For log scale, shift values up by 1 to handle zeros/negatives
            plot_returns = [max(r + 1, 0.1) for r in returns] if log_scale else returns
            bars = ax.bar(x + offset, plot_returns, width, label=label, capsize=3)

            # Add value labels on bars
            for bar, ret in zip(bars, returns):
                height = bar.get_height()
                if log_scale:
                    ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                            f'{ret:.1f}', ha='center', va='bottom', fontsize=6, rotation=90)
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                            f'{ret:.1f}', ha='center', va='bottom', fontsize=6, rotation=90)

        ax.set_xlabel("Game")
        if log_scale:
            ax.set_ylabel("Mean Return + 1 (log scale)")
            ax.set_yscale("log")
        else:
            ax.set_ylabel("Mean Return")
        ax.set_title(f"MLP: {mlp}")
        ax.set_xticks(x)
        ax.set_xticklabels(games, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    suffix = "_log" if log_scale else ""
    plt.suptitle(f"Octax Single-Task Returns (PPOOctax, 5M steps, 2 seeds){' - Log Scale' if log_scale else ''}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"octax_returns_by_game{suffix}.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"octax_returns_by_game{suffix}.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir}/octax_returns_by_game{suffix}.png")
    plt.close()


def plot_throughput_by_game(df: pd.DataFrame, output_dir: Path):
    """Create bar plot of throughput by game."""
    output_dir.mkdir(parents=True, exist_ok=True)

    games = sorted(df["game"].unique())
    mlps = sorted(df["mlp"].unique())

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(games))
    width = 0.35

    for i, mlp in enumerate(mlps):
        mlp_df = df[(df["mlp"] == mlp) & (df["norm"] == "norm")]  # Just use norm runs
        throughputs = []
        for game in games:
            game_df = mlp_df[mlp_df["game"] == game]
            if len(game_df) > 0:
                throughputs.append(game_df["steps_per_sec"].values[0] / 1000)  # K steps/sec
            else:
                throughputs.append(0)

        offset = (i - 0.5) * width
        ax.bar(x + offset, throughputs, width, label=f"MLP: {mlp}")

    ax.set_xlabel("Game")
    ax.set_ylabel("Throughput (K steps/sec)")
    ax.set_title("Octax Training Throughput by Game")
    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "octax_throughput.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir}/octax_throughput.png")
    plt.close()


def plot_norm_comparison(df: pd.DataFrame, output_dir: Path):
    """Scatter plot comparing norm vs no_norm returns."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only games with both norm and no_norm
    games_with_both = []
    for game in df["game"].unique():
        game_df = df[df["game"] == game]
        if len(game_df["norm"].unique()) == 2:
            games_with_both.append(game)

    if not games_with_both:
        print("No games with both norm and no_norm runs, skipping comparison plot")
        return

    mlps = sorted(df["mlp"].unique())

    fig, axes = plt.subplots(1, len(mlps), figsize=(6 * len(mlps), 5))
    if len(mlps) == 1:
        axes = [axes]

    for ax, mlp in zip(axes, mlps):
        mlp_df = df[df["mlp"] == mlp]

        norm_returns = []
        no_norm_returns = []
        labels = []

        for game in games_with_both:
            game_df = mlp_df[mlp_df["game"] == game]
            norm_val = game_df[game_df["norm"] == "norm"]["mean_return"].values
            no_norm_val = game_df[game_df["norm"] == "no_norm"]["mean_return"].values

            if len(norm_val) > 0 and len(no_norm_val) > 0:
                norm_returns.append(norm_val[0])
                no_norm_returns.append(no_norm_val[0])
                labels.append(game)

        ax.scatter(no_norm_returns, norm_returns, s=100)
        for i, label in enumerate(labels):
            ax.annotate(label, (no_norm_returns[i], norm_returns[i]), fontsize=8)

        # Diagonal line
        max_val = max(max(norm_returns), max(no_norm_returns)) * 1.1
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="y=x")

        ax.set_xlabel("No Norm Return")
        ax.set_ylabel("Norm Return")
        ax.set_title(f"MLP: {mlp}")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.suptitle("Effect of Reward Normalization", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "octax_norm_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir}/octax_norm_comparison.png")
    plt.close()


def plot_learning_curves(curves_dir: Path, output_dir: Path):
    """Plot learning curves from saved npz files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_files = list(curves_dir.glob("*_curves.npz"))
    if not curve_files:
        print(f"No learning curve files found in {curves_dir}")
        return

    print(f"Found {len(curve_files)} learning curve files")

    # Group by game
    games = {}
    for f in curve_files:
        data = np.load(f, allow_pickle=True)
        config = data["config"].item()
        game = config["game"]
        if game not in games:
            games[game] = []
        games[game].append((f, data, config))

    # Plot each game
    for game, runs in games.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for f, data, config in runs:
            timesteps = data["timesteps"]
            mean_returns = data["mean_returns"]  # (num_seeds, num_evals)
            mlp_str = config["mlp_str"]
            norm_str = "norm" if config["normalize_rewards"] else "no_norm"

            # Average over seeds
            mean = mean_returns.mean(axis=0)
            std = mean_returns.std(axis=0)

            label = f"{mlp_str} ({norm_str})"
            ax.plot(timesteps / 1e6, mean, label=label)
            ax.fill_between(timesteps / 1e6, mean - std, mean + std, alpha=0.2)

        ax.set_xlabel("Timesteps (M)")
        ax.set_ylabel("Mean Return")
        ax.set_title(f"Learning Curves: {game}")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"learning_curve_{game}.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir}/learning_curve_{game}.png")
        plt.close()

    # Combined plot with all games (subplots)
    n_games = len(games)
    if n_games > 1:
        cols = min(4, n_games)
        rows = (n_games + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = np.array(axes).flatten() if n_games > 1 else [axes]

        for idx, (game, runs) in enumerate(sorted(games.items())):
            ax = axes[idx]
            for f, data, config in runs:
                timesteps = data["timesteps"]
                mean_returns = data["mean_returns"]
                mlp_str = config["mlp_str"]

                mean = mean_returns.mean(axis=0)
                ax.plot(timesteps / 1e6, mean, label=mlp_str)

            ax.set_title(game)
            ax.set_xlabel("Steps (M)")
            ax.set_ylabel("Return")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        # Hide empty subplots
        for idx in range(len(games), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Octax Learning Curves", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "learning_curves_all.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir}/learning_curves_all.png")
        plt.close()


def print_summary_table(df: pd.DataFrame):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("OCTAX SINGLE-TASK RESULTS SUMMARY")
    print("=" * 80)

    for mlp in sorted(df["mlp"].unique()):
        print(f"\n--- MLP: {mlp} ---")
        mlp_df = df[df["mlp"] == mlp]

        pivot = mlp_df.pivot_table(
            values="mean_return",
            index="game",
            columns="norm",
            aggfunc="first"
        )
        print(pivot.to_string())

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-wandb", action="store_true", help="Fetch from wandb API")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/octax_overnight"),
                        help="Local results directory")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/octax_plots"),
                        help="Output directory for plots")
    parser.add_argument("--curves-dir", type=Path, default=None,
                        help="Directory with *_curves.npz files for learning curves")
    args = parser.parse_args()

    print("Loading results...")
    if args.from_wandb:
        df = load_wandb_results()
    else:
        df = load_local_results(args.results_dir)

    print(f"Loaded {len(df)} runs")

    if len(df) == 0:
        print("No results found!")
        return

    print_summary_table(df)

    print("\nGenerating plots...")
    plot_returns_by_game(df, args.output_dir, log_scale=False)
    plot_returns_by_game(df, args.output_dir, log_scale=True)
    plot_throughput_by_game(df, args.output_dir)
    plot_norm_comparison(df, args.output_dir)

    # Save CSV
    df.to_csv(args.output_dir / "octax_results.csv", index=False)
    print(f"Saved: {args.output_dir}/octax_results.csv")

    # Plot learning curves if directory provided
    if args.curves_dir:
        print("\nPlotting learning curves...")
        plot_learning_curves(args.curves_dir, args.output_dir)


if __name__ == "__main__":
    main()
