"""
Fetch learning curves from wandb and plot Lyle-style per-game curves.

X-axis: steps within game (0-10M), normalized from global step
One plot per game
Dashed = cycle 0 (first time), Solid = cycle 1 (second time)

Usage:
    python scripts/fetch_wandb_curves.py [--project PROJECT] [--output-dir DIR]
"""

import os
import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

os.environ['WANDB_API_KEY'] = '8f41a578cf88610097523879e5e2baef5462a723'

plt.style.use('seaborn-v0_8-whitegrid')

# Legacy config colors/labels
LEGACY_COLORS = {
    'baseline': '#1f77b4',
    'ortho_adamo': '#ff7f0e',
    'ortho_adamo_lyle_lr': '#2ca02c',
}
LEGACY_LABELS = {
    'baseline': 'Baseline',
    'ortho_adamo': 'Ortho+GS',
    'ortho_adamo_lyle_lr': 'Ortho+GS+LyleLR',
}

# MLP config colors/labels
MLP_COLORS = {
    'mlp_baseline': '#1f77b4',       # blue
    'mlp_adamo': '#ff7f0e',          # orange
    'mlp_adamo_lyle_continual': '#2ca02c',  # green
}
MLP_LABELS = {
    'mlp_baseline': 'MLP Baseline',
    'mlp_adamo': 'MLP AdaMO+GS',
    'mlp_adamo_lyle_continual': 'MLP AdaMO+LyleLR',
}

# CNN config colors/labels
CNN_COLORS = {
    'cnn_baseline': '#d62728',       # red
    'cnn_adamo': '#9467bd',          # purple
    'cnn_adamo_lyle_continual': '#8c564b',  # brown
}
CNN_LABELS = {
    'cnn_baseline': 'CNN Baseline',
    'cnn_adamo': 'CNN AdaMO',
    'cnn_adamo_lyle_continual': 'CNN AdaMO+LyleLR',
}

# Combined colors/labels for mixed MLP+CNN plots
ALL_COLORS = {**MLP_COLORS, **CNN_COLORS}
ALL_LABELS = {**MLP_LABELS, **CNN_LABELS}

def get_colors_labels(configs):
    """Auto-detect config type and return appropriate colors/labels."""
    has_mlp = any(c.startswith('mlp_') for c in configs)
    has_cnn = any(c.startswith('cnn_') for c in configs)

    if has_mlp and has_cnn:
        return ALL_COLORS, ALL_LABELS
    elif has_mlp:
        return MLP_COLORS, MLP_LABELS
    elif has_cnn:
        return CNN_COLORS, CNN_LABELS
    else:
        return LEGACY_COLORS, LEGACY_LABELS

# For backwards compatibility
COLORS = LEGACY_COLORS
LABELS = LEGACY_LABELS

GAMES = ['Breakout-MinAtar', 'Asterix-MinAtar', 'SpaceInvaders-MinAtar', 'Freeway-MinAtar', 'Seaquest-MinAtar']
GAME_SHORT = ['Breakout', 'Asterix', 'SpaceInvaders', 'Freeway', 'Seaquest']

# Global step offsets for each game (10M per game)
GAME_OFFSETS = {
    0: {game: i * 10_000_000 for i, game in enumerate(GAMES)},  # Cycle 0
    1: {game: (5 + i) * 10_000_000 for i, game in enumerate(GAMES)},  # Cycle 1
}


def fetch_wandb_data(project='2robert-mueller-none/rejax-ppo-continual-minatar', filter_configs=None):
    """Fetch all run data from wandb."""
    api = wandb.Api()
    runs = api.runs(project)

    data = defaultdict(list)
    printed_configs = set()

    for run in runs:
        config_name = run.config.get('experiment_config', {}).get('name', 'unknown')

        # Filter to specific configs if requested
        if filter_configs and config_name not in filter_configs:
            continue

        history = run.history(samples=5000)

        print(f"Fetching {run.name}... ({len(history)} rows)")

        # Debug: print column names for first run of each config type
        if config_name not in printed_configs:
            train_cols = [c for c in history.columns if 'train/' in c or 'cycle_' in c]
            print(f"  [{config_name}] Relevant columns: {train_cols[:10]}")
            printed_configs.add(config_name)

        data[config_name].append({
            'name': run.name,
            'history': history,
        })

    return data


def extract_game_curves(data):
    """
    Extract per-game learning curves with normalized steps (0-10M per game).
    Supports both old format (cycle_{N}/{game}/step) and new format (train/{game}/step with cycle field).
    """
    curves = {}

    for config_name, runs in data.items():
        curves[config_name] = {game: {0: [], 1: []} for game in GAMES}

        for run_data in runs:
            history = run_data['history']

            # Try new format first: train/{game}/step with separate cycle column
            for game in GAMES:
                step_col = f'train/{game}/step'
                return_col = f'train/{game}/return'

                if step_col in history.columns and return_col in history.columns:
                    # New format - need to separate by cycle
                    for cycle in [0, 1]:
                        if 'cycle' in history.columns:
                            mask = (history[step_col].notna() &
                                   history[return_col].notna() &
                                   (history['cycle'] == cycle))
                        else:
                            # No cycle column - infer from step values
                            # Cycle 0: steps 0-50M, Cycle 1: steps 50M-100M
                            cycle_start = cycle * 50_000_000
                            cycle_end = (cycle + 1) * 50_000_000
                            mask = (history[step_col].notna() &
                                   history[return_col].notna())
                            # Will filter by offset later

                        if mask.sum() > 0:
                            steps = history.loc[mask, step_col].values
                            returns = history.loc[mask, return_col].values

                            if len(steps) > 0:
                                # Steps are already local to the game (0-10M range)
                                # Sort by steps
                                sorted_idx = np.argsort(steps)
                                local_steps = steps[sorted_idx]
                                returns = returns[sorted_idx]

                                curves[config_name][game][cycle].append({
                                    'steps': local_steps,
                                    'returns': returns,
                                })
                    continue  # Found new format, skip old format check

                # Fall back to old format: cycle_{N}/{game}/step
                for cycle in [0, 1]:
                    step_col_old = f'cycle_{cycle}/{game}/step'
                    return_col_old = f'cycle_{cycle}/{game}/return'

                    if step_col_old in history.columns and return_col_old in history.columns:
                        mask = history[step_col_old].notna() & history[return_col_old].notna()
                        global_steps = history.loc[mask, step_col_old].values
                        returns = history.loc[mask, return_col_old].values

                        if len(global_steps) > 0:
                            # Normalize to 0-10M by subtracting offset
                            offset = GAME_OFFSETS[cycle][game]
                            local_steps = global_steps - offset

                            # Sort by steps
                            sorted_idx = np.argsort(local_steps)
                            local_steps = local_steps[sorted_idx]
                            returns = returns[sorted_idx]

                            curves[config_name][game][cycle].append({
                                'steps': local_steps,
                                'returns': returns,
                            })

    return curves


def plot_lyle_curves(curves, output_dir, colors=None, labels=None):
    """
    Lyle-style: one plot per game, showing learning curves.
    Dashed = cycle 0, Solid = cycle 1.
    """
    colors = colors or COLORS
    labels = labels or LABELS

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (game, short_name) in enumerate(zip(GAMES, GAME_SHORT)):
        ax = axes[idx]

        for config in curves.keys():
            label = labels.get(config, config)
            color = colors.get(config, '#333333')

            for cycle in [0, 1]:
                all_curves = curves[config][game][cycle]
                if not all_curves:
                    continue

                # Plot each seed with low alpha, then plot mean
                all_steps = []
                all_returns = []

                for curve_data in all_curves:
                    steps = curve_data['steps'] / 1_000_000  # Convert to M
                    returns = curve_data['returns']

                    all_steps.append(steps)
                    all_returns.append(returns)

                    # Individual seed (faint)
                    linestyle = '--' if cycle == 0 else '-'
                    ax.plot(steps, returns, linestyle=linestyle, color=color,
                           alpha=0.2, linewidth=1)

                # Compute and plot mean (interpolated)
                if all_curves:
                    x_common = np.linspace(0, 10, 50)
                    interpolated = []
                    for steps, returns in zip(all_steps, all_returns):
                        if len(steps) > 1:
                            interp = np.interp(x_common, steps, returns,
                                             left=returns[0], right=returns[-1])
                            interpolated.append(interp)

                    if interpolated:
                        mean_returns = np.mean(interpolated, axis=0)
                        std_returns = np.std(interpolated, axis=0)

                        linestyle = '--' if cycle == 0 else '-'
                        linewidth = 2 if cycle == 0 else 2.5
                        alpha = 0.8 if cycle == 0 else 1.0

                        cycle_label = f'{label}' if cycle == 1 and idx == 0 else None
                        ax.plot(x_common, mean_returns, linestyle=linestyle,
                               color=color, linewidth=linewidth, alpha=alpha,
                               label=cycle_label)
                        ax.fill_between(x_common, mean_returns - std_returns,
                                       mean_returns + std_returns,
                                       color=color, alpha=0.1)

        ax.set_title(short_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Steps (M)')
        ax.set_ylabel('Return')
        ax.set_xlim(0, 10)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc='best', fontsize=9)

    # Legend in 6th subplot
    axes[5].axis('off')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Cycle 0 (1st time)'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5, label='Cycle 1 (2nd time)'),
    ]
    for config in curves.keys():
        label = labels.get(config, config)
        color = colors.get(config, '#333333')
        legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=label))

    axes[5].legend(handles=legend_elements, loc='center', fontsize=11)
    axes[5].set_title('Legend', fontsize=12)

    plt.suptitle('Per-Game Learning Curves\nComparing 1st vs 2nd time training on each game',
                fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'lyle_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'lyle_training_curves.png'}")


def plot_single_row(curves, output_dir, colors=None, labels=None):
    """Single row version for papers."""
    colors = colors or COLORS
    labels = labels or LABELS

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))

    for idx, (game, short_name) in enumerate(zip(GAMES, GAME_SHORT)):
        ax = axes[idx]

        for config in curves.keys():
            label = labels.get(config, config)
            color = colors.get(config, '#333333')

            for cycle in [0, 1]:
                all_curves = curves[config][game][cycle]
                if not all_curves:
                    continue

                all_steps = []
                all_returns = []

                for curve_data in all_curves:
                    steps = curve_data['steps'] / 1_000_000
                    returns = curve_data['returns']
                    all_steps.append(steps)
                    all_returns.append(returns)

                # Compute mean
                x_common = np.linspace(0, 10, 50)
                interpolated = []
                for steps, returns in zip(all_steps, all_returns):
                    if len(steps) > 1:
                        interp = np.interp(x_common, steps, returns,
                                         left=returns[0], right=returns[-1])
                        interpolated.append(interp)

                if interpolated:
                    mean_returns = np.mean(interpolated, axis=0)
                    std_returns = np.std(interpolated, axis=0)

                    linestyle = '--' if cycle == 0 else '-'
                    linewidth = 1.5 if cycle == 0 else 2.5
                    alpha = 0.7 if cycle == 0 else 1.0

                    ax.plot(x_common, mean_returns, linestyle=linestyle,
                           color=color, linewidth=linewidth, alpha=alpha)
                    ax.fill_between(x_common, mean_returns - std_returns,
                                   mean_returns + std_returns,
                                   color=color, alpha=0.08 if cycle == 0 else 0.12)

        ax.set_title(short_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Steps (M)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Return', fontsize=10)
        ax.set_xlim(0, 10)
        ax.grid(True, alpha=0.3)

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for config in curves.keys():
        lbl = labels.get(config, config)
        clr = colors.get(config, '#333333')
        legend_elements.append(Line2D([0], [0], color=clr, linewidth=2.5, label=lbl))
    legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Cycle 0'))
    legend_elements.append(Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5, label='Cycle 1'))

    fig.legend(handles=legend_elements, loc='upper center', ncol=5,
              fontsize=9, bbox_to_anchor=(0.5, 1.12))

    plt.suptitle('Per-Game Learning Curves: Cycle 0 (dashed) vs Cycle 1 (solid)',
                fontsize=13, y=1.18)
    plt.tight_layout()
    plt.savefig(output_dir / 'lyle_training_curves_row.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'lyle_training_curves_row.png'}")


def main():
    parser = argparse.ArgumentParser(description="Fetch wandb curves and generate plots")
    parser.add_argument('--project', type=str, default='2robert-mueller-none/rejax-ppo-continual-minatar',
                        help='Wandb project path (e.g., user/project)')
    parser.add_argument('--output-dir', type=str, default='plots_continual',
                        help='Output directory for plots')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        help='Filter to specific config names (e.g., mlp_baseline cnn_baseline)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching data from wandb project: {args.project}")
    data = fetch_wandb_data(args.project, filter_configs=args.configs)

    print("\nExtracting curves...")
    curves = extract_game_curves(data)

    # Auto-detect colors/labels based on config names
    colors, labels = get_colors_labels(list(curves.keys()))

    print("\nGenerating plots...")
    plot_lyle_curves(curves, output_dir, colors, labels)
    plot_single_row(curves, output_dir, colors, labels)

    print(f"\nDone! Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
