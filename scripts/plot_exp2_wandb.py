#!/usr/bin/env python3
"""
Plot learning curves for Experiment 2: Channel Permutation (10 cycles).
Fetches data from wandb. Similar style to plot_continual_4cycle.py.

For 10 cycles, uses color gradient (light→dark) instead of line styles.
"""
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
import os

import matplotlib
matplotlib.use('Agg')

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Config
PROJECT = "continual_minatar_permutations"  # exp2 wandb project
GAMES = ['Breakout-MinAtar', 'Asterix-MinAtar', 'SpaceInvaders-MinAtar', 'Freeway-MinAtar']
STEPS_PER_GAME = 25_000_000
NUM_CYCLES = 10
OUTPUT_DIR = '/Users/robertmueller/Desktop/agents/rejax/results/channel_permutation_4games_256x4_10cycles/plots'

# Method styling
METHOD_STYLES = {
    'mlp_baseline': ('tab:blue', 'Baseline (ReLU)'),
    'mlp_adamo': ('tab:orange', 'AdaMO (GroupSort)'),
}

# Line styles for 10 cycles - maximally distinct
CYCLE_LINESTYLES = {
    0: '-',                      # solid
    1: '--',                     # dashed
    2: ':',                      # dotted
    3: '-.',                     # dashdot
    4: (0, (5, 10)),             # long dash
    5: (0, (5, 1)),              # dense dash
    6: (0, (1, 1)),              # dense dot
    7: (0, (3, 5, 1, 5)),        # dash dot
    8: (0, (3, 1, 1, 1, 1, 1)),  # dash dot dot
    9: (0, (5, 2, 1, 2)),        # dash dot (variant)
}

CYCLE_LABELS = {
    0: 'Cycle 0', 1: 'Cycle 1', 2: 'Cycle 2', 3: 'Cycle 3', 4: 'Cycle 4',
    5: 'Cycle 5', 6: 'Cycle 6', 7: 'Cycle 7', 8: 'Cycle 8', 9: 'Cycle 9',
}


def get_run_data(run):
    """Extract training curves from a wandb run."""
    history = run.history(samples=50000)
    data = defaultdict(lambda: {'steps': [], 'returns': []})

    for _, row in history.iterrows():
        game = row.get('game')
        cycle = row.get('cycle')
        ret = row.get('return')

        if pd.isna(game) or pd.isna(cycle) or pd.isna(ret):
            continue

        cycle = int(cycle)
        step_key = f'train/{game}/step'
        step = row.get(step_key, 0)
        if pd.isna(step):
            step = 0

        data[(game, cycle)]['steps'].append(step)
        data[(game, cycle)]['returns'].append(ret)

    return data


def plot_curves(method_data):
    """Create learning curves plot with different line styles for cycles."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for game_idx, game in enumerate(GAMES):
        ax = axes[game_idx]
        ax.set_title(game.replace('-MinAtar', ''), fontsize=12, fontweight='bold')
        ax.set_xlabel('Steps (M)')
        ax.set_ylabel('Return')

        for method in METHOD_STYLES.keys():
            if method not in method_data:
                continue

            runs_data = method_data[method]
            base_color, label = METHOD_STYLES[method]

            for cycle in range(NUM_CYCLES):
                linestyle = CYCLE_LINESTYLES[cycle]

                all_steps = []
                all_returns = []

                for run_data in runs_data:
                    if (game, cycle) in run_data:
                        steps = run_data[(game, cycle)]['steps']
                        returns = run_data[(game, cycle)]['returns']
                        if len(steps) > 0:
                            all_steps.append(steps)
                            all_returns.append(returns)

                if not all_returns:
                    continue

                x_common = np.linspace(0, STEPS_PER_GAME, 50)
                y_interp = []

                for steps, returns in zip(all_steps, all_returns):
                    if len(steps) > 1:
                        sorted_idx = np.argsort(steps)
                        steps = np.array(steps)[sorted_idx]
                        returns = np.array(returns)[sorted_idx]
                        y_interp.append(np.interp(x_common, steps, returns))

                if not y_interp:
                    continue

                y_mean = np.mean(y_interp, axis=0)
                y_std = np.std(y_interp, axis=0) if len(y_interp) > 1 else np.zeros_like(y_mean)

                x_plot = x_common / 1e6
                ax.plot(x_plot, y_mean, color=base_color, linestyle=linestyle,
                       linewidth=1.5, alpha=0.9)
                if len(y_interp) > 1:
                    ax.fill_between(x_plot, y_mean - y_std, y_mean + y_std,
                                   color=base_color, alpha=0.1)

        ax.grid(True, alpha=0.3)

    # Legend subplot - show all 10 cycle styles
    ax_legend = axes[4]
    ax_legend.axis('off')

    handles = []
    labels = []

    # All 10 cycle line styles
    for cycle in range(NUM_CYCLES):
        linestyle = CYCLE_LINESTYLES[cycle]
        handles.append(Line2D([0], [0], color='gray', linestyle=linestyle, linewidth=2))
        labels.append(f'Cycle {cycle}')

    ax_legend.legend(handles, labels, loc='center left', fontsize=9, frameon=True, ncol=2)
    ax_legend.set_title('Cycles', fontsize=11, fontweight='bold')

    # Method colors in separate legend area
    axes[5].axis('off')
    method_handles = []
    method_labels = []
    for method, (color, label) in METHOD_STYLES.items():
        method_handles.append(Line2D([0], [0], color=color, linestyle='-', linewidth=3))
        method_labels.append(label)
    axes[5].legend(method_handles, method_labels, loc='center', fontsize=11, frameon=True)
    axes[5].set_title('Methods', fontsize=11, fontweight='bold')

    plt.suptitle('Per-Game Learning Curves (Channel Permutation)\n10 Cycles with different line styles',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f'{OUTPUT_DIR}/learning_curves_wandb.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_curves_grouped(method_data):
    """Alternative: group cycles into early/mid/late with line styles."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Group cycles
    CYCLE_GROUPS = {
        'early (0-2)': (range(0, 3), '--'),
        'mid (3-6)': (range(3, 7), '-'),
        'late (7-9)': (range(7, 10), ':'),
    }

    for game_idx, game in enumerate(GAMES):
        ax = axes[game_idx]
        ax.set_title(game.replace('-MinAtar', ''), fontsize=12, fontweight='bold')
        ax.set_xlabel('Steps (M)')
        ax.set_ylabel('Return')

        for method in METHOD_STYLES.keys():
            if method not in method_data:
                continue

            runs_data = method_data[method]
            base_color, label = METHOD_STYLES[method]

            for group_name, (cycles, linestyle) in CYCLE_GROUPS.items():
                all_y_interp = []

                for cycle in cycles:
                    for run_data in runs_data:
                        if (game, cycle) in run_data:
                            steps = run_data[(game, cycle)]['steps']
                            returns = run_data[(game, cycle)]['returns']
                            if len(steps) > 1:
                                sorted_idx = np.argsort(steps)
                                steps = np.array(steps)[sorted_idx]
                                returns = np.array(returns)[sorted_idx]
                                x_common = np.linspace(0, STEPS_PER_GAME, 50)
                                all_y_interp.append(np.interp(x_common, steps, returns))

                if not all_y_interp:
                    continue

                y_mean = np.mean(all_y_interp, axis=0)
                y_std = np.std(all_y_interp, axis=0)
                x_plot = x_common / 1e6

                ax.plot(x_plot, y_mean, color=base_color, linestyle=linestyle,
                       linewidth=1.5, alpha=0.9)
                ax.fill_between(x_plot, y_mean - y_std, y_mean + y_std,
                               color=base_color, alpha=0.1)

        ax.grid(True, alpha=0.3)

    # Legend
    ax_legend = axes[4]
    ax_legend.axis('off')

    handles = []
    labels = []

    for group_name, (_, linestyle) in CYCLE_GROUPS.items():
        handles.append(Line2D([0], [0], color='gray', linestyle=linestyle, linewidth=2))
        labels.append(group_name)

    handles.append(Line2D([0], [0], color='white'))
    labels.append('')

    for method, (color, label) in METHOD_STYLES.items():
        handles.append(Line2D([0], [0], color=color, linewidth=2))
        labels.append(label)

    ax_legend.legend(handles, labels, loc='center', fontsize=11, frameon=True)
    ax_legend.set_title('Legend', fontsize=12, fontweight='bold')

    axes[5].axis('off')

    plt.suptitle('Per-Game Learning Curves (Channel Permutation)\nGrouped: Early (0-2) / Mid (3-6) / Late (7-9)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = f'{OUTPUT_DIR}/learning_curves_grouped.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    api = wandb.Api()

    # Try to find the project
    try:
        runs = api.runs(PROJECT)
    except Exception as e:
        print(f"Could not find project '{PROJECT}': {e}")
        print("Trying with user prefix...")
        try:
            runs = api.runs(f"2robert-mueller-none/{PROJECT}")
        except Exception as e2:
            print(f"Also failed: {e2}")
            return

    method_data = defaultdict(list)

    print("Fetching data from wandb...")
    for run in runs:
        name = run.name
        # Parse method name (format: mlp_baseline_seed0, mlp_adamo_seed1, etc.)
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].startswith('seed'):
            method = parts[0]
        else:
            method = name

        if method not in METHOD_STYLES:
            print(f"  Skipping unknown method: {method}")
            continue

        print(f"  Processing: {name}")
        data = get_run_data(run)
        method_data[method].append(data)

    print(f"\nMethods found: {list(method_data.keys())}")

    if not method_data:
        print("No data found!")
        return

    plot_curves(method_data)
    plot_curves_grouped(method_data)

    print("\nDone!")


if __name__ == '__main__':
    main()
