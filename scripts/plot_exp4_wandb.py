#!/usr/bin/env python3
"""
Plot learning curves for Experiment 4: L2-Init Comparison (4 cycles, 5 games).
Fetches data from wandb. Similar style to plot_exp2_wandb.py.
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
PROJECT = "continual_minatar_l2_init"
GAMES = ['Breakout-MinAtar', 'Asterix-MinAtar', 'SpaceInvaders-MinAtar', 'Freeway-MinAtar', 'Seaquest-MinAtar']
STEPS_PER_GAME = 20_000_000
NUM_CYCLES = 4
OUTPUT_DIR = '/Users/robertmueller/Desktop/agents/rejax/results/l2_init_5games_256x4_4cycles/plots'

# Method styling
METHOD_STYLES = {
    'mlp_baseline': ('tab:blue', 'Baseline'),
    'mlp_l2_init': ('tab:green', 'L2-Init 0.001'),
    'mlp_l2_init_0.01': ('tab:red', 'L2-Init 0.01'),
}

# Line styles for 4 cycles
CYCLE_LINESTYLES = {
    0: '-',      # solid
    1: '--',     # dashed
    2: ':',      # dotted
    3: '-.',     # dashdot
}

CYCLE_LABELS = {
    0: 'Cycle 0 (1st)',
    1: 'Cycle 1 (2nd)',
    2: 'Cycle 2 (3rd)',
    3: 'Cycle 3 (4th)',
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

    # Legend subplot
    ax_legend = axes[5]
    ax_legend.axis('off')

    handles = []
    labels = []

    # Cycle line styles
    for cycle in range(NUM_CYCLES):
        linestyle = CYCLE_LINESTYLES[cycle]
        handles.append(Line2D([0], [0], color='gray', linestyle=linestyle, linewidth=2))
        labels.append(CYCLE_LABELS[cycle])

    handles.append(Line2D([0], [0], color='white'))  # spacer
    labels.append('')

    # Method colors
    for method, (color, label) in METHOD_STYLES.items():
        handles.append(Line2D([0], [0], color=color, linestyle='-', linewidth=3))
        labels.append(label)

    ax_legend.legend(handles, labels, loc='center', fontsize=11, frameon=True)
    ax_legend.set_title('Legend', fontsize=12, fontweight='bold')

    plt.suptitle('Per-Game Learning Curves (L2-Init Comparison)\n4 Cycles, line style = cycle',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f'{OUTPUT_DIR}/learning_curves_wandb.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    api = wandb.Api()

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
        # Parse method name (format: mlp_baseline_seed0, mlp_l2_init_seed1, etc.)
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

    print("\nDone!")


if __name__ == '__main__':
    main()
