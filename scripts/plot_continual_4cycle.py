#!/usr/bin/env python3
"""
Plot learning curves for continual MinAtar experiment (4 cycles).
Similar style to lyle_training_curves.png: solid/dashed for cycles, colors for methods.

Generates 3 plots:
- learning_curves_4cycle.png (all methods)
- learning_curves_4cycle_cnn.png (CNN methods only)
- learning_curves_4cycle_mlp.png (MLP methods only)
"""
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Style settings
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Project and run config
PROJECT = "2robert-mueller-none/continual_minatar_ppo_long_with_metrics_constant_lr"
GAMES = ['Breakout-MinAtar', 'Asterix-MinAtar', 'SpaceInvaders-MinAtar', 'Freeway-MinAtar', 'Seaquest-MinAtar']
STEPS_PER_GAME = 20_000_000
OUTPUT_DIR = '/Users/robertmueller/Desktop/agents/rejax/results/continual_minatar_4cycle/plots'

# Method styling: (color, label, is_cnn)
METHOD_STYLES = {
    'mlp_baseline': ('tab:blue', 'MLP Baseline', False),
    'pgx_baseline': ('tab:red', 'CNN Baseline', True),
    'mlp_adamo': ('tab:orange', 'MLP AdaMO+GS', False),
    'cnn_adamo': ('tab:purple', 'CNN AdaMO', True),
    'mlp_adamo_lyle_continual': ('tab:green', 'MLP AdaMO+LyleLR', False),
    'cnn_adamo_lyle_continual': ('tab:cyan', 'CNN AdaMO+LyleLR', True),  # High contrast vs red
}

# Cycle styling: linestyle
CYCLE_STYLES = {
    0: ('--', 'Cycle 0 (1st)'),
    1: ('-', 'Cycle 1 (2nd)'),
    2: (':', 'Cycle 2 (3rd)'),
    3: ('-.', 'Cycle 3 (4th)'),
}


def get_run_data(run):
    """Extract training curves from a wandb run."""
    history = run.history(samples=10000)
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


def plot_curves(method_data, methods_to_plot, title_suffix, output_name):
    """Create learning curves plot for specified methods."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for game_idx, game in enumerate(GAMES):
        ax = axes[game_idx]
        ax.set_title(game.replace('-MinAtar', ''), fontsize=12, fontweight='bold')
        ax.set_xlabel('Steps (M)')
        ax.set_ylabel('Return')

        for method in methods_to_plot:
            if method not in method_data:
                continue

            runs_data = method_data[method]
            color, label, _ = METHOD_STYLES[method]

            for cycle in range(4):
                linestyle, _ = CYCLE_STYLES[cycle]

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
                ax.plot(x_plot, y_mean, color=color, linestyle=linestyle,
                       linewidth=1.5, alpha=0.9)
                if len(y_interp) > 1:
                    ax.fill_between(x_plot, y_mean - y_std, y_mean + y_std,
                                   color=color, alpha=0.15)

    # Legend
    ax_legend = axes[5]
    ax_legend.axis('off')

    from matplotlib.lines import Line2D
    handles = []
    labels = []

    # Cycle styles
    for cycle, (ls, lbl) in CYCLE_STYLES.items():
        handles.append(Line2D([0], [0], color='gray', linestyle=ls, linewidth=1.5))
        labels.append(lbl)

    # Method colors (only those plotted)
    for method in methods_to_plot:
        if method in METHOD_STYLES:
            color, label, _ = METHOD_STYLES[method]
            handles.append(Line2D([0], [0], color=color, linestyle='-', linewidth=2))
            labels.append(label)

    ax_legend.legend(handles, labels, loc='center', fontsize=11, frameon=True)
    ax_legend.set_title('Legend', fontsize=12, fontweight='bold')

    plt.suptitle(f'Per-Game Learning Curves{title_suffix}\n4 Cycles of Continual Training',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f'{OUTPUT_DIR}/{output_name}'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    api = wandb.Api()
    runs = api.runs(PROJECT)

    method_data = defaultdict(list)

    print("Fetching data from wandb...")
    for run in runs:
        name = run.name
        method = '_'.join(name.split('_')[:-1])

        if method not in METHOD_STYLES:
            print(f"  Skipping unknown method: {method}")
            continue

        print(f"  Processing: {name}")
        data = get_run_data(run)
        method_data[method].append(data)

    print(f"\nMethods found: {list(method_data.keys())}")

    # All methods
    all_methods = list(METHOD_STYLES.keys())
    plot_curves(method_data, all_methods, '', 'learning_curves_4cycle.png')

    # CNN only
    cnn_methods = [m for m, (_, _, is_cnn) in METHOD_STYLES.items() if is_cnn]
    plot_curves(method_data, cnn_methods, ' (CNN)', 'learning_curves_4cycle_cnn.png')

    # MLP only
    mlp_methods = [m for m, (_, _, is_cnn) in METHOD_STYLES.items() if not is_cnn]
    plot_curves(method_data, mlp_methods, ' (MLP)', 'learning_curves_4cycle_mlp.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
