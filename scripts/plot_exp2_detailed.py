#!/usr/bin/env python3
"""
Detailed plots for Experiment 2: Channel Permutation (256x4, 10 cycles, 4 games).

Lyle-style learning curves adapted for 10 cycles.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.style.use('seaborn-v0_8-whitegrid')

RESULTS_DIR = Path('/Users/robertmueller/Desktop/agents/rejax/results/channel_permutation_4games_256x4_10cycles')
OUTPUT_DIR = RESULTS_DIR / 'plots'

GAMES = ['Breakout-MinAtar', 'Asterix-MinAtar', 'SpaceInvaders-MinAtar', 'Freeway-MinAtar']
GAME_SHORT = ['Breakout', 'Asterix', 'SpaceInvaders', 'Freeway']
NUM_CYCLES = 10
NUM_GAMES = 4
TOTAL_CHECKPOINTS = NUM_CYCLES * NUM_GAMES  # 40

COLORS = {
    'mlp_baseline': '#1f77b4',
    'mlp_adamo': '#ff7f0e',
}
LABELS = {
    'mlp_baseline': 'Baseline (ReLU)',
    'mlp_adamo': 'AdaMO (GroupSort)',
}


def load_results():
    """Load most recent results JSON."""
    json_files = sorted(RESULTS_DIR.glob('continual_results_*.json'))
    if not json_files:
        raise FileNotFoundError("No results files found")
    latest = json_files[-1]
    print(f"Loading: {latest}")
    with open(latest) as f:
        return json.load(f)


def extract_learning_curves(data):
    """
    Extract eval_all_games data to build learning curves.

    Returns: {config: {eval_game: {'means': [...], 'stds': [...]}}}
    where each list has 40 entries (4 games × 10 cycles)
    """
    curves = {}

    for config_name, seed_results in data.items():
        curves[config_name] = {game: {'values': [[] for _ in range(TOTAL_CHECKPOINTS)]} for game in GAMES}

        for seed_data in seed_results:
            for i, game_result in enumerate(seed_data['per_game_results']):
                if i >= TOTAL_CHECKPOINTS:
                    break
                eval_all = game_result.get('eval_all_games', {})
                for eval_game in GAMES:
                    if eval_game in eval_all:
                        curves[config_name][eval_game]['values'][i].append(eval_all[eval_game])

        # Compute mean and std
        for eval_game in GAMES:
            values = curves[config_name][eval_game]['values']
            curves[config_name][eval_game]['means'] = [np.mean(v) if v else 0 for v in values]
            curves[config_name][eval_game]['stds'] = [np.std(v) if len(v) > 1 else 0 for v in values]

    return curves


def extract_final_returns(data):
    """Extract final returns per game per cycle."""
    returns = {}

    for config_name, seed_results in data.items():
        returns[config_name] = {game: {'by_cycle': defaultdict(list)} for game in GAMES}

        for seed_data in seed_results:
            for result in seed_data['per_game_results']:
                game = result['game']
                cycle = result['cycle']
                final_return = result['final_return']
                returns[config_name][game]['by_cycle'][cycle].append(final_return)

        # Compute stats
        for game in GAMES:
            by_cycle = returns[config_name][game]['by_cycle']
            returns[config_name][game]['means'] = [np.mean(by_cycle[c]) for c in range(NUM_CYCLES)]
            returns[config_name][game]['stds'] = [np.std(by_cycle[c]) if len(by_cycle[c]) > 1 else 0 for c in range(NUM_CYCLES)]

    return returns


def plot_lyle_curves(curves):
    """Main Lyle-style plot with 10 cycles."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    x = np.arange(TOTAL_CHECKPOINTS)

    for idx, (game, short_name) in enumerate(zip(GAMES, GAME_SHORT)):
        ax = axes[idx]

        for config in curves.keys():
            label = LABELS.get(config, config)
            color = COLORS.get(config, '#333333')
            means = np.array(curves[config][game]['means'])
            stds = np.array(curves[config][game]['stds'])

            ax.plot(x, means, '-', label=label, color=color, linewidth=1.5, alpha=0.9)
            ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.15)

        # Mark when this game was trained (every 4 checkpoints + game_idx)
        game_idx = GAMES.index(game)
        for cycle in range(NUM_CYCLES):
            train_x = cycle * NUM_GAMES + game_idx
            ax.axvline(x=train_x, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

        ax.set_title(f'{short_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Phase')
        ax.set_ylabel('Eval Return')

        # Add cycle boundaries
        for c in range(1, NUM_CYCLES):
            ax.axvline(x=c * NUM_GAMES - 0.5, color='black', linestyle='-', linewidth=1, alpha=0.3)

        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # X ticks at cycle boundaries
        ax.set_xticks([c * NUM_GAMES for c in range(NUM_CYCLES)])
        ax.set_xticklabels([f'C{c}' for c in range(NUM_CYCLES)])

    plt.suptitle('Eval Performance Over Training (Channel Permutation, 10 Cycles)\nVertical lines = cycle boundaries',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lyle_curves_10cycles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'lyle_curves_10cycles.png'}")


def plot_final_returns_by_cycle(returns):
    """Plot final return per game across cycles."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    x = np.arange(NUM_CYCLES)

    for idx, (game, short_name) in enumerate(zip(GAMES, GAME_SHORT)):
        ax = axes[idx]

        for config in returns.keys():
            label = LABELS.get(config, config)
            color = COLORS.get(config, '#333333')
            means = np.array(returns[config][game]['means'])
            stds = np.array(returns[config][game]['stds'])

            ax.plot(x, means, 'o-', label=label, color=color, linewidth=2, markersize=6)
            ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.2)

        ax.set_title(f'{short_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Final Return')
        ax.set_xticks(x)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Summary: average across games
    ax_avg = axes[4]
    for config in returns.keys():
        label = LABELS.get(config, config)
        color = COLORS.get(config, '#333333')

        avg_means = np.zeros(NUM_CYCLES)
        for game in GAMES:
            avg_means += np.array(returns[config][game]['means'])
        avg_means /= NUM_GAMES

        ax_avg.plot(x, avg_means, 'o-', label=label, color=color, linewidth=2, markersize=6)

    ax_avg.set_title('Average Across Games', fontsize=14, fontweight='bold')
    ax_avg.set_xlabel('Cycle')
    ax_avg.set_ylabel('Final Return')
    ax_avg.set_xticks(x)
    ax_avg.legend(loc='best', fontsize=10)
    ax_avg.grid(True, alpha=0.3)

    # Hide last subplot
    axes[5].axis('off')

    plt.suptitle('Final Return Per Cycle (Channel Permutation, 10 Cycles)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'final_returns_by_cycle.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'final_returns_by_cycle.png'}")


def plot_performance_delta(returns):
    """Plot performance change: cycle N vs cycle 0."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(NUM_GAMES)
    width = 0.35

    for i, config in enumerate(returns.keys()):
        label = LABELS.get(config, config)
        color = COLORS.get(config, '#333333')

        # Compute: (cycle 9 return) / (cycle 0 return) - 1
        deltas = []
        for game in GAMES:
            cycle0 = returns[config][game]['means'][0]
            cycle9 = returns[config][game]['means'][-1]
            if cycle0 > 0:
                delta = (cycle9 - cycle0) / cycle0 * 100
            else:
                delta = 0
            deltas.append(delta)

        bars = ax.bar(x + i * width - width/2, deltas, width, label=label, color=color, alpha=0.8)

        # Add value labels
        for bar, delta in zip(bars, deltas):
            height = bar.get_height()
            ax.annotate(f'{delta:+.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -12),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=9)

    ax.set_ylabel('Performance Change (%)', fontsize=12)
    ax.set_xlabel('Game', fontsize=12)
    ax.set_title('Performance Change: Cycle 9 vs Cycle 0\n(Positive = improved, Negative = degraded)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(GAME_SHORT)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'performance_delta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'performance_delta.png'}")


def plot_heatmap(returns):
    """Heatmap: games x cycles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, config in zip(axes, returns.keys()):
        label = LABELS.get(config, config)

        matrix = np.zeros((NUM_GAMES, NUM_CYCLES))
        for i, game in enumerate(GAMES):
            matrix[i, :] = returns[config][game]['means']

        im = ax.imshow(matrix, aspect='auto', cmap='viridis')
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Game')
        ax.set_xticks(range(NUM_CYCLES))
        ax.set_yticks(range(NUM_GAMES))
        ax.set_yticklabels(GAME_SHORT)

        plt.colorbar(im, ax=ax, label='Final Return')

    plt.suptitle('Final Return Heatmap (Games × Cycles)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heatmap_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'heatmap_detailed.png'}")


def plot_plasticity_summary(returns):
    """Summary plot showing plasticity loss over cycles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Normalized performance (cycle 0 = 100%)
    ax1 = axes[0]
    x = np.arange(NUM_CYCLES)

    for config in returns.keys():
        label = LABELS.get(config, config)
        color = COLORS.get(config, '#333333')

        # Normalize each game to cycle 0, then average
        normalized = []
        for cycle in range(NUM_CYCLES):
            norm_values = []
            for game in GAMES:
                cycle0 = returns[config][game]['means'][0]
                cycleN = returns[config][game]['means'][cycle]
                if cycle0 > 0:
                    norm_values.append(cycleN / cycle0 * 100)
            normalized.append(np.mean(norm_values))

        ax1.plot(x, normalized, 'o-', label=label, color=color, linewidth=2, markersize=6)

    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Cycle', fontsize=12)
    ax1.set_ylabel('Normalized Performance (%)', fontsize=12)
    ax1.set_title('Normalized Performance\n(Cycle 0 = 100%)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right: Cumulative performance (area under curve)
    ax2 = axes[1]

    bar_data = []
    for config in returns.keys():
        label = LABELS.get(config, config)

        # Sum of returns across all cycles and games
        total = 0
        for game in GAMES:
            total += sum(returns[config][game]['means'])
        bar_data.append((label, total))

    labels = [b[0] for b in bar_data]
    values = [b[1] for b in bar_data]
    colors_bar = [COLORS.get(config, '#333333') for config in returns.keys()]

    bars = ax2.bar(labels, values, color=colors_bar, alpha=0.8)
    ax2.set_ylabel('Cumulative Return', fontsize=12)
    ax2.set_title('Total Performance\n(Sum across all cycles and games)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, values):
        ax2.annotate(f'{val:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plasticity_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'plasticity_summary.png'}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Experiment 2: Channel Permutation (256x4, 10 Cycles)")
    print("=" * 60)

    data = load_results()
    curves = extract_learning_curves(data)
    returns = extract_final_returns(data)

    plot_lyle_curves(curves)
    plot_final_returns_by_cycle(returns)
    plot_performance_delta(returns)
    plot_heatmap(returns)
    plot_plasticity_summary(returns)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
