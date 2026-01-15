#!/usr/bin/env python3
"""
Plot results from plasticity study with mean +/- std across seeds.

Usage:
    python examples/plot_results.py --log_dir runs --metric eval/return
    python examples/plot_results.py --log_dir runs --metric critic/loss --smooth 10
"""

import argparse
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_data(log_dir, metric):
    """Load metric data from all seed directories."""
    data_by_config = defaultdict(list)

    # Walk through log directory structure: log_dir/config_name/timestamp/seed_X/
    for config_name in os.listdir(log_dir):
        config_path = os.path.join(log_dir, config_name)
        if not os.path.isdir(config_path):
            continue

        for timestamp in os.listdir(config_path):
            timestamp_path = os.path.join(config_path, timestamp)
            if not os.path.isdir(timestamp_path):
                continue

            # Collect all seeds for this config
            seeds_data = []
            for seed_dir in sorted(os.listdir(timestamp_path)):
                if not seed_dir.startswith("seed_"):
                    continue

                seed_path = os.path.join(timestamp_path, seed_dir)
                try:
                    ea = EventAccumulator(seed_path)
                    ea.Reload()

                    if metric in ea.Tags()['scalars']:
                        events = ea.Scalars(metric)
                        steps = [e.step for e in events]
                        values = [e.value for e in events]
                        seeds_data.append((steps, values))
                except Exception as e:
                    print(f"Warning: Could not load {seed_path}: {e}")

            if seeds_data:
                data_by_config[config_name] = seeds_data

    return data_by_config


def interpolate_to_common_steps(seeds_data, num_points=500):
    """Interpolate all seeds to common step values for mean/std computation."""
    if not seeds_data:
        return None, None, None

    # Find common step range
    all_steps = []
    for steps, values in seeds_data:
        all_steps.extend(steps)

    if not all_steps:
        return None, None, None

    min_step = min(all_steps)
    max_step = max(all_steps)

    common_steps = np.linspace(min_step, max_step, num_points)

    # Interpolate each seed
    interpolated = []
    for steps, values in seeds_data:
        if len(steps) < 2:
            continue
        interp_values = np.interp(common_steps, steps, values)
        interpolated.append(interp_values)

    if not interpolated:
        return None, None, None

    interpolated = np.array(interpolated)
    mean = np.mean(interpolated, axis=0)
    std = np.std(interpolated, axis=0)

    return common_steps, mean, std


def smooth(values, window=10):
    """Simple moving average smoothing."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def plot_results(data_by_config, metric, smooth_window=1, output_path=None):
    """Create plot with mean +/- std for each config."""
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data_by_config)))

    for (config_name, seeds_data), color in zip(sorted(data_by_config.items()), colors):
        steps, mean, std = interpolate_to_common_steps(seeds_data)

        if steps is None:
            print(f"Skipping {config_name}: no valid data")
            continue

        # Apply smoothing
        if smooth_window > 1:
            mean = smooth(mean, smooth_window)
            std = smooth(std, smooth_window)
            steps = steps[:len(mean)]

        # Plot mean line
        ax.plot(steps, mean, label=config_name, color=color, linewidth=2)

        # Plot std shading
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} (mean ± std across seeds)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_individual_runs(data_by_config, metric, output_path=None):
    """Create plot showing all individual seed runs."""
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data_by_config)))

    for (config_name, seeds_data), color in zip(sorted(data_by_config.items()), colors):
        for i, (steps, values) in enumerate(seeds_data):
            label = config_name if i == 0 else None
            alpha = 0.7 if len(seeds_data) == 1 else 0.4
            ax.plot(steps, values, label=label, color=color, alpha=alpha, linewidth=1)

    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} (individual runs)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def list_available_metrics(log_dir):
    """List all available metrics in the log directory."""
    metrics = set()

    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                try:
                    ea = EventAccumulator(root)
                    ea.Reload()
                    metrics.update(ea.Tags().get('scalars', []))
                except:
                    pass
                break  # Only check one events file per directory

    return sorted(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot plasticity study results")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--metric", type=str, default="eval/return", help="Metric to plot")
    parser.add_argument("--smooth", type=int, default=1, help="Smoothing window size")
    parser.add_argument("--output", type=str, default=None, help="Output file path (e.g., plot.png)")
    parser.add_argument("--individual", action="store_true", help="Plot individual runs instead of mean/std")
    parser.add_argument("--list_metrics", action="store_true", help="List available metrics and exit")
    args = parser.parse_args()

    if args.list_metrics:
        print("Available metrics:")
        for m in list_available_metrics(args.log_dir):
            print(f"  {m}")
        exit(0)

    print(f"Loading data from {args.log_dir}...")
    data = load_tensorboard_data(args.log_dir, args.metric)

    if not data:
        print(f"No data found for metric '{args.metric}'")
        print("Available metrics:")
        for m in list_available_metrics(args.log_dir):
            print(f"  {m}")
        exit(1)

    print(f"Found {len(data)} configurations:")
    for config, seeds in data.items():
        print(f"  {config}: {len(seeds)} seeds")

    if args.individual:
        plot_individual_runs(data, args.metric, args.output)
    else:
        plot_results(data, args.metric, args.smooth, args.output)
