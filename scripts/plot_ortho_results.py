"""
Plot ortho scaling benchmark results.
Generates learning curves and depth study plots.
"""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

RESULTS_DIR = Path("ortho_scaling_results")
PLOTS_DIR = Path("plots_ortho")
PLOTS_DIR.mkdir(exist_ok=True)


def load_all_results():
    """Load all JSON result files."""
    all_results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        # Skip small files (smoke tests)
        if f.stat().st_size < 10000:
            continue
        with open(f) as fp:
            data = json.load(fp)
            all_results.extend(data)
    return all_results


def categorize_config(r):
    """Categorize a result into baseline/ortho_opt/ortho_loss."""
    mode = r.get('ortho_mode')
    if mode is None or mode == 'none':
        return 'baseline'
    elif mode == 'optimizer':
        coeff = r.get('ortho_coeff', 0)
        return f'ortho_opt_{coeff}'
    elif mode == 'loss':
        lam = r.get('ortho_lambda', 0)
        return f'ortho_loss_{lam}'
    return 'unknown'


def get_config_label(config):
    """Get human-readable label for config."""
    if config == 'baseline':
        return 'Baseline (tanh)'
    elif config.startswith('ortho_opt_'):
        coeff = config.replace('ortho_opt_', '')
        return f'Ortho Opt (coeff={coeff})'
    elif config.startswith('ortho_loss_'):
        lam = config.replace('ortho_loss_', '')
        return f'Ortho Loss (λ={lam})'
    return config


def plot_learning_curves_per_env(results):
    """Plot learning curves per environment, grouped by config type."""
    envs = sorted(set(r['env'] for r in results))

    for env in envs:
        env_results = [r for r in results if r['env'] == env]

        # Group by config
        configs = {}
        for r in env_results:
            cfg = categorize_config(r)
            if cfg not in configs:
                configs[cfg] = []
            configs[cfg].append(r)

        # Plot each config
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

        for (cfg, cfg_results), color in zip(sorted(configs.items()), colors):
            # Aggregate across depths - use depth=8 as representative
            depth_results = [r for r in cfg_results if r['depth'] == 8]
            if not depth_results:
                depth_results = cfg_results[:1]

            for r in depth_results:
                psd = r.get('per_seed_data')
                if not psd:
                    continue

                steps = np.array(psd['eval_steps'])
                returns_mean = np.array(psd['returns_mean'])
                returns_std = np.array(psd['returns_std'])

                label = f"{get_config_label(cfg)} (d={r['depth']})"
                ax.plot(steps / 1e6, returns_mean, label=label, color=color, linewidth=2)
                ax.fill_between(steps / 1e6, returns_mean - returns_std,
                               returns_mean + returns_std, alpha=0.2, color=color)

        ax.set_xlabel('Timesteps (M)')
        ax.set_ylabel('Return')
        ax.set_title(f'Learning Curves - {env}')
        ax.legend(loc='best', fontsize=9)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'learning_curves_{env.replace("-", "_")}.png', dpi=150)
        plt.close()
        print(f"Saved: learning_curves_{env.replace('-', '_')}.png")


def plot_depth_study(results):
    """Plot final return vs depth for each setting."""
    envs = sorted(set(r['env'] for r in results))

    for env in envs:
        env_results = [r for r in results if r['env'] == env]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by major config type (baseline, ortho_opt, ortho_loss)
        config_groups = {
            'baseline': [],
            'ortho_opt': [],
            'ortho_loss': []
        }

        for r in env_results:
            cfg = categorize_config(r)
            if cfg == 'baseline':
                config_groups['baseline'].append(r)
            elif cfg.startswith('ortho_opt'):
                config_groups['ortho_opt'].append(r)
            elif cfg.startswith('ortho_loss'):
                config_groups['ortho_loss'].append(r)

        colors = {'baseline': 'C0', 'ortho_opt': 'C1', 'ortho_loss': 'C2'}
        labels = {'baseline': 'Baseline (tanh)', 'ortho_opt': 'Ortho Optimizer (mean)', 'ortho_loss': 'Ortho Loss (mean)'}

        for cfg_type, cfg_results in config_groups.items():
            if not cfg_results:
                continue

            # Group by depth, mean over hyperparams
            depth_returns = {}
            for r in cfg_results:
                d = r['depth']
                if d not in depth_returns:
                    depth_returns[d] = []
                depth_returns[d].append(r.get('final_return_mean', 0))

            depths = sorted(depth_returns.keys())
            means = [np.mean(depth_returns[d]) for d in depths]
            stds = [np.std(depth_returns[d]) for d in depths]

            ax.errorbar(depths, means, yerr=stds, label=labels[cfg_type],
                       color=colors[cfg_type], linewidth=2, marker='o', capsize=5)

        ax.set_xlabel('Network Depth')
        ax.set_ylabel('Final Return')
        ax.set_title(f'Depth Study - {env}')
        ax.legend()
        ax.set_xticks([2, 4, 8, 16, 32])

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'depth_study_{env.replace("-", "_")}.png', dpi=150)
        plt.close()
        print(f"Saved: depth_study_{env.replace('-', '_')}.png")


def plot_hyperparam_comparison(results):
    """Plot return vs hyperparameter for ortho modes."""
    envs = sorted(set(r['env'] for r in results))

    for env in envs:
        env_results = [r for r in results if r['env'] == env]

        # Ortho optimizer coefficients
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Ortho optimizer
        ax = axes[0]
        opt_results = [r for r in env_results if r.get('ortho_mode') == 'optimizer']
        if opt_results:
            coeffs = sorted(set(r.get('ortho_coeff', 0) for r in opt_results))

            for depth in [2, 4, 8, 16, 32]:
                depth_data = [r for r in opt_results if r['depth'] == depth]
                if depth_data:
                    x = [r.get('ortho_coeff', 0) for r in depth_data]
                    y = [r.get('final_return_mean', 0) for r in depth_data]
                    # Sort by x
                    xy = sorted(zip(x, y))
                    x, y = zip(*xy) if xy else ([], [])
                    ax.plot(x, y, 'o-', label=f'd={depth}', linewidth=2, markersize=8)

            ax.set_xscale('log')
            ax.set_xlabel('Ortho Coefficient')
            ax.set_ylabel('Final Return')
            ax.set_title(f'Ortho Optimizer - {env}')
            ax.legend()

        # Right: Ortho loss
        ax = axes[1]
        loss_results = [r for r in env_results if r.get('ortho_mode') == 'loss']
        if loss_results:
            lambdas = sorted(set(r.get('ortho_lambda', 0) for r in loss_results))

            for depth in [2, 4, 8, 16, 32]:
                depth_data = [r for r in loss_results if r['depth'] == depth]
                if depth_data:
                    x = [r.get('ortho_lambda', 0) for r in depth_data]
                    y = [r.get('final_return_mean', 0) for r in depth_data]
                    xy = sorted(zip(x, y))
                    x, y = zip(*xy) if xy else ([], [])
                    ax.plot(x, y, 'o-', label=f'd={depth}', linewidth=2, markersize=8)

            ax.set_xscale('log')
            ax.set_xlabel('Ortho Lambda')
            ax.set_ylabel('Final Return')
            ax.set_title(f'Ortho Loss - {env}')
            ax.legend()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'hyperparam_{env.replace("-", "_")}.png', dpi=150)
        plt.close()
        print(f"Saved: hyperparam_{env.replace('-', '_')}.png")


def plot_throughput_vs_depth(results):
    """Plot throughput vs depth for each setting."""
    envs = sorted(set(r['env'] for r in results))

    for env in envs:
        env_results = [r for r in results if r['env'] == env]

        fig, ax = plt.subplots(figsize=(10, 6))

        config_groups = {
            'baseline': [],
            'ortho_opt': [],
            'ortho_loss': []
        }

        for r in env_results:
            cfg = categorize_config(r)
            if cfg == 'baseline':
                config_groups['baseline'].append(r)
            elif cfg.startswith('ortho_opt'):
                config_groups['ortho_opt'].append(r)
            elif cfg.startswith('ortho_loss'):
                config_groups['ortho_loss'].append(r)

        colors = {'baseline': 'C0', 'ortho_opt': 'C1', 'ortho_loss': 'C2'}
        labels = {'baseline': 'Baseline', 'ortho_opt': 'Ortho Optimizer', 'ortho_loss': 'Ortho Loss'}

        for cfg_type, cfg_results in config_groups.items():
            if not cfg_results:
                continue

            depth_throughput = {}
            for r in cfg_results:
                d = r['depth']
                if d not in depth_throughput:
                    depth_throughput[d] = []
                depth_throughput[d].append(r.get('steps_per_second', 0))

            depths = sorted(depth_throughput.keys())
            means = [np.mean(depth_throughput[d]) / 1e6 for d in depths]

            ax.plot(depths, means, 'o-', label=labels[cfg_type],
                   color=colors[cfg_type], linewidth=2, markersize=8)

        ax.set_xlabel('Network Depth')
        ax.set_ylabel('Throughput (M steps/sec)')
        ax.set_title(f'Throughput vs Depth - {env}')
        ax.legend()
        ax.set_xticks([2, 4, 8, 16, 32])

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'throughput_{env.replace("-", "_")}.png', dpi=150)
        plt.close()
        print(f"Saved: throughput_{env.replace('-', '_')}.png")


def print_summary_table(results):
    """Print summary table of best configs."""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    envs = sorted(set(r['env'] for r in results))

    for env in envs:
        print(f"\n{env}:")
        print("-" * 60)
        env_results = [r for r in results if r['env'] == env]

        # Find best per config type
        best = {}
        for r in env_results:
            cfg = categorize_config(r)
            ret = r.get('final_return_mean', 0)
            if cfg not in best or ret > best[cfg][0]:
                best[cfg] = (ret, r)

        for cfg in sorted(best.keys()):
            ret, r = best[cfg]
            print(f"  {get_config_label(cfg):30s}: {ret:.2f} (d={r['depth']})")


if __name__ == "__main__":
    print("Loading results...")
    results = load_all_results()
    print(f"Loaded {len(results)} benchmark entries")

    print("\nGenerating plots...")
    plot_learning_curves_per_env(results)
    plot_depth_study(results)
    plot_hyperparam_comparison(results)
    plot_throughput_vs_depth(results)

    print_summary_table(results)

    print(f"\nAll plots saved to: {PLOTS_DIR}/")
