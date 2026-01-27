#!/usr/bin/env python3
"""
Run Octax PPO directly using their implementation.
No gymnax wrapper, no rejax PPO - just their code.
"""
import sys
sys.path.insert(0, '/tmp/octax')

import jax
import jax.numpy as jnp
import time

# Install octax if needed
import subprocess
subprocess.run(['pip', 'install', '-q', 'hydra-core'], check=True)

try:
    from octax.agents.ppo import PPOOctax
    from octax.environments import create_environment
    from octax.wrappers import OctaxGymnaxWrapper
except ImportError:
    print("Installing octax...")
    subprocess.run(['rm', '-rf', '/tmp/octax'], check=False)
    subprocess.run(['git', 'clone', 'https://github.com/riiswa/octax.git', '/tmp/octax'], check=True)
    subprocess.run(['pip', 'install', '-q', '/tmp/octax'], check=True)
    from octax.agents.ppo import PPOOctax
    from octax.environments import create_environment
    from octax.wrappers import OctaxGymnaxWrapper


def run_octax_ppo(game: str, num_seeds: int = 2, total_timesteps: int = 5_000_000):
    """Run octax's own PPO implementation."""
    print(f"\n{'='*60}")
    print(f"Running Octax PPO on {game}")
    print(f"Seeds: {num_seeds}, Steps: {total_timesteps:,}")
    print(f"{'='*60}")

    # Create env exactly like they do
    octax_env, metadata = create_environment(game)
    env = OctaxGymnaxWrapper(octax_env)
    env_params = env.default_params

    # Their default config (from conf/config.yaml)
    cfg = {
        'num_envs': 512,
        'num_steps': 32,
        'num_epochs': 4,
        'num_minibatches': 32,
        'learning_rate': 5e-4,
        'total_timesteps': total_timesteps,
        'eval_freq': 131072,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_eps': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    }

    # Create agent exactly like they do
    agent = PPOOctax.create_agent(cfg, env, env_params)
    algo = PPOOctax(env=env, env_params=env_params, agent=agent, eval_callback=lambda *args: None, **cfg)

    # Train with vmap over seeds
    print("Compiling...")
    vmap_train = jax.jit(jax.vmap(algo.train))
    keys = jax.random.split(jax.random.PRNGKey(0), num_seeds)

    start = time.time()
    train_states, metrics = vmap_train(keys)
    jax.block_until_ready(train_states)
    elapsed = time.time() - start

    print(f"Done in {elapsed:.1f}s")
    print(f"Steps/sec: {total_timesteps * num_seeds / elapsed:,.0f}")

    # Eval
    from rejax.evaluate import evaluate

    all_returns = []
    for seed_idx in range(num_seeds):
        # PPOOctaxState structure: agent_ts.params (not just params)
        params = jax.tree.map(lambda x: x[seed_idx], train_states.agent_ts.params)
        lengths, returns = evaluate(
            algo.make_act(params), env, env_params,
            128, jax.random.PRNGKey(seed_idx)
        )
        mean_return = float(returns.mean())
        all_returns.append(mean_return)
        print(f"  Seed {seed_idx}: return = {mean_return:.1f}")

    print(f"\nMean return: {sum(all_returns)/len(all_returns):.1f}")
    return all_returns


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='brix')
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--steps', type=int, default=5_000_000)
    args = parser.parse_args()

    run_octax_ppo(args.game, args.seeds, args.steps)
