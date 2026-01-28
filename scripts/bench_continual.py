"""
Continual learning benchmark for MinAtar games following Lyle et al. NaP methodology.

Compares baseline PPO vs Ortho Optimizer on sequential game training without weight resets.

Usage:
    # Smoke test
    python scripts/bench_continual.py --steps-per-game 100000 --num-cycles 1 --num-seeds 1

    # Full experiment
    python scripts/bench_continual.py --steps-per-game 10000000 --num-cycles 2 --num-seeds 3 --use-wandb

    # Test action mapping
    python scripts/bench_continual.py --test-action-mapping

    # Resume from checkpoint
    python scripts/bench_continual.py --resume checkpoints/continual_baseline_game2.ckpt
"""
import argparse
import json
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, struct
from flax.training.train_state import TrainState
from gymnax.environments.minatar.breakout import MinBreakout
from gymnax.environments.minatar.asterix import MinAsterix
from gymnax.environments.minatar.space_invaders import MinSpaceInvaders
from gymnax.environments.minatar.freeway import MinFreeway
from minatar_seaquest_fixed import MinSeaquestFixed

from rejax import PPO
from rejax.regularization import compute_gram_regularization_loss


def compute_ortho_metrics(actor_params, critic_params):
    """Compute Gram deviation metrics for actor and critic networks.

    Returns dict with ortho/actor_loss, ortho/critic_loss, ortho/total_loss
    and per-layer breakdowns.
    """
    # Compute actor ortho loss (exclude output layer)
    _, actor_metrics = compute_gram_regularization_loss(
        actor_params, lambda_coeff=1.0, exclude_output=True, log_diagnostics=False
    )

    # Compute critic ortho loss (exclude output layer)
    _, critic_metrics = compute_gram_regularization_loss(
        critic_params, lambda_coeff=1.0, exclude_output=True, log_diagnostics=False
    )

    # Rename metrics with actor/critic prefix
    metrics = {}
    for k, v in actor_metrics.items():
        metrics[k.replace("ortho/", "ortho/actor_")] = float(v)
    for k, v in critic_metrics.items():
        metrics[k.replace("ortho/", "ortho/critic_")] = float(v)

    # Add combined totals
    metrics["ortho/actor_loss"] = float(actor_metrics.get("ortho/total_loss", 0.0))
    metrics["ortho/critic_loss"] = float(critic_metrics.get("ortho/total_loss", 0.0))
    metrics["ortho/total_loss"] = metrics["ortho/actor_loss"] + metrics["ortho/critic_loss"]

    return metrics


def create_cached_eval_fn(ppo):
    """Create a cached evaluation function that doesn't recompile on each call.

    The key insight: the act function in evaluate() is marked as static_argnames,
    so each new act closure triggers recompilation. Instead, we create ONE
    JIT-compiled function that takes params as a regular pytree argument.

    This reduces eval compilations from ~1000 (40 per game × 25 games) to just 5
    (one per game, reused across cycles).
    """
    env = ppo.env
    env_params = ppo.env_params
    actor = ppo.actor
    max_steps = env_params.max_steps_in_episode

    @jax.jit
    def cached_eval(actor_params, rng, num_episodes=128):
        """Evaluate actor with given params (no recompilation)."""

        def eval_episode(rng):
            """Run single episode, return (length, return)."""
            rng_reset, rng_ep = jax.random.split(rng)
            obs, env_state = env.reset(rng_reset, env_params)

            def step_fn(carry, _):
                obs, env_state, rng, done, ret, length = carry
                rng, rng_act, rng_step = jax.random.split(rng, 3)

                # Use actor directly with params (not via closure)
                action = actor.apply(actor_params, jnp.expand_dims(obs, 0),
                                    rng_act, method="act")
                action = jnp.squeeze(action)

                next_obs, next_state, reward, next_done, _ = env.step(
                    rng_step, env_state, action, env_params)

                # Only update if not already done
                ret = ret + reward * (1 - done)
                length = length + (1 - done).astype(jnp.int32)

                return (next_obs, next_state, rng, done | next_done, ret, length), None

            init_carry = (obs, env_state, rng_ep, False, 0.0, 0)
            (_, _, _, _, final_return, final_length), _ = jax.lax.scan(
                step_fn, init_carry, None, length=max_steps)

            return final_length, final_return

        rngs = jax.random.split(rng, num_episodes)
        lengths, returns = jax.vmap(eval_episode)(rngs)
        return lengths, returns

    return cached_eval


def print_update_diagnostics(config: dict, total_timesteps: int):
    """Print update ratio diagnostics to verify gradient steps are in target range."""
    num_envs = config.get("num_envs", 2048)
    num_steps = config.get("num_steps", 128)
    num_epochs = config.get("num_epochs", 4)
    num_minibatches = config.get("num_minibatches", 4)

    B = num_envs * num_steps
    num_updates = total_timesteps // B
    grad_steps_per_update = num_epochs * num_minibatches
    total_grad_steps = num_updates * grad_steps_per_update

    print(f"  Update diagnostics:")
    print(f"    B (batch_size) = {B:,}")
    print(f"    num_updates = {num_updates:,}")
    print(f"    grad_steps/update = {grad_steps_per_update}")
    print(f"    total_grad_steps = {total_grad_steps:,}")
    print(f"    Target range: 10k-50k (Pgx baseline: ~14.6k)")
    if total_grad_steps < 10_000:
        print(f"    WARNING: total_grad_steps ({total_grad_steps:,}) is below target range!")
    elif total_grad_steps > 50_000:
        print(f"    WARNING: total_grad_steps ({total_grad_steps:,}) is above target range!")


# MinAtar game configurations
# Note: Seaquest uses fixed implementation adapted from pgx-minatar (gymnax version is broken)
MINATAR_GAMES = {
    "Breakout-MinAtar": {"channels": 4, "actions": 3, "env_cls": MinBreakout},
    "Asterix-MinAtar": {"channels": 4, "actions": 5, "env_cls": MinAsterix},
    "SpaceInvaders-MinAtar": {"channels": 6, "actions": 4, "env_cls": MinSpaceInvaders},
    "Freeway-MinAtar": {"channels": 7, "actions": 3, "env_cls": MinFreeway},
    "Seaquest-MinAtar": {"channels": 10, "actions": 6, "env_cls": MinSeaquestFixed},
}

GAME_ORDER = [
    "Breakout-MinAtar",
    "Asterix-MinAtar",
    "SpaceInvaders-MinAtar",
    "Freeway-MinAtar",
    "Seaquest-MinAtar",
]

# Unified observation space: 10 channels (max from Seaquest)
UNIFIED_CHANNELS = 10
# Unified action space: 6 actions (max from Seaquest)
UNIFIED_ACTIONS = 6


class PaddedMinAtarEnv:
    """Wrapper that pads observations and unifies action space for MinAtar games.

    Optionally applies channel permutation before padding for continual learning
    experiments that need to prevent memorization of fixed channel orderings.
    """

    def __init__(self, env, original_channels: int, original_actions: int,
                 channel_perm: Optional[np.ndarray] = None):
        self._env = env
        self.original_channels = original_channels
        self.original_actions = original_actions
        self.channel_perm = channel_perm  # e.g., [2, 0, 3, 1] for 4 channels

    def __getattr__(self, name):
        if name in ["_env", "original_channels", "original_actions", "channel_perm",
                    "reset", "step", "observation_space", "action_space"]:
            return super().__getattribute__(name)
        return getattr(self._env, name)

    @property
    def default_params(self):
        return self._env.default_params

    def observation_space(self, params):
        # Return padded observation space
        return gymnax.environments.spaces.Box(
            low=0.0, high=1.0, shape=(10, 10, UNIFIED_CHANNELS), dtype=jnp.float32
        )

    def action_space(self, params):
        # Return unified action space
        return gymnax.environments.spaces.Discrete(UNIFIED_ACTIONS)

    def reset(self, key, params):
        obs, state = self._env.reset(key, params)
        obs = self._pad_obs(obs)
        return obs, state

    def step(self, key, state, action, params):
        # Map invalid actions to no-op (action 0)
        valid_action = jnp.where(action < self.original_actions, action, 0)
        obs, state, reward, done, info = self._env.step(key, state, valid_action, params)
        obs = self._pad_obs(obs)
        return obs, state, reward, done, info

    def _pad_obs(self, obs):
        """Pad observations from (10, 10, C) to (10, 10, 10).

        If channel_perm is set, permutes ALL 10 channels (including padding)
        to prevent memorization of fixed channel positions. The network must
        learn which channels contain content (have at least one non-zero pixel)
        vs padding (all zeros).
        """
        # First pad to unified channels
        if self.original_channels < UNIFIED_CHANNELS:
            pad_width = UNIFIED_CHANNELS - self.original_channels
            obs = jnp.pad(obs, ((0, 0), (0, 0), (0, pad_width)))
        # Then apply channel permutation to all 10 channels
        if self.channel_perm is not None:
            obs = obs[:, :, self.channel_perm]
        return obs.astype(jnp.float32)


def create_padded_env(game_name: str, channel_perm: Optional[np.ndarray] = None) -> Tuple[PaddedMinAtarEnv, Any]:
    """Create a padded MinAtar environment for the given game.

    Args:
        game_name: Name of the MinAtar game
        channel_perm: Optional channel permutation array. If provided, observation
            channels are reordered according to this permutation before padding.
            E.g., for a 4-channel game, channel_perm=[2, 0, 3, 1] would swap channels.
    """
    game_info = MINATAR_GAMES[game_name]
    env = game_info["env_cls"]()
    padded_env = PaddedMinAtarEnv(
        env,
        original_channels=game_info["channels"],
        original_actions=game_info["actions"],
        channel_perm=channel_perm,
    )
    return padded_env, padded_env.default_params


def create_lyle_lr_schedule(initial_lr: float, final_lr: float, total_steps: int):
    """
    Lyle et al. linear decay schedule: 6.25e-5 → 1e-6.

    Args:
        initial_lr: Starting learning rate (e.g., 6.25e-5)
        final_lr: Final learning rate (e.g., 1e-6)
        total_steps: Total number of gradient steps

    Returns:
        optax schedule function
    """
    return optax.linear_schedule(
        init_value=initial_lr,
        end_value=final_lr,
        transition_steps=total_steps,
    )


def create_lyle_continual_schedule(peak_lr: float, final_lr: float, warmup_steps: int, total_steps: int):
    """
    Lyle et al. cosine decay with warmup schedule for continual learning.

    Per paper B.2: init_value=1e-8, peak after warmup, cosine decay to end_value.
    This schedule should be RESTARTED at every game change.

    Args:
        peak_lr: Peak learning rate after warmup (e.g., 6.25e-4)
        final_lr: Final learning rate (e.g., 1e-6)
        warmup_steps: Number of warmup steps (e.g., 1000)
        total_steps: Total number of gradient steps per game

    Returns:
        optax schedule function
    """
    # For smoke tests with small step counts, adjust warmup and decay proportionally
    if total_steps <= warmup_steps:
        # Split total_steps: 10% warmup, 90% decay (minimum 1 each)
        effective_warmup = max(1, total_steps // 10)
        decay_steps = max(1, total_steps - effective_warmup)
    else:
        effective_warmup = warmup_steps
        decay_steps = total_steps - warmup_steps

    return optax.warmup_cosine_decay_schedule(
        init_value=1e-8,
        peak_value=peak_lr,
        warmup_steps=effective_warmup,
        decay_steps=decay_steps,
        end_value=final_lr,
    )


def create_ppo_config(
    env,
    env_params,
    config_name: str,
    total_timesteps: int,
    ortho_mode: Optional[str] = None,
    ortho_coeff: float = 0.1,
    activation: str = "tanh",
    lr_schedule: str = "constant",
    learning_rate: float = 2.5e-4,
    final_lr: float = 1e-6,
    warmup_steps: int = 1000,
    use_bias: bool = True,
    use_orthogonal_init: bool = True,  # Important for PPO stability
    num_envs: int = 2048,
    num_steps: int = 128,
    num_epochs: int = 4,
    num_minibatches: int = 4,
    eval_freq: int = 500_000,
    network_type: str = "mlp",
    hidden_layer_sizes: Tuple[int, ...] = (256, 256, 256, 256),
    mlp_hidden_sizes: Optional[Tuple[int, ...]] = None,  # CNN MLP layers (default 4x256)
    conv_channels: int = 16,
    anneal_lr: bool = False,
    l2_init_coeff: Optional[float] = None,  # L2-Init regularization coefficient
    nap_enabled: bool = False,  # NaP (Normalize-and-Project)
    use_nap_layernorm: bool = False,  # NaP: LayerNorm before activations (non-learnable)
    scale_enabled: bool = False,  # Scale-AdaMO: per-layer learnable α
    scale_reg_coeff: Optional[float] = 0.01,  # Scale-AdaMO: log(α)² regularization
) -> Dict:
    """Create PPO configuration dict."""
    if network_type == "cnn":
        # CNN architecture for MinAtar
        # Conv(N, k=3, VALID) -> Flatten -> MLP
        cnn_mlp_sizes = mlp_hidden_sizes if mlp_hidden_sizes is not None else (64, 64)  # PGX default
        agent_kwargs = {
            "network_type": "cnn",
            "conv_channels": conv_channels,
            "mlp_hidden_sizes": cnn_mlp_sizes,
            "kernel_size": 3,
            "activation": activation,
            "use_bias": use_bias,
            "use_orthogonal_init": use_orthogonal_init,
            "use_nap_layernorm": use_nap_layernorm,
        }
    else:
        # MLP architecture (configurable, default: 4x256 per Lyle et al.)
        agent_kwargs = {
            "network_type": "mlp",
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "use_bias": use_bias,
            "use_orthogonal_init": use_orthogonal_init,
            "use_nap_layernorm": use_nap_layernorm,
        }

    config = {
        "env": env,
        "env_params": env_params,
        "agent_kwargs": agent_kwargs,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "num_epochs": num_epochs,
        "num_minibatches": num_minibatches,
        "learning_rate": learning_rate,
        "anneal_lr": anneal_lr,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "skip_initial_evaluation": True,
    }

    if ortho_mode and ortho_mode != "none":
        config["ortho_mode"] = ortho_mode
        config["ortho_coeff"] = ortho_coeff

    # L2-Init regularization
    if l2_init_coeff is not None:
        config["l2_init_coeff"] = l2_init_coeff

    # NaP (Normalize-and-Project)
    if nap_enabled:
        config["nap_enabled"] = True

    # Scale-AdaMO (per-layer learnable scaling)
    if scale_enabled:
        config["scale_enabled"] = True
        if scale_reg_coeff is not None:
            config["scale_reg_coeff"] = scale_reg_coeff

    # Calculate total gradient steps per game (used by multiple schedules)
    iteration_steps = num_envs * num_steps
    num_iterations = max(1, total_timesteps // iteration_steps)
    total_grad_steps = num_iterations * num_epochs * num_minibatches

    # Handle learning rate schedules
    if lr_schedule == "lyle_continual":
        # Adjust warmup to be proportional (max 10% of total, or specified warmup)
        effective_warmup = min(warmup_steps, max(1, total_grad_steps // 10))
        print(f"  LyleLR schedule: {total_grad_steps} grad steps, {effective_warmup} warmup steps")

        config["lr_schedule_fn"] = create_lyle_continual_schedule(
            peak_lr=learning_rate,
            final_lr=final_lr,
            warmup_steps=effective_warmup,
            total_steps=total_grad_steps,
        )
    return config


# Experiment configurations
#
# Architecture notes:
# - MLP: 4x256 layers (Lyle et al.) - deeper networks show more plasticity loss
# - CNN: pgx MinAtar style (conv32-k2 + avgpool + mlp64x3) - standard for MinAtar
#
# Activation notes:
# - GroupSort: Only for MLP layers, designed for FC networks (1-Lipschitz)
# - ReLU: Standard for CNNs, groupsort doesn't work well with conv layers
#
# =============================================================================
# PGX-matched configs (proven to work on MinAtar)
# Key differences: num_minibatches=128 gives ~24x more gradient steps
# =============================================================================
EXPERIMENT_CONFIGS_PGX = [
    {
        "name": "pgx_cnn_baseline",
        "network_type": "cnn",
        "conv_channels": 32,              # PGX: Conv(32, k=2)
        "mlp_hidden_sizes": (64, 64),     # PGX: 2x64 MLP
        "ortho_mode": None,
        "activation": "relu",             # ReLU for MinAtar
        "lr_schedule": "constant",
        "learning_rate": 3e-4,            # PGX: 0.0003
        "num_epochs": 3,                  # PGX: 3
        "num_minibatches": 128,           # PGX: minibatch_size=4096 -> 128 minibatches
        "anneal_lr": True,                # Important for stability
        "use_bias": True,
        "use_orthogonal_init": True,
    },
    {
        "name": "pgx_cnn_adamo",
        "network_type": "cnn",
        "conv_channels": 32,
        "mlp_hidden_sizes": (64, 64),
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "relu",             # ReLU for CNN (groupsort incompatible)
        "lr_schedule": "constant",
        "learning_rate": 3e-4,
        "num_epochs": 3,
        "num_minibatches": 128,
        "anneal_lr": True,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
]

# =============================================================================
# MLP configs (Lyle et al. style with 4x256 layers for plasticity research)
# =============================================================================
EXPERIMENT_CONFIGS_MLP = [
    {
        "name": "mlp_baseline",
        "network_type": "mlp",
        "hidden_layer_sizes": (256, 256, 256, 256),  # 4 layers per Lyle et al.
        "ortho_mode": None,
        "activation": "relu",             # ReLU for MinAtar
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,           # High for more grad steps
        "use_bias": True,
        "use_orthogonal_init": True,
    },
    {
        "name": "mlp_adamo",
        "network_type": "mlp",
        "hidden_layer_sizes": (256, 256, 256, 256),
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "groupsort",  # 1-Lipschitz activation for ortho networks
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": False,  # Disable bias for ortho experiments
        "use_orthogonal_init": True,
    },
    {
        "name": "mlp_adamo_lyle_lr",
        "network_type": "mlp",
        "hidden_layer_sizes": (256, 256, 256, 256),
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "groupsort",
        "lr_schedule": "linear",
        "learning_rate": 6.25e-5,   # Lyle et al. initial LR
        "final_lr": 1e-6,           # Lyle et al. final LR
        "num_minibatches": 128,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
    {
        "name": "mlp_adamo_lyle_continual",
        "network_type": "mlp",
        "hidden_layer_sizes": (256, 256, 256, 256),
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "groupsort",
        "lr_schedule": "lyle_continual",  # Warmup + cosine decay, reset per game
        "learning_rate": 6.25e-4,         # Peak LR after warmup
        "warmup_steps": 1000,
        "final_lr": 1e-6,
        "num_minibatches": 128,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
    # Small network variants - 64x4 (deeper, more plasticity stress)
    {
        "name": "mlp_baseline_small",
        "network_type": "mlp",
        "hidden_layer_sizes": (64, 64, 64, 64),
        "ortho_mode": None,
        "activation": "relu",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": True,
        "use_orthogonal_init": True,
    },
    {
        "name": "mlp_adamo_small",
        "network_type": "mlp",
        "hidden_layer_sizes": (64, 64, 64, 64),
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "groupsort",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
    # AdamO with ReLU activation (ablation: orthogonalization without GroupSort)
    {
        "name": "mlp_adamo_relu_small",
        "network_type": "mlp",
        "hidden_layer_sizes": (64, 64, 64, 64),
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "relu",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
    # Small network variants - 64x3 (shallower alternative)
    {
        "name": "mlp_baseline_small3",
        "network_type": "mlp",
        "hidden_layer_sizes": (64, 64, 64),
        "ortho_mode": None,
        "activation": "relu",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": True,
        "use_orthogonal_init": True,
    },
    {
        "name": "mlp_adamo_small3",
        "network_type": "mlp",
        "hidden_layer_sizes": (64, 64, 64),
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "groupsort",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
    # L2-Init baselines (regenerative regularization)
    {
        "name": "mlp_l2_init",
        "network_type": "mlp",
        "hidden_layer_sizes": (256, 256, 256, 256),
        "l2_init_coeff": 0.001,
        "ortho_mode": None,
        "activation": "relu",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": True,
        "use_orthogonal_init": True,
    },
    {
        "name": "mlp_l2_init_0.01",
        "network_type": "mlp",
        "hidden_layer_sizes": (256, 256, 256, 256),
        "l2_init_coeff": 0.01,  # Higher coefficient per literature
        "ortho_mode": None,
        "activation": "relu",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": True,
        "use_orthogonal_init": True,
    },
    # L2-Init + AdaMO combined
    {
        "name": "mlp_l2_init_adamo",
        "network_type": "mlp",
        "hidden_layer_sizes": (256, 256, 256, 256),
        "l2_init_coeff": 0.001,
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "groupsort",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
    # Small network variants with L2-Init (64x3)
    {
        "name": "mlp_l2_init_small",
        "network_type": "mlp",
        "hidden_layer_sizes": (64, 64, 64),
        "l2_init_coeff": 0.001,
        "ortho_mode": None,
        "activation": "relu",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": True,
        "use_orthogonal_init": True,
    },
    # Small network variants with L2-Init (64x4) - matches paper experiment
    {
        "name": "mlp_l2_init_small4",
        "network_type": "mlp",
        "hidden_layer_sizes": (64, 64, 64, 64),
        "l2_init_coeff": 0.001,
        "ortho_mode": None,
        "activation": "relu",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": True,
        "use_orthogonal_init": True,
    },
    {
        "name": "mlp_l2_init_small4_0.01",
        "network_type": "mlp",
        "hidden_layer_sizes": (64, 64, 64, 64),
        "l2_init_coeff": 0.01,  # Higher coefficient
        "ortho_mode": None,
        "activation": "relu",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": True,
        "use_orthogonal_init": True,
    },
    # NaP (Normalize-and-Project) baselines
    {
        "name": "mlp_nap",
        "network_type": "mlp",
        "hidden_layer_sizes": (256, 256, 256, 256),
        "nap_enabled": True,
        "ortho_mode": None,
        "activation": "relu",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": True,
        "use_orthogonal_init": True,
    },
    # NaP with small network (64x4) for faster experiments
    {
        "name": "mlp_nap_small",
        "network_type": "mlp",
        "hidden_layer_sizes": (64, 64, 64, 64),
        "nap_enabled": True,
        "use_nap_layernorm": True,  # Normalize (LayerNorm) and Project
        "ortho_mode": None,
        "activation": "relu",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": False,  # Paper: "we remove bias terms as these are made redundant by the learnable offset"
        "use_orthogonal_init": True,
    },
    # ==========================================================================
    # Scale-AdaMO: AdaMO with per-layer learnable scaling
    # ==========================================================================
    # Scale-AdaMO addresses the "scaling bottleneck" in orthonormal networks:
    # - Hidden layers are orthonormalized (W@W.T ≈ I), preserving signal magnitude
    # - Output layer is the ONLY layer that can scale outputs
    # - Different games have different return magnitudes (2-150x range)
    # - Per-layer learnable α allows scaling to be distributed across layers
    #   while maintaining orthonormality benefits (stable gradients, feature preservation)
    {
        "name": "mlp_scale_adamo",
        "network_type": "mlp",
        "hidden_layer_sizes": (256, 256, 256, 256),
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "scale_enabled": True,
        "scale_reg_coeff": 0.01,  # log(α)² regularization to keep α near 1
        "activation": "groupsort",  # 1-Lipschitz activation
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
    # Scale-AdaMO with small network (64x4) for faster experiments
    {
        "name": "mlp_scale_adamo_small",
        "network_type": "mlp",
        "hidden_layer_sizes": (64, 64, 64, 64),
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "scale_enabled": True,
        "scale_reg_coeff": 0.01,
        "activation": "groupsort",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
]

# =============================================================================
# CNN configs: deeper MLP for AdaMO research (different from PGX baseline)
# Note: AdaMO with CNN uses ReLU throughout - groupsort not compatible with conv layers
# =============================================================================
EXPERIMENT_CONFIGS_CNN = [
    {
        "name": "cnn_baseline",
        "network_type": "cnn",
        "conv_channels": 32,
        "mlp_hidden_sizes": (256, 256, 256, 256),  # 4x256 MLP for fair comparison
        "ortho_mode": None,
        "activation": "relu",
        "lr_schedule": "constant",
        "learning_rate": 3e-4,
        "num_minibatches": 128,           # High for more grad steps
        "anneal_lr": True,
        "use_bias": True,
        "use_orthogonal_init": True,
    },
    {
        "name": "cnn_adamo",
        "network_type": "cnn",
        "conv_channels": 32,
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "relu",
        "lr_schedule": "constant",
        "learning_rate": 3e-4,
        "num_envs": 4096,             # Match pgx_baseline
        "num_steps": 128,
        "num_epochs": 3,              # Match pgx_baseline
        "num_minibatches": 128,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
    {
        "name": "cnn_adamo_lyle_lr",
        "network_type": "cnn",
        "conv_channels": 32,
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "relu",
        "lr_schedule": "linear",
        "learning_rate": 6.25e-5,
        "final_lr": 1e-6,
        "num_minibatches": 128,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
    {
        "name": "cnn_adamo_lyle_continual",
        "network_type": "cnn",
        "conv_channels": 32,
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "relu",
        "lr_schedule": "lyle_continual",
        "learning_rate": 6.25e-4,
        "warmup_steps": 1000,
        "num_envs": 4096,             # Match pgx_baseline
        "num_steps": 128,
        "num_epochs": 3,              # Match pgx_baseline
        "final_lr": 1e-6,
        "num_minibatches": 128,
        "anneal_lr": True,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
]

# =============================================================================
# Recommended Baseline Configs (fixes update ratio for proper learning)
# Key insight: num_minibatches controls gradient steps per data collection
# Target range for MinAtar PPO: total_grad_steps ~ 10k-50k at 20M frames
# =============================================================================
EXPERIMENT_CONFIGS_BASELINE = [
    # Config A: Pgx-style (Maximum throughput, matches published results)
    # For 20M: B=524k, updates≈38, grad_steps/update=384, total≈14.6k
    {
        "name": "pgx_baseline",
        "network_type": "cnn",
        "conv_channels": 32,
        "mlp_hidden_sizes": (256, 256, 256, 256),  # 4-layer head for plasticity research
        "ortho_mode": None,
        "activation": "relu",
        "use_orthogonal_init": True,  # Important for PPO stability
        "lr_schedule": "constant",
        "learning_rate": 3e-4,
        "num_envs": 4096,
        "num_steps": 128,
        "num_epochs": 3,
        "num_minibatches": 128,  # minibatch_size=4096
        "use_bias": True,
    },
    # Config B: Stable/Debuggable (More policy updates, easier to track learning)
    # For 20M: B=65k, updates≈305, grad_steps/update=128, total≈39k
    {
        "name": "stable_baseline",
        "network_type": "cnn",
        "conv_channels": 32,
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "ortho_mode": None,
        "activation": "relu",
        "use_orthogonal_init": True,
        "learning_rate": 2.5e-4,
        "num_envs": 512,
        "num_steps": 128,
        "num_epochs": 4,
        "num_minibatches": 32,  # minibatch_size=2048
        "anneal_lr": True,
        "use_bias": True,
    },
    # Config C: Fast + Stable (Good throughput without fragility)
    # For 20M: B=262k, updates≈76, grad_steps/update=512, total≈39k
    {
        "name": "fast_stable_baseline",
        "network_type": "cnn",
        "conv_channels": 32,
        "mlp_hidden_sizes": (256, 256, 256, 256),
        "ortho_mode": None,
        "activation": "relu",
        "use_orthogonal_init": True,
        "learning_rate": 3e-4,
        "num_envs": 2048,
        "num_steps": 128,
        "num_epochs": 4,
        "num_minibatches": 128,  # minibatch_size=2048
        "anneal_lr": True,
        "use_bias": True,
    },
]

# Combined for backward compatibility
EXPERIMENT_CONFIGS = EXPERIMENT_CONFIGS_PGX + EXPERIMENT_CONFIGS_MLP + EXPERIMENT_CONFIGS_CNN + EXPERIMENT_CONFIGS_BASELINE

# Legacy aliases (kept for backward compatibility, updated with proper settings)
EXPERIMENT_CONFIGS_LEGACY = [
    {
        "name": "baseline",
        "network_type": "mlp",
        "ortho_mode": None,
        "activation": "relu",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "anneal_lr": True,
        "use_bias": True,
        "use_orthogonal_init": True,
    },
    {
        "name": "ortho_adamo",
        "network_type": "mlp",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "groupsort",
        "lr_schedule": "constant",
        "learning_rate": 2.5e-4,
        "num_minibatches": 128,
        "anneal_lr": True,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
    {
        "name": "ortho_adamo_lyle_lr",
        "network_type": "mlp",
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "groupsort",
        "lr_schedule": "linear",
        "learning_rate": 6.25e-5,
        "final_lr": 1e-6,
        "num_minibatches": 128,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
    {
        "name": "ortho_adamo_lyle_continual",
        "network_type": "mlp",
        "hidden_layer_sizes": (256, 256, 256, 256),
        "ortho_mode": "optimizer",
        "ortho_coeff": 0.1,
        "activation": "groupsort",
        "lr_schedule": "lyle_continual",  # Cosine warmup schedule, restarted per game
        "learning_rate": 6.25e-4,         # Peak LR (Rainbow default)
        "warmup_steps": 1000,
        "final_lr": 1e-6,
        "num_minibatches": 128,
        "use_bias": False,
        "use_orthogonal_init": True,
    },
]


def save_checkpoint(checkpoint_dir: Path, name: str, train_state, game_idx: int, cycle_idx: int, metadata: Dict):
    """Save checkpoint to disk."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}_cycle{cycle_idx}_game{game_idx}.ckpt"

    checkpoint = {
        "game_idx": game_idx,
        "cycle_idx": cycle_idx,
        "metadata": metadata,
    }

    # Serialize the train_state separately and save alongside metadata
    with open(checkpoint_path, "wb") as f:
        f.write(serialization.to_bytes((train_state, checkpoint)))

    print(f"  Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path, train_state_template):
    """Load checkpoint from disk."""
    with open(checkpoint_path, "rb") as f:
        train_state, checkpoint = serialization.from_bytes(
            (train_state_template, {"game_idx": 0, "cycle_idx": 0, "metadata": {}}),
            f.read()
        )
    return train_state, checkpoint


class ContinualTrainer:
    """Manages continual learning across multiple games."""

    def __init__(
        self,
        config_name: str,
        experiment_config: Dict,
        steps_per_game: int,
        num_cycles: int,
        num_envs: int = 2048,
        eval_freq: int = 500_000,
        checkpoint_dir: Optional[Path] = None,
        use_wandb: bool = False,
        wandb_project: str = "rejax-continual",
        permute_channels: bool = False,
        random_game_order: bool = False,
        seed: int = 0,
        exclude_games: Optional[List[str]] = None,
    ):
        self.config_name = config_name
        self.experiment_config = experiment_config
        self.steps_per_game = steps_per_game
        self.num_cycles = num_cycles
        self.num_envs = num_envs
        self.eval_freq = eval_freq
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.permute_channels = permute_channels
        self.random_game_order = random_game_order
        self.seed = seed

        # Filter games if exclusions specified
        self.game_list = [g for g in GAME_ORDER if g not in (exclude_games or [])]

        # Results storage
        self.results = {
            "config_name": config_name,
            "experiment_config": experiment_config,
            "steps_per_game": steps_per_game,
            "num_cycles": num_cycles,
            "games": self.game_list,
            "permute_channels": permute_channels,
            "random_game_order": random_game_order,
            "per_game_results": [],
        }

        # Cache PPO instances per game to avoid recompilation across cycles
        # This keeps only 5 compiled functions in memory (one per game)
        # When using channel permutation, cache is keyed by (game, cycle) instead
        self._ppo_cache = {}

    def _get_ppo_for_game(self, game_name: str, channel_perm: Optional[np.ndarray] = None) -> Tuple[PPO, Any, Any, Any]:
        """Get cached (PPO, train_chunk, train_chunk_with_metrics, cached_eval) for a game.

        Caches PPO, its JIT-compiled train_chunk (fast, no metrics), train_chunk_with_metrics
        (for research logging), and a cached eval function to avoid recompilation across cycles.

        Without caching, each cycle would create new function objects that JAX traces
        separately, causing OOM after ~14 games.

        The cached_eval is critical: the standard evaluate() has act as static_argnames,
        so each new act closure from make_act() triggers recompilation. cached_eval
        takes actor_params as a pytree argument instead, so it compiles once per game.

        When channel permutation is enabled, cache key includes permutation tuple to
        ensure different permutations get separate compiled functions.
        """
        # Cache key includes permutation when enabled
        cache_key = (game_name, tuple(channel_perm) if channel_perm is not None else None)

        if cache_key not in self._ppo_cache:
            ppo = self._create_ppo_for_game(game_name, channel_perm)

            # Fast train_chunk (no metrics) - use for non-research runs
            @jax.jit
            def train_chunk(ts, num_iters):
                def body(_, ts):
                    return ppo.train_iteration(ts)
                return jax.lax.fori_loop(0, num_iters, body, ts)

            # Train_chunk with metrics - use for research/logging
            # Cache for metrics train chunks (keyed by num_iters since scan needs static length)
            metrics_chunk_cache = {}

            def get_train_chunk_with_metrics(n_iters):
                """Get or create a JIT-compiled metrics train chunk for given iteration count."""
                if n_iters not in metrics_chunk_cache:
                    @jax.jit
                    def _train_chunk_with_metrics(ts):
                        def body(ts, _):
                            ts, metrics = ppo.train_iteration_with_metrics(ts)
                            return ts, metrics
                        ts, all_metrics = jax.lax.scan(body, ts, None, length=n_iters)
                        mean_metrics = jax.tree.map(jnp.mean, all_metrics)
                        return ts, mean_metrics
                    metrics_chunk_cache[n_iters] = _train_chunk_with_metrics
                return metrics_chunk_cache[n_iters]

            # Create cached eval function that doesn't recompile on each call
            cached_eval = create_cached_eval_fn(ppo)

            self._ppo_cache[cache_key] = (ppo, train_chunk, get_train_chunk_with_metrics, cached_eval)
        return self._ppo_cache[cache_key]

    def _create_ppo_for_game(self, game_name: str, channel_perm: Optional[np.ndarray] = None) -> PPO:
        """Create PPO instance configured for a specific game."""
        env, env_params = create_padded_env(game_name, channel_perm)
        # Use config's num_envs if specified, otherwise fall back to constructor param
        effective_num_envs = self.experiment_config.get("num_envs", self.num_envs)
        config = create_ppo_config(
            env=env,
            env_params=env_params,
            config_name=self.config_name,
            total_timesteps=self.steps_per_game,
            ortho_mode=self.experiment_config.get("ortho_mode"),
            ortho_coeff=self.experiment_config.get("ortho_coeff", 0.1),
            activation=self.experiment_config.get("activation", "tanh"),
            lr_schedule=self.experiment_config.get("lr_schedule", "constant"),
            learning_rate=self.experiment_config.get("learning_rate", 2.5e-4),
            final_lr=self.experiment_config.get("final_lr", 1e-6),
            warmup_steps=self.experiment_config.get("warmup_steps", 1000),
            use_bias=self.experiment_config.get("use_bias", True),
            use_orthogonal_init=self.experiment_config.get("use_orthogonal_init", True),
            num_envs=effective_num_envs,
            num_steps=self.experiment_config.get("num_steps", 128),
            num_epochs=self.experiment_config.get("num_epochs", 4),
            num_minibatches=self.experiment_config.get("num_minibatches", 4),
            eval_freq=self.eval_freq,
            network_type=self.experiment_config.get("network_type", "mlp"),
            hidden_layer_sizes=self.experiment_config.get("hidden_layer_sizes", (256, 256, 256, 256)),
            mlp_hidden_sizes=self.experiment_config.get("mlp_hidden_sizes"),
            conv_channels=self.experiment_config.get("conv_channels", 32),
            anneal_lr=self.experiment_config.get("anneal_lr", False),
            l2_init_coeff=self.experiment_config.get("l2_init_coeff"),
            nap_enabled=self.experiment_config.get("nap_enabled", False),
            use_nap_layernorm=self.experiment_config.get("use_nap_layernorm", False),
            scale_enabled=self.experiment_config.get("scale_enabled", False),
            scale_reg_coeff=self.experiment_config.get("scale_reg_coeff", 0.01),
        )
        return PPO.create(**config)

    def _transfer_train_state(self, old_ts, new_ppo, rng):
        """Transfer network weights to a new environment's train state.

        Always resets optimizer state (Adam momentum) at task boundaries
        to give each task a fair start without bias from previous task's
        gradient history.

        Preserves init_params from the original initialization for L2-Init
        regularization (weights should stay close to original random init
        throughout all games).
        """
        new_ts = new_ppo.init_state(rng)

        # Always reset optimizer for continual learning
        # - Transfers network weights (features + heads)
        # - Fresh optimizer state (zeroed Adam momentum)
        # - Reset global_step (for any LR schedules like lyle_continual)
        # - Preserve init_params from original training start (for L2-Init)
        new_ts = new_ts.replace(
            actor_ts=new_ts.actor_ts.replace(
                params=old_ts.actor_ts.params,
                # opt_state stays fresh from new_ppo.init_state
            ),
            critic_ts=new_ts.critic_ts.replace(
                params=old_ts.critic_ts.params,
                # opt_state stays fresh from new_ppo.init_state
            ),
            global_step=jnp.array(0),
            # Preserve original init_params for L2-Init regularization
            actor_init_params=old_ts.actor_init_params,
            critic_init_params=old_ts.critic_init_params,
            # Preserve original init_norms for NaP
            actor_init_norms=old_ts.actor_init_norms,
            critic_init_norms=old_ts.critic_init_norms,
        )
        return new_ts

    def _make_progress_callback(self, game_name: str, cycle_idx: int, original_eval_callback):
        """Create a callback for logging progress."""
        start_time = [time.time()]

        def progress_callback(ppo, train_state, rng):
            lengths, returns = original_eval_callback(ppo, train_state, rng)

            def log(step, returns):
                elapsed = time.time() - start_time[0]
                steps_per_sec = step.item() / elapsed if elapsed > 0 else 0
                mean_ret = returns.mean().item()
                print(f"    [Cycle {cycle_idx} | {game_name}] step={step.item():,} "
                      f"| return={mean_ret:.1f} | {steps_per_sec:,.0f} steps/s")

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        f"cycle_{cycle_idx}/{game_name}/return": mean_ret,
                        f"cycle_{cycle_idx}/{game_name}/step": step.item(),
                        "global_step": train_state.global_step.item(),
                    })

            jax.experimental.io_callback(log, (), train_state.global_step, returns)
            return lengths, returns

        return progress_callback

    def train_single_game(self, game_name: str, train_state, rng, cycle_idx: int,
                          channel_perm: Optional[np.ndarray] = None,
                          game_idx_in_cycle: int = 0):
        """Train on a single game, optionally continuing from existing train_state."""
        # Use cached PPO, train_chunk, and eval to avoid recompilation across cycles
        ppo, train_chunk, get_train_chunk_with_metrics, cached_eval = self._get_ppo_for_game(
            game_name, channel_perm)

        if train_state is not None:
            # Transfer weights from previous game
            rng, init_rng = jax.random.split(rng)
            train_state = self._transfer_train_state(train_state, ppo, init_rng)
        else:
            # Fresh initialization
            rng, init_rng = jax.random.split(rng)
            train_state = ppo.init_state(init_rng)

        # Train
        print(f"  Training on {game_name}...", flush=True)
        start_time = time.time()

        # Use train_iteration manually to allow step-by-step control
        iteration_steps = ppo.num_envs * ppo.num_steps
        num_iterations = int(np.ceil(self.steps_per_game / iteration_steps))
        eval_interval = int(np.ceil(self.eval_freq / iteration_steps))

        # train_chunk is cached with PPO - no recompilation after cycle 0
        # Compile on first call
        print(f"    Compiling...", flush=True)
        compile_start = time.time()

        # Run training in chunks with periodic evaluation
        total_iters = 0
        first_chunk = True
        while total_iters < num_iterations:
            chunk_size = min(eval_interval, num_iterations - total_iters)

            # Use metrics version when wandb is enabled, fast version otherwise
            if self.use_wandb:
                train_chunk_with_metrics = get_train_chunk_with_metrics(chunk_size)
                train_state, chunk_metrics = train_chunk_with_metrics(train_state)
            else:
                train_state = train_chunk(train_state, chunk_size)
                chunk_metrics = None

            jax.block_until_ready(train_state)

            if first_chunk:
                compile_time = time.time() - compile_start
                print(f"    Compiled in {compile_time:.1f}s", flush=True)
                first_chunk = False

            # Evaluation (use cached_eval to avoid recompilation)
            rng, eval_rng = jax.random.split(rng)
            lengths, returns = cached_eval(train_state.actor_ts.params, eval_rng)

            # Progress
            current_steps = (total_iters + chunk_size) * iteration_steps
            elapsed = time.time() - start_time
            steps_per_sec = current_steps / elapsed if elapsed > 0 else 0
            mean_return = float(returns.mean())
            pct = 100 * current_steps / self.steps_per_game
            print(f"    [{game_name}] {current_steps:,}/{self.steps_per_game:,} ({pct:.0f}%) "
                  f"| return={mean_return:.1f} | {steps_per_sec:,.0f} steps/s", flush=True)

            if self.use_wandb:
                import wandb
                # Calculate cumulative step for Lyle-style plots
                # Use game_idx_in_cycle (position in shuffled/filtered order) not GAME_ORDER.index
                num_games = len(self.game_list)
                cumulative_step = (
                    cycle_idx * num_games * self.steps_per_game +  # previous cycles
                    game_idx_in_cycle * self.steps_per_game +      # previous games this cycle
                    current_steps                                   # current game progress
                )

                # Game ID is consistent across shuffles (based on original GAME_ORDER)
                game_id = GAME_ORDER.index(game_name) if game_name in GAME_ORDER else -1

                log_dict = {
                    # Per-game tracking
                    f"train/{game_name}/return": mean_return,
                    f"train/{game_name}/step": current_steps,
                    # Per-cycle tracking (for Lyle-style plots)
                    f"cycle_{cycle_idx}/return": mean_return,
                    f"cycle_{cycle_idx}/game": game_name,
                    # Global tracking
                    "return": mean_return,
                    "cycle": cycle_idx,
                    "game_idx": game_idx_in_cycle,  # position in this cycle (0,1,2,3)
                    "game_id": game_id,              # consistent ID: Breakout=0, Asterix=1, SI=2, Freeway=3, Seaquest=4
                    "game": game_name,
                }

                # Add training metrics from chunk_metrics
                if chunk_metrics is not None:
                    log_dict["loss/policy"] = float(chunk_metrics["loss/policy"])
                    log_dict["loss/value"] = float(chunk_metrics["loss/value"])
                    log_dict["loss/entropy"] = float(chunk_metrics["loss/entropy"])
                    log_dict["ppo/approx_kl"] = float(chunk_metrics["ppo/approx_kl"])
                    log_dict["ppo/clip_fraction"] = float(chunk_metrics["ppo/clip_fraction"])
                    log_dict["gram/actor"] = float(chunk_metrics["gram/actor"])
                    log_dict["gram/critic"] = float(chunk_metrics["gram/critic"])
                    log_dict["train/learning_rate"] = float(chunk_metrics["train/learning_rate"])
                    # L2-init distance metrics (always logged for diagnostics)
                    log_dict["l2_init/actor_distance"] = float(chunk_metrics["l2_init/actor_distance"])
                    log_dict["l2_init/critic_distance"] = float(chunk_metrics["l2_init/critic_distance"])

                wandb.log(log_dict, step=cumulative_step)

            total_iters += chunk_size

        elapsed = time.time() - start_time
        steps_per_sec = self.steps_per_game / elapsed

        # Final evaluation (use cached_eval to avoid recompilation)
        rng, eval_rng = jax.random.split(rng)
        lengths, returns = cached_eval(train_state.actor_ts.params, eval_rng)
        final_return = float(returns.mean())

        print(f"  Completed {game_name}: final_return={final_return:.1f}, "
              f"elapsed={elapsed:.1f}s, {steps_per_sec:,.0f} steps/s", flush=True)

        return train_state, rng, {
            "game": game_name,
            "cycle": cycle_idx,
            "final_return": final_return,
            "elapsed_s": elapsed,
            "steps_per_sec": steps_per_sec,
        }

    def evaluate_all_games(self, train_state, rng, cycle_idx: int, current_game_idx: int):
        """Evaluate current policy on all games in the game list."""
        print(f"  Evaluating on all games after game {current_game_idx}...")
        eval_results = {}

        for game_name in self.game_list:
            # Use cached_eval to avoid recompilation
            ppo, _, _, cached_eval = self._get_ppo_for_game(game_name)

            # Evaluate using current actor params directly
            # (cached_eval works with any actor params, no transfer needed for eval)
            rng, eval_rng = jax.random.split(rng)
            lengths, returns = cached_eval(train_state.actor_ts.params, eval_rng)
            mean_return = float(returns.mean())
            eval_results[game_name] = mean_return
            print(f"    {game_name}: {mean_return:.1f}")

        return eval_results, rng

    def _generate_channel_perm(self, game_name: str, cycle_idx: int) -> Optional[np.ndarray]:
        """Generate a channel permutation for a game if permute_channels is enabled.

        Always permutes all 10 unified channels (including padding zeros).
        This forces the network to learn which channels contain content
        (at least one non-zero pixel) vs padding (all zeros).
        """
        if not self.permute_channels:
            return None
        # Use deterministic seed based on game, cycle, and base seed
        # Use game index to avoid hash overflow issues
        game_idx = GAME_ORDER.index(game_name) if game_name in GAME_ORDER else 0
        perm_seed = abs(self.seed * 10000 + game_idx * 100 + cycle_idx) % (2**31)
        rng = np.random.default_rng(perm_seed)
        # Always permute all 10 unified channels (content + padding)
        perm = rng.permutation(UNIFIED_CHANNELS)
        return perm

    def run(self, rng, start_cycle: int = 0, start_game: int = 0, initial_train_state=None):
        """Run the full continual learning experiment."""
        train_state = initial_train_state

        # Print update diagnostics at start
        print(f"\nConfig: {self.config_name}")
        print_update_diagnostics(self.experiment_config, self.steps_per_game)
        if self.permute_channels:
            print(f"  Channel permutation: ENABLED")
        if self.random_game_order:
            print(f"  Random game order: ENABLED")

        for cycle_idx in range(start_cycle, self.num_cycles):
            print(f"\n{'='*60}", flush=True)
            print(f"Cycle {cycle_idx + 1}/{self.num_cycles}", flush=True)
            print(f"{'='*60}", flush=True)

            # Determine game order for this cycle
            if self.random_game_order:
                # Shuffle game order with deterministic seed per cycle
                order_rng = np.random.default_rng(self.seed + cycle_idx * 7919)
                cycle_game_order = order_rng.permutation(self.game_list).tolist()
                print(f"  Game order: {' -> '.join(cycle_game_order)}")
            else:
                cycle_game_order = self.game_list

            game_start = start_game if cycle_idx == start_cycle else 0

            for game_idx, game_name in enumerate(cycle_game_order[game_start:], start=game_start):
                # Generate channel permutation for this game/cycle
                channel_perm = self._generate_channel_perm(game_name, cycle_idx)
                if channel_perm is not None:
                    print(f"\nGame {game_idx + 1}/{len(cycle_game_order)}: {game_name} (channel perm: {channel_perm})", flush=True)
                else:
                    print(f"\nGame {game_idx + 1}/{len(cycle_game_order)}: {game_name}", flush=True)

                # Train on this game (PPO instances cached to avoid recompilation)
                train_state, rng, game_result = self.train_single_game(
                    game_name, train_state, rng, cycle_idx,
                    channel_perm=channel_perm, game_idx_in_cycle=game_idx
                )
                self.results["per_game_results"].append(game_result)

                # Evaluate on all games at game boundaries
                eval_results, rng = self.evaluate_all_games(
                    train_state, rng, cycle_idx, game_idx
                )
                game_result["eval_all_games"] = eval_results

                # Save checkpoint
                if self.checkpoint_dir:
                    save_checkpoint(
                        self.checkpoint_dir,
                        self.config_name,
                        train_state,
                        game_idx,
                        cycle_idx,
                        metadata={"experiment_config": self.experiment_config},
                    )

                # Log to wandb
                if self.use_wandb:
                    import wandb
                    for eval_game, eval_return in eval_results.items():
                        wandb.log({
                            f"eval_after_{game_name}/{eval_game}": eval_return,
                            "game_idx": game_idx,
                            "cycle_idx": cycle_idx,
                        })

                # Save intermediate results after each game completes
                if self.checkpoint_dir:
                    intermediate_path = self.checkpoint_dir.parent / f"{self.config_name}_intermediate.json"
                    with open(intermediate_path, "w") as f:
                        json.dump(self.results, f, indent=2, default=str)
                    print(f"  Saved intermediate results: {intermediate_path}")

            # Reset start_game for subsequent cycles
            start_game = 0

        return self.results


def create_ppo_for_game_with_config(
    game_name: str,
    experiment_config: Dict,
    steps_per_game: int,
    num_envs: int,
) -> PPO:
    """Create PPO instance for a game with the given experiment config."""
    env, env_params = create_padded_env(game_name)
    # Use config's num_envs if specified, otherwise fall back to function param
    effective_num_envs = experiment_config.get("num_envs", num_envs)
    config = create_ppo_config(
        env=env,
        env_params=env_params,
        config_name=experiment_config["name"],
        total_timesteps=steps_per_game,
        ortho_mode=experiment_config.get("ortho_mode"),
        ortho_coeff=experiment_config.get("ortho_coeff", 0.1),
        activation=experiment_config.get("activation", "tanh"),
        lr_schedule=experiment_config.get("lr_schedule", "constant"),
        learning_rate=experiment_config.get("learning_rate", 2.5e-4),
        final_lr=experiment_config.get("final_lr", 1e-6),
        warmup_steps=experiment_config.get("warmup_steps", 1000),
        use_bias=experiment_config.get("use_bias", True),
        use_orthogonal_init=experiment_config.get("use_orthogonal_init", True),
        num_envs=effective_num_envs,
        num_steps=experiment_config.get("num_steps", 128),
        num_epochs=experiment_config.get("num_epochs", 4),
        num_minibatches=experiment_config.get("num_minibatches", 4),
        eval_freq=steps_per_game,  # Eval only at end for parallel mode
        network_type=experiment_config.get("network_type", "mlp"),
        hidden_layer_sizes=experiment_config.get("hidden_layer_sizes", (256, 256, 256, 256)),
        mlp_hidden_sizes=experiment_config.get("mlp_hidden_sizes"),
        conv_channels=experiment_config.get("conv_channels", 32),
        anneal_lr=experiment_config.get("anneal_lr", False),
        l2_init_coeff=experiment_config.get("l2_init_coeff"),
        nap_enabled=experiment_config.get("nap_enabled", False),
        use_nap_layernorm=experiment_config.get("use_nap_layernorm", False),
        scale_enabled=experiment_config.get("scale_enabled", False),
        scale_reg_coeff=experiment_config.get("scale_reg_coeff", 0.01),
    )
    return PPO.create(**config)


def run_experiment_parallel(
    experiment_config: Dict,
    steps_per_game: int,
    num_cycles: int,
    num_seeds: int,
    num_envs: int,
    seed: int = 0,
):
    """
    Run experiment with all seeds in parallel using vmap.

    This is much faster than sequential seed execution but doesn't support
    wandb logging or checkpointing during training.
    """
    config_name = experiment_config["name"]
    print(f"\n{'#'*70}")
    print(f"# Experiment (PARALLEL): {config_name}")
    print(f"# Seeds: {num_seeds}, Steps per game: {steps_per_game:,}, Cycles: {num_cycles}")
    print(f"{'#'*70}")

    # Print update diagnostics
    print_update_diagnostics(experiment_config, steps_per_game)

    # Create PPO instances for all games (same config, different envs)
    ppos = [create_ppo_for_game_with_config(game, experiment_config, steps_per_game, num_envs)
            for game in GAME_ORDER]

    # Create vmapped train functions for each game
    vmap_trains = [jax.jit(jax.vmap(PPO.train, in_axes=(None, 0))) for _ in ppos]

    # Generate seeds
    rngs = jax.random.split(jax.random.PRNGKey(seed), num_seeds)

    # Results storage: (num_cycles, num_games, num_seeds)
    all_returns = []

    print(f"\nCompiling training functions...")
    compile_start = time.time()

    # Compile by running first game
    first_train_states, _ = vmap_trains[0](ppos[0], rngs)
    jax.block_until_ready(first_train_states)
    print(f"  Compiled in {time.time() - compile_start:.1f}s")

    for cycle_idx in range(num_cycles):
        print(f"\n{'='*60}")
        print(f"Cycle {cycle_idx + 1}/{num_cycles}")
        print(f"{'='*60}")

        cycle_returns = []

        for game_idx, game_name in enumerate(GAME_ORDER):
            print(f"\n  Training on {game_name}...")
            start_time = time.time()

            ppo = ppos[game_idx]
            vmap_train = vmap_trains[game_idx]

            if cycle_idx == 0 and game_idx == 0:
                # Use already-computed first game results
                train_states = first_train_states
                # Re-evaluate to get returns
                eval_rngs = jax.random.split(jax.random.PRNGKey(seed + 1000), num_seeds)

                @jax.jit
                def vmap_eval(ppo, ts, rngs):
                    def eval_single(ts, rng):
                        return ppo.eval_callback(ppo, ts, rng)
                    return jax.vmap(eval_single)(ts, rngs)

                _, returns = vmap_eval(ppo, train_states, eval_rngs)
            else:
                # Transfer weights from previous game to this game's PPO
                if game_idx > 0 or cycle_idx > 0:
                    # Initialize fresh train states for this game
                    init_rngs = jax.random.split(jax.random.PRNGKey(seed + cycle_idx * 100 + game_idx), num_seeds)

                    @jax.jit
                    def vmap_init(ppo, rngs):
                        return jax.vmap(ppo.init_state)(rngs)

                    new_train_states = vmap_init(ppo, init_rngs)

                    # Always reset optimizer for continual learning
                    # - Transfers network weights (features + heads)
                    # - Fresh optimizer state (zeroed Adam momentum)
                    # - Reset global_step (for any LR schedules like lyle_continual)
                    new_train_states = new_train_states.replace(
                        actor_ts=new_train_states.actor_ts.replace(
                            params=train_states.actor_ts.params,
                        ),
                        critic_ts=new_train_states.critic_ts.replace(
                            params=train_states.critic_ts.params,
                        ),
                        global_step=jnp.zeros_like(train_states.global_step),
                    )

                    # Continue training from transferred weights
                    train_rngs = jax.random.split(jax.random.PRNGKey(seed + cycle_idx * 1000 + game_idx * 10), num_seeds)

                    @jax.jit
                    def vmap_continue_train(ppo, ts, rngs):
                        def train_single(ts, rng):
                            # Run training iterations
                            iteration_steps = ppo.num_envs * ppo.num_steps
                            num_iterations = int(np.ceil(steps_per_game / iteration_steps))
                            def body(_, ts):
                                return ppo.train_iteration(ts)
                            return jax.lax.fori_loop(0, num_iterations, body, ts)
                        return jax.vmap(train_single)(ts, rngs)

                    train_states = vmap_continue_train(ppo, new_train_states, train_rngs)
                else:
                    # Fresh training for first game of first cycle
                    train_states, _ = vmap_train(ppo, rngs)

                jax.block_until_ready(train_states)

                # Evaluate
                eval_rngs = jax.random.split(jax.random.PRNGKey(seed + 2000 + cycle_idx * 100 + game_idx), num_seeds)

                @jax.jit
                def vmap_eval(ppo, ts, rngs):
                    def eval_single(ts, rng):
                        return ppo.eval_callback(ppo, ts, rng)
                    return jax.vmap(eval_single)(ts, rngs)

                _, returns = vmap_eval(ppo, train_states, eval_rngs)

            # returns shape: (num_seeds, num_eval_episodes)
            mean_returns = returns.mean(axis=-1)  # (num_seeds,)
            elapsed = time.time() - start_time
            steps_per_sec = steps_per_game * num_seeds / elapsed if elapsed > 0 else 0

            print(f"    {game_name}: return={mean_returns.mean():.1f}±{mean_returns.std():.1f} "
                  f"| {steps_per_sec:,.0f} steps/s")

            cycle_returns.append({
                "game": game_name,
                "cycle": cycle_idx,
                "returns_per_seed": mean_returns.tolist(),
                "mean_return": float(mean_returns.mean()),
                "std_return": float(mean_returns.std()),
            })

        all_returns.append(cycle_returns)

    # Compile final results
    results = {
        "config_name": config_name,
        "experiment_config": experiment_config,
        "steps_per_game": steps_per_game,
        "num_cycles": num_cycles,
        "num_seeds": num_seeds,
        "games": GAME_ORDER,
        "results_by_cycle": all_returns,
    }

    return results


def run_experiment(
    experiment_config: Dict,
    steps_per_game: int,
    num_cycles: int,
    num_seeds: int,
    num_envs: int,
    eval_freq: int,
    checkpoint_dir: Path,
    use_wandb: bool,
    wandb_project: str,
    seed: int = 0,
):
    """Run a single experiment configuration with multiple seeds (sequential)."""
    config_name = experiment_config["name"]
    print(f"\n{'#'*70}")
    print(f"# Experiment: {config_name}")
    print(f"# Seeds: {num_seeds}, Steps per game: {steps_per_game:,}, Cycles: {num_cycles}")
    print(f"{'#'*70}")

    all_results = []

    for seed_idx in range(num_seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed_idx + 1}/{num_seeds}")
        print(f"{'='*60}")

        actual_seed = seed + seed_idx
        rng = jax.random.PRNGKey(actual_seed)

        if use_wandb:
            import wandb
            run_name = f"{config_name}_seed{actual_seed}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "experiment_config": experiment_config,
                    "steps_per_game": steps_per_game,
                    "num_cycles": num_cycles,
                    "seed": actual_seed,
                },
                reinit=True,
            )

        trainer = ContinualTrainer(
            config_name=f"{config_name}_seed{actual_seed}",
            experiment_config=experiment_config,
            steps_per_game=steps_per_game,
            num_cycles=num_cycles,
            num_envs=num_envs,
            eval_freq=eval_freq,
            checkpoint_dir=checkpoint_dir / config_name,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )

        results = trainer.run(rng)
        all_results.append(results)

        if use_wandb:
            import wandb
            wandb.finish()

    return all_results


def test_action_mapping():
    """Test that invalid actions map to no-op correctly."""
    print("Testing action mapping for each game...")

    for game_name in GAME_ORDER:
        game_info = MINATAR_GAMES[game_name]
        env, params = create_padded_env(game_name)

        print(f"\n{game_name}:")
        print(f"  Original actions: {game_info['actions']}")
        print(f"  Unified actions: {UNIFIED_ACTIONS}")

        # Test each action
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng, params)
        print(f"  Obs shape: {obs.shape}")

        for action in range(UNIFIED_ACTIONS):
            rng, step_rng = jax.random.split(rng)
            try:
                obs, state, reward, done, info = env.step(step_rng, state, action, params)
                valid = action < game_info['actions']
                print(f"  Action {action}: {'valid' if valid else 'mapped to no-op'}")
            except Exception as e:
                print(f"  Action {action}: ERROR - {e}")

    print("\nAction mapping test complete!")


def save_results(results: Dict, output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"continual_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Continual learning benchmark for MinAtar")
    parser.add_argument("--steps-per-game", type=int, default=10_000_000,
                        help="Training steps per game")
    parser.add_argument("--num-cycles", type=int, default=2,
                        help="Number of cycles through all games")
    parser.add_argument("--num-seeds", type=int, default=3,
                        help="Number of random seeds")
    parser.add_argument("--num-envs", type=int, default=2048,
                        help="Parallel environments")
    parser.add_argument("--eval-freq", type=int, default=250_000,
                        help="Evaluation frequency in steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/continual",
                        help="Directory for checkpoints")
    parser.add_argument("--output-dir", type=str, default="results/continual",
                        help="Directory for results JSON")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="rejax-continual",
                        help="W&B project name")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint for resuming")
    parser.add_argument("--test-action-mapping", action="store_true",
                        help="Test action mapping and exit")
    # Get all config names
    all_config_names = [c["name"] for c in EXPERIMENT_CONFIGS] + [c["name"] for c in EXPERIMENT_CONFIGS_LEGACY]
    parser.add_argument("--configs", nargs="+",
                        default=["pgx_baseline"],  # Recommended main baseline
                        choices=all_config_names,
                        help="Experiment configurations to run")
    parser.add_argument("--parallel-seeds", action="store_true",
                        help="Run all seeds in parallel using vmap (faster but no wandb/checkpoints)")
    parser.add_argument("--permute-channels", action="store_true",
                        help="Randomly permute observation channels for each game")
    parser.add_argument("--random-game-order", action="store_true",
                        help="Shuffle game order each cycle")
    parser.add_argument("--exclude-games", nargs="+", default=[],
                        choices=GAME_ORDER,
                        help="Games to exclude (e.g., Seaquest-MinAtar)")

    args = parser.parse_args()

    if args.test_action_mapping:
        test_action_mapping()
        return

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    # Filter configs (check both new and legacy lists)
    all_configs = EXPERIMENT_CONFIGS + EXPERIMENT_CONFIGS_LEGACY
    configs_to_run = [c for c in all_configs if c["name"] in args.configs]

    all_experiment_results = {c["name"]: [] for c in configs_to_run}

    if args.parallel_seeds:
        # Parallel execution: all seeds at once per config
        if args.use_wandb:
            print("Warning: --use-wandb is ignored with --parallel-seeds")
        for experiment_config in configs_to_run:
            results = run_experiment_parallel(
                experiment_config=experiment_config,
                steps_per_game=args.steps_per_game,
                num_cycles=args.num_cycles,
                num_seeds=args.num_seeds,
                num_envs=args.num_envs,
                seed=args.seed,
            )
            all_experiment_results[experiment_config["name"]] = results
    else:
        # Sequential execution: run all configs for each seed before moving to next seed
        # This allows comparing configs at the same seed more easily
        for seed_idx in range(args.num_seeds):
            print(f"\n{'='*70}", flush=True)
            print(f"SEED {seed_idx + 1}/{args.num_seeds}", flush=True)
            print(f"{'='*70}", flush=True)

            for experiment_config in configs_to_run:
                config_name = experiment_config["name"]
                print(f"\n{'#'*60}", flush=True)
                actual_seed = args.seed + seed_idx
                print(f"# Config: {config_name} | Seed: {actual_seed}", flush=True)
                print(f"{'#'*60}", flush=True)

                rng = jax.random.PRNGKey(actual_seed)

                if args.use_wandb:
                    import wandb
                    run_name = f"{config_name}_seed{actual_seed}"
                    wandb.init(
                        project=args.wandb_project,
                        name=run_name,
                        config={
                            "experiment_config": experiment_config,
                            "steps_per_game": args.steps_per_game,
                            "num_cycles": args.num_cycles,
                            "seed": actual_seed,
                            "permute_channels": args.permute_channels,
                            "random_game_order": args.random_game_order,
                            "exclude_games": args.exclude_games,
                        },
                        reinit=True,
                    )

                trainer = ContinualTrainer(
                    config_name=f"{config_name}_seed{actual_seed}",
                    experiment_config=experiment_config,
                    steps_per_game=args.steps_per_game,
                    num_cycles=args.num_cycles,
                    num_envs=args.num_envs,
                    eval_freq=args.eval_freq,
                    checkpoint_dir=checkpoint_dir / config_name,
                    use_wandb=args.use_wandb,
                    wandb_project=args.wandb_project,
                    permute_channels=args.permute_channels,
                    random_game_order=args.random_game_order,
                    seed=actual_seed,
                    exclude_games=args.exclude_games,
                )

                results = trainer.run(rng)
                all_experiment_results[config_name].append(results)

                # Save intermediate results after each seed/config completes
                save_results(all_experiment_results, output_dir)

                if args.use_wandb:
                    import wandb
                    wandb.finish()

    # Save all results
    save_results(all_experiment_results, output_dir)

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    for config_name, results_list in all_experiment_results.items():
        print(f"\n{config_name}:")
        for seed_idx, results in enumerate(results_list):
            final_returns = [r["final_return"] for r in results["per_game_results"]]
            print(f"  Seed {seed_idx}: mean_return={np.mean(final_returns):.1f}")


if __name__ == "__main__":
    main()
