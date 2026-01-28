"""
PPO with shared CNN backbone for Octax environments.

Matches the octax paper (arXiv 2510.01764) architecture exactly:
- Shared CNN feature extractor for actor and critic
- Combined actor+critic loss with single backward pass
- optax.clip (element-wise) instead of clip_by_global_norm
- Single optimizer for shared network

This is separate from the main PPO class to keep that code clean
and provide exact paper reproduction for benchmarking.
"""
from typing import Callable, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    OnPolicyMixin,
)
from rejax.algos.ppo import AdvantageMinibatch, Trajectory
from rejax.networks import DiscretePolicy, VNetwork


class OctaxAgent(nn.Module):
    """Shared backbone agent for Octax - matches paper exactly.

    Architecture:
    - 3-layer CNN feature extractor (shared between actor and critic)
    - Separate MLP heads for actor and critic

    Input: (batch, 4, 32, 64) from OctaxGymnaxWrapper
    Note: Paper interprets this as NHWC (H=4, W=32, C=64) without transpose.
    This is technically unconventional but matches their exact implementation.
    """
    action_dim: int
    conv_features: Tuple[int, int, int] = (32, 64, 64)
    kernel_sizes: Tuple = ((8, 4), (4, 4), (3, 3))
    strides: Tuple = ((4, 2), (2, 2), (1, 1))
    mlp_hidden_sizes: Tuple[int, ...] = (256,)
    activation: Callable = nn.relu
    use_orthogonal_init: bool = False  # Paper uses Flax defaults

    def setup(self):
        # Build CNN layers
        conv_layers = []
        for features, kernel, stride in zip(self.conv_features, self.kernel_sizes, self.strides):
            conv_layers.append(nn.Conv(features=features, kernel_size=kernel, strides=stride))
            conv_layers.append(self.activation)
        # Flatten
        conv_layers.append(lambda x: x.reshape(x.shape[0], -1))
        self.features = nn.Sequential(conv_layers)

        # Actor and critic heads (share the CNN features)
        self.actor = DiscretePolicy(
            action_dim=self.action_dim,
            hidden_layer_sizes=self.mlp_hidden_sizes,
            activation=self.activation,
            use_orthogonal_init=self.use_orthogonal_init,
        )
        self.critic = VNetwork(
            hidden_layer_sizes=self.mlp_hidden_sizes,
            activation=self.activation,
            use_orthogonal_init=self.use_orthogonal_init,
        )

    def __call__(self, obs, rng, action=None):
        """Forward pass returning action, log_prob, entropy, value.

        Note: Paper does NOT transpose. Input (4, 32, 64) is interpreted as
        NHWC with H=4, W=32, C=64. This is technically "wrong" but works
        empirically and matches their exact architecture.
        """
        features = self.features(obs)

        value = self.critic(features)

        if action is None:
            action, log_prob, entropy = self.actor(features, rng)
            return action, log_prob, entropy, value
        else:
            log_prob, entropy = self.actor.log_prob_entropy(features, action)
            return action, log_prob, entropy, value

    def call_critic(self, obs):
        """Get value only (for bootstrapping)."""
        features = self.features(obs)
        return self.critic(features)

    def call_actor(self, obs, rng):
        """Get action only (for evaluation)."""
        features = self.features(obs)
        return self.actor(features, rng)


class PPOOctax(OnPolicyMixin, NormalizeObservationsMixin, NormalizeRewardsMixin, Algorithm):
    """PPO with shared CNN backbone - matches octax paper architecture.

    Key differences from standard PPO:
    - Single shared network (OctaxAgent) instead of separate actor/critic
    - Combined loss function with single backward pass
    - optax.clip instead of clip_by_global_norm
    - Single optimizer state

    Supports reward normalization for continual learning across games
    with different reward scales.
    """
    agent: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=4)  # Paper default
    gae_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar = struct.field(pytree_node=True, default=0.01)

    def make_act(self, ts):
        """Create action function for evaluation."""
        def act(obs, rng):
            if self.normalize_observations:
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action, _, _, _ = self.agent.apply(ts.agent_ts.params, obs, rng)
            return jnp.squeeze(action)
        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        """Create the shared backbone agent."""
        action_dim = env.action_space(env_params).n

        agent_kwargs = config.pop("agent_kwargs", {})

        # Parse activation
        activation_name = agent_kwargs.pop("activation", "relu")
        if isinstance(activation_name, str):
            activation = getattr(nn, activation_name)
        else:
            activation = activation_name

        # MLP hidden sizes
        mlp_hidden_sizes = agent_kwargs.pop("hidden_layer_sizes",
                          agent_kwargs.pop("mlp_hidden_sizes", (256,)))
        if not isinstance(mlp_hidden_sizes, tuple):
            mlp_hidden_sizes = tuple(mlp_hidden_sizes)

        return {
            "agent": OctaxAgent(
                action_dim=action_dim,
                mlp_hidden_sizes=mlp_hidden_sizes,
                activation=activation,
                **agent_kwargs,
            )
        }

    @register_init
    def initialize_network_params(self, rng):
        """Initialize network with single optimizer."""
        obs_shape = self.env.observation_space(self.env_params).shape
        obs_ph = jnp.empty([1, *obs_shape])

        agent_params = self.agent.init(rng, obs_ph, rng)

        # Paper uses optax.clip (element-wise), not clip_by_global_norm
        tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),  # Default eps=1e-8
        )

        agent_ts = TrainState.create(apply_fn=(), params=agent_params, tx=tx)

        return {"agent_ts": agent_ts}

    def train_iteration(self, ts):
        """Single training iteration - collect trajectories and update."""
        ts, trajectories = self.collect_trajectories(ts)

        # Bootstrap value for incomplete trajectories
        last_val = self.agent.apply(
            ts.agent_ts.params, ts.last_obs, method="call_critic"
        )
        last_val = jnp.where(ts.last_done, 0, last_val)

        advantages, targets = self.calculate_gae(trajectories, last_val)

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (self.update(ts, mbs), None),
                ts,
                minibatches,
            )
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts

    def collect_trajectories(self, ts):
        """Collect trajectories from parallel environments."""
        def env_step(ts, unused):
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, self.num_envs)

            # Forward pass through shared network
            action, log_prob, _, value = self.agent.apply(
                ts.agent_ts.params, ts.last_obs, rng_action
            )

            # Step environment
            next_obs, env_state, reward, done, _ = self.vmap_step(
                rng_steps, ts.env_state, action, self.env_params
            )

            # Optional normalization
            if self.normalize_observations:
                obs_rms_state, next_obs = self.update_and_normalize_obs(
                    ts.obs_rms_state, next_obs
                )
                ts = ts.replace(obs_rms_state=obs_rms_state)

            if self.normalize_rewards:
                rew_rms_state, reward = self.update_and_normalize_rew(
                    ts.rew_rms_state, reward, done
                )
                ts = ts.replace(rew_rms_state=rew_rms_state)

            transition = Trajectory(
                ts.last_obs, action, log_prob, reward, value, done
            )
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=ts.global_step + self.num_envs,
            )
            return ts, transition

        ts, trajectories = jax.lax.scan(env_step, ts, None, self.num_steps)
        return ts, trajectories

    def calculate_gae(self, trajectories, last_val):
        """Compute GAE advantages and value targets."""
        def get_advantages(carry, transition):
            advantage, next_value = carry
            delta = (
                transition.reward.squeeze()
                + self.gamma * next_value * (1 - transition.done)
                - transition.value
            )
            advantage = (
                delta + self.gamma * self.gae_lambda * (1 - transition.done) * advantage
            )
            return (advantage, transition.value), advantage

        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val), last_val),
            trajectories,
            reverse=True,
        )
        return advantages, advantages + trajectories.value

    def update(self, ts, batch):
        """Combined actor-critic update with single backward pass."""
        def loss_fn(params):
            _, log_prob, entropy, value = self.agent.apply(
                params, batch.trajectories.obs, None, batch.trajectories.action
            )

            # Actor loss
            entropy = entropy.mean()
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std() + 1e-8
            )
            clipped_ratio = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            pi_loss1 = ratio * advantages
            pi_loss2 = clipped_ratio * advantages
            pi_loss = -jnp.minimum(pi_loss1, pi_loss2).mean()
            actor_loss = pi_loss - self.ent_coef * entropy

            # Value loss (clipped)
            value_pred_clipped = batch.trajectories.value + (
                value - batch.trajectories.value
            ).clip(-self.clip_eps, self.clip_eps)
            value_losses = jnp.square(value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            value_loss = self.vf_coef * value_loss

            return actor_loss + value_loss

        grads = jax.grad(loss_fn)(ts.agent_ts.params)
        return ts.replace(agent_ts=ts.agent_ts.apply_gradients(grads=grads))
