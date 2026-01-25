from typing import Callable, Optional

import chex
import gymnax
import jax
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    OnPolicyMixin,
)
from rejax.networks import (
    DiscretePolicy,
    GaussianPolicy,
    VNetwork,
    parse_activation_fn,
    DiscreteCNNPolicy,
    CNNVNetwork,
)
from rejax.regularization import compute_gram_regularization_loss, apply_ortho_update


class Trajectory(struct.PyTreeNode):
    obs: chex.Array
    action: chex.Array
    log_prob: chex.Array
    reward: chex.Array
    value: chex.Array
    done: chex.Array


class AdvantageMinibatch(struct.PyTreeNode):
    trajectories: Trajectory
    advantages: chex.Array
    targets: chex.Array


class PPO(OnPolicyMixin, NormalizeObservationsMixin, NormalizeRewardsMixin, Algorithm):
    actor: nn.Module = struct.field(pytree_node=False, default=None)
    critic: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=8)
    gae_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar = struct.field(pytree_node=True, default=0.01)

    # Learning rate annealing (like PureJaxRL)
    anneal_lr: bool = struct.field(pytree_node=False, default=False)

    # Custom LR schedule function (takes precedence over anneal_lr)
    lr_schedule_fn: Optional[Callable[[int], float]] = struct.field(pytree_node=False, default=None)

    # Orthonormalization settings
    ortho_mode: Optional[str] = struct.field(pytree_node=False, default=None)  # None, "loss", "optimizer"
    ortho_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.2)  # loss mode coefficient
    ortho_coeff: chex.Scalar = struct.field(pytree_node=True, default=1e-3)  # optimizer mode coefficient
    ortho_exclude_output: bool = struct.field(pytree_node=False, default=True)

    def make_act(self, ts):
        def act(obs, rng):
            if getattr(self, "normalize_observations", False):
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = self.actor.apply(ts.actor_ts.params, obs, rng, method="act")
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        action_space = env.action_space(env_params)
        obs_space = env.observation_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)

        agent_kwargs = config.pop("agent_kwargs", {})

        # Network type: "auto", "mlp", or "cnn"
        network_type = agent_kwargs.pop("network_type", "auto")

        # Auto-detect: use CNN if observation is 3D (H, W, C)
        if network_type == "auto":
            obs_shape = obs_space.shape
            network_type = "cnn" if len(obs_shape) == 3 else "mlp"

        # Parse activation function
        activation = agent_kwargs.pop("activation", "relu" if network_type == "cnn" else "swish")
        activation_fn = parse_activation_fn(activation)

        if network_type == "cnn":
            # CNN architecture for MinAtar (10x10 images)
            # Conv(16, k=3, VALID) -> flatten -> MLP (default 4x256 for AdaMO research)
            conv_channels = agent_kwargs.pop("conv_channels", 16)
            # Support both mlp_hidden_sizes (new) and mlp_hidden_size (old)
            mlp_hidden_sizes = agent_kwargs.pop("mlp_hidden_sizes", None)
            mlp_hidden_size = agent_kwargs.pop("mlp_hidden_size", None)
            if mlp_hidden_sizes is None:
                if mlp_hidden_size is not None:
                    # Single int -> convert to tuple
                    mlp_hidden_sizes = (mlp_hidden_size,)
                else:
                    # Default: 4x256 for AdaMO research
                    mlp_hidden_sizes = (256, 256, 256, 256)
            kernel_size = agent_kwargs.pop("kernel_size", 3)

            # Remove deprecated params if present
            agent_kwargs.pop("use_avgpool", None)
            agent_kwargs.pop("pool_size", None)
            agent_kwargs.pop("hidden_layer_sizes", None)

            cnn_kwargs = {
                "conv_channels": conv_channels,
                "mlp_hidden_sizes": tuple(mlp_hidden_sizes),
                "activation": activation_fn,
                "kernel_size": kernel_size,
                **agent_kwargs,  # pass through use_bias, use_orthogonal_init, etc.
            }

            if not discrete:
                raise NotImplementedError("CNN with continuous actions not yet supported")

            actor = DiscreteCNNPolicy(action_space.n, **cnn_kwargs)
            critic = CNNVNetwork(**cnn_kwargs)
        else:
            # MLP architecture (original behavior)
            hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
            # Remove CNN-specific args if present
            agent_kwargs.pop("conv_channels", None)
            agent_kwargs.pop("mlp_hidden_sizes", None)
            agent_kwargs.pop("kernel_size", None)

            mlp_kwargs = {
                "hidden_layer_sizes": tuple(hidden_layer_sizes),
                "activation": activation_fn,
                **agent_kwargs,
            }

            if discrete:
                actor = DiscretePolicy(action_space.n, **mlp_kwargs)
            else:
                actor = GaussianPolicy(
                    np.prod(action_space.shape),
                    (action_space.low, action_space.high),
                    **mlp_kwargs,
                )

            critic = VNetwork(**mlp_kwargs)

        return {"actor": actor, "critic": critic}

    @register_init
    def initialize_network_params(self, rng):
        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        obs_ph = jnp.empty([1, *self.env.observation_space(self.env_params).shape])

        actor_params = self.actor.init(rng_actor, obs_ph, rng_actor)
        critic_params = self.critic.init(rng_critic, obs_ph)

        # Learning rate: priority is lr_schedule_fn > anneal_lr > constant
        # Use optax.linear_schedule to avoid capturing self.learning_rate (a pytree leaf/tracer) in a closure
        if self.lr_schedule_fn is not None:
            # Custom schedule function takes precedence
            learning_rate = self.lr_schedule_fn
        elif self.anneal_lr:
            # Number of updates = total_timesteps / (num_envs * num_steps)
            # Each update has num_epochs * num_minibatches gradient steps
            num_updates = self.total_timesteps // (self.num_envs * self.num_steps)
            total_steps = num_updates * self.num_epochs * self.num_minibatches
            learning_rate = optax.linear_schedule(
                init_value=self.learning_rate,
                end_value=0.0,
                transition_steps=total_steps,
            )
        else:
            learning_rate = self.learning_rate

        tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=learning_rate, eps=1e-5),
        )
        actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)
        critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)
        return {"actor_ts": actor_ts, "critic_ts": critic_ts}

    def train_iteration(self, ts):
        """Train iteration (fast path, discards metrics)."""
        ts, trajectories = self.collect_trajectories(ts)

        last_val = self.critic.apply(ts.critic_ts.params, ts.last_obs)
        last_val = jnp.where(ts.last_done, 0, last_val)
        advantages, targets = self.calculate_gae(trajectories, last_val)

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)

            def update_minibatch(ts, mbs):
                ts, _ = self.update(ts, mbs)  # Discard metrics for speed
                return ts, None

            ts, _ = jax.lax.scan(update_minibatch, ts, minibatches)
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts

    def train_iteration_with_metrics(self, ts):
        """Train iteration that returns metrics dict. Use for research/logging."""
        ts, trajectories = self.collect_trajectories(ts)

        last_val = self.critic.apply(ts.critic_ts.params, ts.last_obs)
        last_val = jnp.where(ts.last_done, 0, last_val)
        advantages, targets = self.calculate_gae(trajectories, last_val)

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)

            def update_minibatch(ts, mbs):
                ts, metrics = self.update(ts, mbs)
                return ts, metrics

            ts, epoch_metrics = jax.lax.scan(update_minibatch, ts, minibatches)
            return ts, epoch_metrics

        ts, all_epoch_metrics = jax.lax.scan(update_epoch, ts, None, self.num_epochs)

        # all_epoch_metrics: dict of arrays (num_epochs, num_minibatches)
        # Flatten and mean across epochs and minibatches
        final_metrics = jax.tree.map(lambda x: x.mean(), all_epoch_metrics)

        # Compute Gram deviation after all updates
        _, actor_gram_metrics = compute_gram_regularization_loss(
            ts.actor_ts.params, lambda_coeff=1.0, exclude_output=True
        )
        _, critic_gram_metrics = compute_gram_regularization_loss(
            ts.critic_ts.params, lambda_coeff=1.0, exclude_output=True
        )
        final_metrics["gram/actor"] = actor_gram_metrics["ortho/total_loss"]
        final_metrics["gram/critic"] = critic_gram_metrics["ortho/total_loss"]

        # Compute current learning rate
        final_metrics["train/learning_rate"] = self._get_current_lr(ts.actor_ts.step)

        return ts, final_metrics

    def _get_current_lr(self, step):
        """Compute current learning rate based on step count and schedule."""
        if self.lr_schedule_fn is not None:
            # Custom schedule
            return self.lr_schedule_fn(step)
        elif self.anneal_lr:
            # Linear annealing from learning_rate to 0
            num_updates = self.total_timesteps // (self.num_envs * self.num_steps)
            total_steps = num_updates * self.num_epochs * self.num_minibatches
            # Linear interpolation: lr * (1 - step/total)
            progress = jnp.minimum(step / total_steps, 1.0)
            return self.learning_rate * (1.0 - progress)
        else:
            # Constant learning rate
            return self.learning_rate

    def collect_trajectories(self, ts):
        def env_step(ts, unused):
            # Get keys for sampling action and stepping environment
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, self.num_envs)

            # Sample action
            unclipped_action, log_prob = self.actor.apply(
                ts.actor_ts.params, ts.last_obs, rng_action, method="action_log_prob"
            )
            value = self.critic.apply(ts.critic_ts.params, ts.last_obs)

            # Clip action
            if self.discrete:
                action = unclipped_action
            else:
                low = self.env.action_space(self.env_params).low
                high = self.env.action_space(self.env_params).high
                action = jnp.clip(unclipped_action, low, high)

            # Step environment
            t = self.vmap_step(rng_steps, ts.env_state, action, self.env_params)
            next_obs, env_state, reward, done, _ = t

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

            # Return updated runner state and transition
            transition = Trajectory(
                ts.last_obs, unclipped_action, log_prob, reward, value, done
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
        def get_advantages(advantage_and_next_value, transition):
            advantage, next_value = advantage_and_next_value
            delta = (
                transition.reward.squeeze()  # For gymnax envs that return shape (1, )
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

    def update_actor(self, ts, batch):
        def actor_loss_fn(params):
            log_prob, entropy = self.actor.apply(
                params,
                batch.trajectories.obs,
                batch.trajectories.action,
                method="log_prob_entropy",
            )
            entropy = entropy.mean()

            # Calculate actor loss
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std() + 1e-8
            )
            clipped_ratio = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            pi_loss1 = ratio * advantages
            pi_loss2 = clipped_ratio * advantages
            pi_loss = -jnp.minimum(pi_loss1, pi_loss2).mean()

            # Compute diagnostics
            approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
            clip_fraction = (jnp.abs(ratio - 1) > self.clip_eps).mean()

            total_loss = pi_loss - self.ent_coef * entropy

            # Add ortho loss if in loss mode
            if self.ortho_mode == "loss":
                ortho_loss, _ = compute_gram_regularization_loss(
                    params,
                    lambda_coeff=self.ortho_lambda,
                    exclude_output=self.ortho_exclude_output,
                )
                total_loss = total_loss + ortho_loss

            return total_loss, (pi_loss, entropy, approx_kl, clip_fraction)

        grads, (pi_loss, entropy, approx_kl, clip_fraction) = jax.grad(
            actor_loss_fn, has_aux=True
        )(ts.actor_ts.params)
        new_actor_ts = ts.actor_ts.apply_gradients(grads=grads)

        # Apply ortho update if in optimizer mode
        if self.ortho_mode == "optimizer":
            new_params = apply_ortho_update(
                new_actor_ts.params,
                lr=self.learning_rate,
                ortho_coeff=self.ortho_coeff,
                exclude_output=self.ortho_exclude_output,
            )
            new_actor_ts = new_actor_ts.replace(params=new_params)

        metrics = {
            "loss/policy": pi_loss,
            "loss/entropy": entropy,
            "ppo/approx_kl": approx_kl,
            "ppo/clip_fraction": clip_fraction,
        }
        return ts.replace(actor_ts=new_actor_ts), metrics

    def update_critic(self, ts, batch):
        def critic_loss_fn(params):
            value = self.critic.apply(params, batch.trajectories.obs)
            value_pred_clipped = batch.trajectories.value + (
                value - batch.trajectories.value
            ).clip(-self.clip_eps, self.clip_eps)
            value_losses = jnp.square(value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            total_loss = self.vf_coef * value_loss

            # Add ortho loss if in loss mode
            if self.ortho_mode == "loss":
                ortho_loss, _ = compute_gram_regularization_loss(
                    params,
                    lambda_coeff=self.ortho_lambda,
                    exclude_output=self.ortho_exclude_output,
                )
                total_loss = total_loss + ortho_loss

            return total_loss, value_loss

        grads, value_loss = jax.grad(critic_loss_fn, has_aux=True)(ts.critic_ts.params)
        new_critic_ts = ts.critic_ts.apply_gradients(grads=grads)

        # Apply ortho update if in optimizer mode
        if self.ortho_mode == "optimizer":
            new_params = apply_ortho_update(
                new_critic_ts.params,
                lr=self.learning_rate,
                ortho_coeff=self.ortho_coeff,
                exclude_output=self.ortho_exclude_output,
            )
            new_critic_ts = new_critic_ts.replace(params=new_params)

        metrics = {"loss/value": value_loss}
        return ts.replace(critic_ts=new_critic_ts), metrics

    def update(self, ts, batch):
        ts, actor_metrics = self.update_actor(ts, batch)
        ts, critic_metrics = self.update_critic(ts, batch)
        metrics = {**actor_metrics, **critic_metrics}
        return ts, metrics
