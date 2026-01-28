"""
PPO with PopArt normalization for Octax environments.

PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets)
normalizes value targets to have zero mean and unit variance, while adjusting
the final layer weights to preserve unnormalized outputs when statistics change.

References:
- "Learning values across many orders of magnitude" (van Hasselt et al., 2016)
- "Multi-task Deep Reinforcement Learning with PopArt" (Hessel et al., 2019)

Design choices and rationale:

1. **Single unified value head**: Unlike some multi-task approaches that use separate
   value heads per task, PopArt allows a single critic to handle all tasks by
   adaptively normalizing the target scale. This is more parameter-efficient and
   enables better transfer.

2. **Statistics update timing**: We update PopArt statistics AFTER all PPO epochs
   within a train_iteration, not per-minibatch. This provides more stable statistics
   and matches the original paper's approach. The beta parameter controls the EMA
   decay rate.

3. **Weight preservation**: When statistics change, we adjust the critic's final
   layer (W, b) so that: σ_new * (W_new @ h + b_new) + μ_new = σ_old * (W_old @ h + b_old) + μ_old
   This ensures the network's actual value predictions don't jump when stats update.

4. **Normalized storage**: We store normalized values in trajectories and denormalize
   only for GAE computation. This keeps the value network's outputs in a stable range.

5. **AdaMo compatibility**: The architecture separates the PopArt weight adjustment
   from the optimizer update, making it compatible with orthogonal regularization.
   When AdaMo is added, the orthogonal projection happens during the optimizer step,
   while PopArt adjustment happens after (preserving outputs but potentially breaking
   exact orthogonality slightly - this tradeoff is acceptable as PopArt changes are small).
"""
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import OnPolicyMixin
from rejax.algos.ppo import AdvantageMinibatch, Trajectory
from rejax.networks import DiscretePolicy, VNetwork


class PopArtState(struct.PyTreeNode):
    """State for PopArt normalization statistics.

    Design: Uses second moment (nu) rather than variance directly because:
    - EMA of variance is biased (Var(X) != E[X^2] - E[X]^2 for streaming data)
    - Computing sigma = sqrt(nu - mu^2) from moments is the correct approach
    - This matches the original PopArt paper (Equation 7)
    """
    mu: chex.Array  # Running mean of value targets
    nu: chex.Array  # Running second moment E[x^2] (NOT variance!)
    sigma: chex.Array  # Running std = sqrt(nu - mu^2), clipped for stability

    @classmethod
    def create(cls):
        """Initialize with mu=0, sigma=1 (identity transform initially).

        nu=1 ensures sigma=sqrt(1-0)=1 at start.
        """
        return cls(
            mu=jnp.array(0.0),
            nu=jnp.array(1.0),
            sigma=jnp.array(1.0),
        )

    def reset(self):
        """Reset to initial state (identity transform).

        Use at task boundaries in continual learning when resetting optimizer.
        This allows the critic to adapt to new reward scales from scratch.
        """
        return PopArtState.create()


class OctaxAgentPopArt(nn.Module):
    """Shared backbone agent with PopArt-compatible critic.

    Architecture:
    - 3-layer CNN feature extractor (shared between actor and critic)
    - Separate MLP heads for actor and critic
    - Critic outputs NORMALIZED values (denormalization happens externally via PopArt state)

    Design for AdaMo compatibility:
    - Actor and critic have separate MLPs, allowing independent orthogonal constraints
    - The shared CNN features are not part of PopArt adjustment
    - Only the critic's FINAL Dense layer (output layer) is adjusted by PopArt

    PopArt interaction:
    - Network outputs: normalized_value = W @ h + b (where h is hidden representation)
    - Actual value: actual_value = sigma * normalized_value + mu
    - When (mu, sigma) change, we adjust (W, b) to preserve actual_value
    """
    action_dim: int
    conv_features: Tuple[int, int, int] = (32, 64, 64)
    kernel_sizes: Tuple = ((8, 4), (4, 4), (3, 3))
    strides: Tuple = ((4, 2), (2, 2), (1, 1))
    mlp_hidden_sizes: Tuple[int, ...] = (256,)
    activation: Callable = nn.relu
    use_orthogonal_init: bool = False  # Paper uses Flax defaults; set True for CleanRL-style

    def setup(self):
        # Build CNN layers
        conv_layers = []
        for features, kernel, stride in zip(self.conv_features, self.kernel_sizes, self.strides):
            conv_layers.append(nn.Conv(features=features, kernel_size=kernel, strides=stride))
            conv_layers.append(self.activation)
        conv_layers.append(lambda x: x.reshape(x.shape[0], -1))
        self.features = nn.Sequential(conv_layers)

        # Actor head
        self.actor = DiscretePolicy(
            action_dim=self.action_dim,
            hidden_layer_sizes=self.mlp_hidden_sizes,
            activation=self.activation,
            use_orthogonal_init=self.use_orthogonal_init,
        )

        # Critic head - outputs normalized values
        # The final Dense layer's weight/bias will be adjusted by PopArt
        self.critic = VNetwork(
            hidden_layer_sizes=self.mlp_hidden_sizes,
            activation=self.activation,
            use_orthogonal_init=self.use_orthogonal_init,
        )

    def __call__(self, obs, rng, action=None):
        """Forward pass returning action, log_prob, entropy, normalized_value."""
        features = self.features(obs)
        normalized_value = self.critic(features)

        if action is None:
            action, log_prob, entropy = self.actor(features, rng)
            return action, log_prob, entropy, normalized_value
        else:
            log_prob, entropy = self.actor.log_prob_entropy(features, action)
            return action, log_prob, entropy, normalized_value

    def call_critic(self, obs):
        """Get normalized value only (for bootstrapping)."""
        features = self.features(obs)
        return self.critic(features)

    def call_actor(self, obs, rng):
        """Get action only (for evaluation)."""
        features = self.features(obs)
        return self.actor(features, rng)


def update_popart_stats(popart_state: PopArtState, targets: chex.Array, beta: float) -> PopArtState:
    """Update PopArt statistics using exponential moving average.

    Design rationale:
    - Uses EMA (exponential moving average) rather than cumulative mean because:
      1. Non-stationary targets: reward scales can change over training
      2. Memory efficiency: no need to track count
      3. Recency bias: more recent data is more relevant for current scale

    - Beta controls adaptation speed:
      - Small beta (0.0001): slow adaptation, stable but may lag behind scale changes
      - Large beta (0.1): fast adaptation, responsive but potentially noisy
      - Default 0.0001 matches original PopArt paper for stable training

    Args:
        popart_state: Current PopArt state
        targets: Value targets (unnormalized) to update statistics with
        beta: EMA coefficient (higher = more weight on new data)

    Returns:
        Updated PopArt state with new mu, nu, sigma
    """
    # Compute batch statistics (flatten to handle any shape)
    targets_flat = targets.reshape(-1)
    batch_mean = jnp.mean(targets_flat)
    batch_second_moment = jnp.mean(targets_flat ** 2)

    # Handle NaN (e.g., empty batch) - keep old stats
    batch_mean = jnp.where(jnp.isnan(batch_mean), popart_state.mu, batch_mean)
    batch_second_moment = jnp.where(
        jnp.isnan(batch_second_moment), popart_state.nu, batch_second_moment
    )

    # Exponential moving average update (Equation 7 from PopArt paper)
    # μ_t = (1 - β) * μ_{t-1} + β * batch_mean
    # ν_t = (1 - β) * ν_{t-1} + β * batch_second_moment
    new_mu = (1 - beta) * popart_state.mu + beta * batch_mean
    new_nu = (1 - beta) * popart_state.nu + beta * batch_second_moment

    # Compute sigma from moments: σ = sqrt(E[x²] - E[x]²)
    # Clipping ensures numerical stability:
    # - Lower bound (1e-4): prevents division by zero in normalization
    # - Upper bound (1e6): prevents overflow in denormalization
    variance = jnp.maximum(new_nu - new_mu ** 2, 1e-8)
    new_sigma = jnp.sqrt(variance)
    new_sigma = jnp.clip(new_sigma, 1e-4, 1e6)

    return PopArtState(mu=new_mu, nu=new_nu, sigma=new_sigma)


def _find_critic_output_layer_key(critic_params: dict) -> str:
    """Find the key for the critic's output layer (the Dense layer with output_dim=1).

    Design: VNetwork structure is MLP (N hidden Dense layers) + 1 output Dense layer.
    With @nn.compact, layers are named Dense_0, Dense_1, ..., Dense_N.
    The output layer is the last one (Dense_N where N = num_hidden_layers).

    We find it by looking for the Dense layer with kernel shape (..., 1).
    """
    for key in critic_params:
        if key.startswith('Dense_'):
            kernel = critic_params[key].get('kernel')
            if kernel is not None and kernel.shape[-1] == 1:
                return key

    # Fallback: find the highest-numbered Dense layer
    dense_keys = [k for k in critic_params if k.startswith('Dense_')]
    if dense_keys:
        return max(dense_keys, key=lambda k: int(k.split('_')[1]))

    raise ValueError(f"Could not find critic output layer in params: {list(critic_params.keys())}")


def popart_preserve_outputs(
    params: dict,
    old_mu: chex.Array,
    old_sigma: chex.Array,
    new_mu: chex.Array,
    new_sigma: chex.Array,
) -> dict:
    """Adjust critic's final layer to preserve unnormalized outputs.

    Design rationale:
    When PopArt statistics change from (μ_old, σ_old) to (μ_new, σ_new), the
    network would output different unnormalized values. To preserve the actual
    value predictions, we need:

        σ_new * (W_new @ h + b_new) + μ_new = σ_old * (W_old @ h + b_old) + μ_old

    For this to hold for all hidden representations h, we need:
        σ_new * W_new = σ_old * W_old
        σ_new * b_new + μ_new = σ_old * b_old + μ_old

    Solving:
        W_new = (σ_old / σ_new) * W_old
        b_new = (σ_old * b_old + μ_old - μ_new) / σ_new

    This is the "POP" (Preserving Outputs Precisely) part of PopArt.

    AdaMo compatibility note:
    This adjustment happens AFTER the optimizer step, so it doesn't interfere
    with gradient-based orthogonal regularization. However, it may slightly
    break orthogonality of the output layer weights. Since:
    1. PopArt adjustments are typically small (small beta)
    2. The output layer is 1D (scalar output), so orthogonality is trivial
    This interaction is benign in practice.

    Args:
        params: Network parameters (FrozenDict from Flax)
        old_mu, old_sigma: Previous PopArt statistics
        new_mu, new_sigma: Updated PopArt statistics

    Returns:
        New parameters dict with adjusted critic output layer
    """
    # Navigate to critic params
    critic_params = params['params']['critic']

    # Find the output layer dynamically (robust to different MLP depths)
    output_layer_key = _find_critic_output_layer_key(critic_params)

    old_kernel = critic_params[output_layer_key]['kernel']
    old_bias = critic_params[output_layer_key]['bias']

    # Compute scaling factor (avoid division by zero - sigma is already clipped)
    scale = old_sigma / new_sigma

    # Apply weight preservation equations
    new_kernel = scale * old_kernel
    new_bias = (old_sigma * old_bias + old_mu - new_mu) / new_sigma

    # Build new params dict (immutable update pattern for JAX)
    # Using nested dict updates to avoid mutating the original
    new_output_layer = {
        'kernel': new_kernel,
        'bias': new_bias,
    }

    # Preserve any other keys in the output layer (unlikely but safe)
    for key in critic_params[output_layer_key]:
        if key not in new_output_layer:
            new_output_layer[key] = critic_params[output_layer_key][key]

    new_critic_params = {
        **critic_params,
        output_layer_key: new_output_layer,
    }

    new_params = {
        **params,
        'params': {
            **params['params'],
            'critic': new_critic_params,
        }
    }

    return new_params


class PPOOctaxPopArt(OnPolicyMixin, Algorithm):
    """PPO with PopArt normalization for continual/multi-task learning.

    PopArt normalizes value targets and adjusts critic weights to preserve
    outputs when statistics change. This enables learning across tasks with
    vastly different reward scales without task-specific value heads.

    Key features:
    - Single unified value head for all tasks
    - Adaptive target normalization (mu, sigma tracking)
    - Weight preservation when statistics update
    - No reward clipping or manual scaling needed

    Comparison with reward normalization (NormalizeRewardsMixin):
    - Reward normalization: scales rewards BEFORE they enter the value function
    - PopArt: scales value TARGETS during training, adjusts weights to compensate
    - PopArt advantage: preserves the actual learned value function across scale changes
    - PopArt can handle more extreme scale changes (e.g., Atari's 1-point vs 1000-point games)

    Training flow:
    1. collect_trajectories: store NORMALIZED values
    2. calculate_gae: DENORMALIZE values for TD computation (in original scale)
    3. update: train on NORMALIZED targets
    4. update_popart_stats: update (mu, sigma) from unnormalized targets
    5. popart_preserve_outputs: adjust critic weights to preserve value predictions

    AdaMo integration notes:
    - PopArt weight adjustment is a post-hoc linear rescaling of output layer
    - This happens AFTER gradient step, so orthogonal projections are unaffected
    - For deep networks, only the final 1x256 (or similar) layer is rescaled
    - Orthogonality of intermediate layers is preserved exactly
    """
    agent: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=4)
    gae_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar = struct.field(pytree_node=True, default=0.01)

    # PopArt hyperparameters
    # Beta controls EMA speed: smaller = more stable, larger = faster adaptation
    # 0.0001 is conservative (from original paper), 0.001-0.01 for faster adaptation
    popart_beta: chex.Scalar = struct.field(pytree_node=True, default=0.0001)

    def make_act(self, ts):
        """Create action function for evaluation."""
        def act(obs, rng):
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
            "agent": OctaxAgentPopArt(
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
            optax.adam(learning_rate=self.learning_rate),
        )

        agent_ts = TrainState.create(apply_fn=(), params=agent_params, tx=tx)

        return {"agent_ts": agent_ts}

    @register_init
    def initialize_popart_state(self, rng):
        """Initialize PopArt normalization state."""
        return {"popart_state": PopArtState.create()}

    def train_iteration(self, ts):
        """Single training iteration - collect trajectories and update."""
        ts, trajectories = self.collect_trajectories(ts)

        # Bootstrap value for incomplete trajectories (normalized)
        last_val_normalized = self.agent.apply(
            ts.agent_ts.params, ts.last_obs, method="call_critic"
        )
        # Denormalize for GAE computation
        last_val = last_val_normalized * ts.popart_state.sigma + ts.popart_state.mu
        last_val = jnp.where(ts.last_done, 0, last_val)

        # Denormalize trajectory values for GAE
        traj_values_denorm = trajectories.value * ts.popart_state.sigma + ts.popart_state.mu

        advantages, targets = self.calculate_gae(trajectories, traj_values_denorm, last_val)

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)

            # Normalize targets for training
            normalized_targets = (targets - ts.popart_state.mu) / ts.popart_state.sigma

            batch = AdvantageMinibatch(trajectories, advantages, normalized_targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (self.update(ts, mbs), None),
                ts,
                minibatches,
            )
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)

        # Update PopArt statistics after all epochs
        old_mu = ts.popart_state.mu
        old_sigma = ts.popart_state.sigma

        new_popart_state = update_popart_stats(ts.popart_state, targets, self.popart_beta)

        # Preserve outputs by adjusting critic weights
        new_params = popart_preserve_outputs(
            ts.agent_ts.params,
            old_mu, old_sigma,
            new_popart_state.mu, new_popart_state.sigma,
        )

        ts = ts.replace(
            agent_ts=ts.agent_ts.replace(params=new_params),
            popart_state=new_popart_state,
        )

        return ts

    def collect_trajectories(self, ts):
        """Collect trajectories from parallel environments."""
        def env_step(ts, unused):
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, self.num_envs)

            # Forward pass - returns normalized value
            action, log_prob, _, normalized_value = self.agent.apply(
                ts.agent_ts.params, ts.last_obs, rng_action
            )

            # Step environment
            next_obs, env_state, reward, done, _ = self.vmap_step(
                rng_steps, ts.env_state, action, self.env_params
            )

            # Store normalized value in trajectory
            # (will be denormalized for GAE computation)
            transition = Trajectory(
                ts.last_obs, action, log_prob, reward, normalized_value, done
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

    def calculate_gae(self, trajectories, values_denorm, last_val):
        """Compute GAE advantages and value targets in original (denormalized) scale.

        Args:
            trajectories: Collected trajectories (value field contains normalized values)
            values_denorm: Denormalized values from trajectories
            last_val: Denormalized bootstrap value

        Returns:
            advantages: GAE advantages (unnormalized)
            targets: Value targets (unnormalized) = advantages + values
        """
        def get_advantages(carry, inputs):
            advantage, next_value = carry
            reward, value, done = inputs

            delta = reward.squeeze() + self.gamma * next_value * (1 - done) - value
            advantage = delta + self.gamma * self.gae_lambda * (1 - done) * advantage
            return (advantage, value), advantage

        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val), last_val),
            (trajectories.reward, values_denorm, trajectories.done),
            reverse=True,
        )

        # Targets in original scale
        targets = advantages + values_denorm
        return advantages, targets

    def update(self, ts, batch):
        """Combined actor-critic update with normalized value targets."""
        def loss_fn(params):
            _, log_prob, entropy, normalized_value = self.agent.apply(
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

            # Value loss on normalized targets
            # batch.targets is already normalized
            value_pred_clipped = batch.trajectories.value + (
                normalized_value - batch.trajectories.value
            ).clip(-self.clip_eps, self.clip_eps)
            value_losses = jnp.square(normalized_value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            value_loss = self.vf_coef * value_loss

            return actor_loss + value_loss

        grads = jax.grad(loss_fn)(ts.agent_ts.params)
        return ts.replace(agent_ts=ts.agent_ts.apply_gradients(grads=grads))
