from collections.abc import Callable, Sequence

import distrax
import jax
import numpy as np
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from jax import numpy as jnp
from jax import lax


def groupsort(x, group_size=2):
    """
    GroupSort activation function. Splits the input into groups of size `group_size`
    and sorts each group in ascending order.

    For group_size=2, uses fast min/max implementation.
    For larger groups, falls back to jnp.sort.
    """
    if group_size == 1:
        return x

    shape = x.shape
    # Ensure the last dimension is divisible by group_size
    if shape[-1] % group_size != 0:
        raise ValueError(
            f"Last dimension of input ({shape[-1]}) must be divisible by group_size ({group_size})"
        )

    # Fast path for group_size=2 using min/max (much faster on GPU)
    if group_size == 2:
        x = x.reshape(shape[:-1] + (-1, 2))
        x0, x1 = x[..., 0], x[..., 1]
        x_min = jnp.minimum(x0, x1)
        x_max = jnp.maximum(x0, x1)
        x = jnp.stack([x_min, x_max], axis=-1)
        return x.reshape(shape)

    # General case: use sort
    x = x.reshape(shape[:-1] + (-1, group_size))
    x = jnp.sort(x, axis=-1)
    return x.reshape(shape)


def groupsort2(x):
    """GroupSort with group_size=2 (1-Lipschitz). Uses fast min/max."""
    return groupsort(x, group_size=2)


def groupsort4(x):
    """GroupSort with group_size=4."""
    return groupsort(x, group_size=4)


def groupsort8(x):
    """GroupSort with group_size=8."""
    return groupsort(x, group_size=8)


def parse_activation_fn(name: str):
    """
    Parse an activation function name string into a callable.

    Supports:
    - Standard Flax activations: 'relu', 'tanh', 'swish', 'gelu', 'silu', etc.
    - GroupSort variants: 'groupsort', 'groupsort2', 'groupsort4', 'groupsort8'

    Args:
        name: Name of the activation function

    Returns:
        Callable activation function
    """
    name_lower = name.lower()

    # GroupSort variants
    if name_lower == "groupsort" or name_lower == "groupsort2":
        return groupsort2
    elif name_lower == "groupsort4":
        return groupsort4
    elif name_lower == "groupsort8":
        return groupsort8

    # Standard Flax activations
    if hasattr(nn, name_lower):
        return getattr(nn, name_lower)
    if hasattr(nn, name):
        return getattr(nn, name)

    raise ValueError(f"Unknown activation function: {name}")


# CNN Feature Extractor and Networks


class CNN(nn.Module):
    """
    CNN feature extractor for MinAtar (10x10 images).

    Architecture:
    - Conv: 16 filters, 3x3 kernel, stride 1, VALID padding -> 8x8x16
    - Flatten -> 1024
    - MLP: configurable depth (default 4x256 for AdaMO research)

    For AdaMO/plasticity research, deeper MLP (4x256) is important.
    For simple baselines, can use shallow MLP (1x128).
    """
    conv_channels: int = 16  # CleanRL MinAtar default
    mlp_hidden_sizes: Sequence[int] = (256, 256, 256, 256)  # 4x256 for AdaMO
    activation: Callable = nn.relu
    kernel_size: int = 3
    use_bias: bool = True
    use_orthogonal_init: bool = True

    @nn.compact
    def __call__(self, x):
        # Single conv layer
        # Input: (batch, 10, 10, C), Output: (batch, 8, 8, 16)
        if self.use_orthogonal_init:
            x = nn.Conv(
                features=self.conv_channels,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding="VALID",  # No padding -> 10-3+1 = 8
                use_bias=self.use_bias,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
        else:
            x = nn.Conv(
                features=self.conv_channels,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding="VALID",
                use_bias=self.use_bias,
            )(x)
        x = self.activation(x)

        # Flatten: 8*8*16 = 1024
        x = x.reshape((x.shape[0], -1))

        # MLP layers (4x256 default for AdaMO research)
        for size in self.mlp_hidden_sizes:
            if self.use_orthogonal_init:
                x = nn.Dense(
                    size,
                    use_bias=self.use_bias,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(x)
            else:
                x = nn.Dense(size, use_bias=self.use_bias)(x)
            x = self.activation(x)

        return x


class DiscreteCNNPolicy(nn.Module):
    """
    Discrete policy with CNN backbone for MinAtar (10x10 images).

    Architecture:
    - Conv: 16 filters, 3x3, VALID -> 8x8x16
    - Flatten -> 1024
    - MLP: configurable depth (default 4x256 for AdaMO research)
    - Actor head: action_dim
    """
    action_dim: int
    conv_channels: int = 16  # CleanRL MinAtar default
    mlp_hidden_sizes: Sequence[int] = (256, 256, 256, 256)  # 4x256 for AdaMO
    activation: Callable = nn.relu
    kernel_size: int = 3
    use_bias: bool = True
    use_orthogonal_init: bool = True  # CleanRL uses orthogonal init

    def setup(self):
        self.features = CNN(
            conv_channels=self.conv_channels,
            mlp_hidden_sizes=self.mlp_hidden_sizes,
            activation=self.activation,
            kernel_size=self.kernel_size,
            use_bias=self.use_bias,
            use_orthogonal_init=self.use_orthogonal_init,
        )
        if self.use_orthogonal_init:
            # Small scale (0.01) for action output - critical for stability
            self.action_logits = nn.Dense(
                self.action_dim,
                use_bias=self.use_bias,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )
        else:
            self.action_logits = nn.Dense(self.action_dim, use_bias=self.use_bias)

    def _action_dist(self, obs):
        features = self.features(obs)
        action_logits = self.action_logits(features)
        return distrax.Categorical(logits=action_logits)

    def __call__(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action), action_dist.entropy()

    def act(self, obs, rng):
        action, _, _ = self(obs, rng)
        return action

    def log_prob_entropy(self, obs, action):
        action_dist = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()

    def action_log_prob(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action)


class CNNVNetwork(nn.Module):
    """
    Value network with CNN backbone for MinAtar (10x10 images).

    Architecture:
    - Conv: 16 filters, 3x3, VALID -> 8x8x16
    - Flatten -> 1024
    - MLP: configurable depth (default 4x256 for AdaMO research)
    - Critic head: 1
    """
    conv_channels: int = 16  # CleanRL MinAtar default
    mlp_hidden_sizes: Sequence[int] = (256, 256, 256, 256)  # 4x256 for AdaMO
    activation: Callable = nn.relu
    kernel_size: int = 3
    use_bias: bool = True
    use_orthogonal_init: bool = True  # CleanRL uses orthogonal init

    @nn.compact
    def __call__(self, obs):
        x = CNN(
            conv_channels=self.conv_channels,
            mlp_hidden_sizes=self.mlp_hidden_sizes,
            activation=self.activation,
            kernel_size=self.kernel_size,
            use_bias=self.use_bias,
            use_orthogonal_init=self.use_orthogonal_init,
        )(obs)

        if self.use_orthogonal_init:
            return nn.Dense(
                1,
                use_bias=self.use_bias,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
            )(x).squeeze(-1)
        return nn.Dense(1, use_bias=self.use_bias)(x).squeeze(-1)


class MLP(nn.Module):
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    use_bias: bool = True
    use_orthogonal_init: bool = False

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for size in self.hidden_layer_sizes:
            if self.use_orthogonal_init:
                x = nn.Dense(
                    size,
                    use_bias=self.use_bias,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(x)
            else:
                x = nn.Dense(size, use_bias=self.use_bias)(x)
            x = self.activation(x)
        return x


# Policy networks


class DiscretePolicy(nn.Module):
    action_dim: int
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    use_bias: bool = True
    use_orthogonal_init: bool = False

    def setup(self):
        self.features = MLP(
            self.hidden_layer_sizes,
            self.activation,
            use_bias=self.use_bias,
            use_orthogonal_init=self.use_orthogonal_init,
        )
        if self.use_orthogonal_init:
            # Small scale (0.01) for action output - critical for stability
            self.action_logits = nn.Dense(
                self.action_dim,
                use_bias=self.use_bias,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )
        else:
            self.action_logits = nn.Dense(self.action_dim, use_bias=self.use_bias)

    def _action_dist(self, obs):
        features = self.features(obs)
        action_logits = self.action_logits(features)
        return distrax.Categorical(logits=action_logits)

    def __call__(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action), action_dist.entropy()

    def act(self, obs, rng):
        action, _, _ = self(obs, rng)
        return action

    def log_prob_entropy(self, obs, action):
        action_dist = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()

    def action_log_prob(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action)


def EpsilonGreedyPolicy(qnet: nn.Module) -> type[nn.Module]:  # noqa:  N802
    class EpsilonGreedyPolicy(qnet):
        def _action_dist(self, obs, epsilon):
            q = self(obs)
            return distrax.EpsilonGreedy(q, epsilon=epsilon)

        def act(self, obs, rng, epsilon=0.05):
            action_dist = self._action_dist(obs, epsilon)
            action = action_dist.sample(seed=rng)
            return action

    return EpsilonGreedyPolicy


class GaussianPolicy(nn.Module):
    action_dim: int
    action_range: tuple[int, int]
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    use_bias: bool = True
    use_orthogonal_init: bool = False

    def setup(self):
        self.features = MLP(
            self.hidden_layer_sizes,
            self.activation,
            use_bias=self.use_bias,
            use_orthogonal_init=self.use_orthogonal_init,
        )
        if self.use_orthogonal_init:
            # Small scale (0.01) for action output - critical for stability
            self.action_mean = nn.Dense(
                self.action_dim,
                use_bias=self.use_bias,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )
        else:
            self.action_mean = nn.Dense(self.action_dim, use_bias=self.use_bias)
        self.action_log_std = self.param(
            "action_log_std", constant(0.0), (self.action_dim,)
        )

    def _action_dist(self, obs):
        features = self.features(obs)
        action_mean = self.action_mean(features)
        return distrax.MultivariateNormalDiag(
            loc=action_mean, scale_diag=jnp.exp(self.action_log_std)
        )

    def __call__(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action), action_dist.entropy()

    def act(self, obs, rng):
        action, _, _ = self(obs, rng)
        return jnp.clip(action, self.action_range[0], self.action_range[1])

    def log_prob_entropy(self, obs, action):
        action_dist = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()

    def action_log_prob(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action)


class SquashedGaussianPolicy(nn.Module):
    action_dim: int
    action_range: tuple[float, float]
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    log_std_range: tuple[float, float]
    use_bias: bool = True

    def setup(self):
        self.features = MLP(self.hidden_layer_sizes, self.activation, use_bias=self.use_bias)
        self.action_mean = nn.Dense(self.action_dim, use_bias=self.use_bias)
        self.action_log_std = nn.Dense(self.action_dim, use_bias=self.use_bias)
        self.bij = distrax.Tanh()

    @property
    def action_loc(self):
        return (self.action_range[1] + self.action_range[0]) / 2

    @property
    def action_scale(self):
        return (self.action_range[1] - self.action_range[0]) / 2

    def _action_dist(self, obs):
        # We have to transform the action manually, since we need to calculate log_probs
        # *before* the tanh transform. Doing it afterwards runs into numerical issues
        # because we cannot invert the tanh for +-1, which can easily be sampled.
        # (e.g. jnp.tanh(8) = 1)
        features = self.features(obs)
        action_mean = self.action_mean(features)
        action_log_std = self.action_log_std(features)
        action_log_std = jnp.clip(
            action_log_std, *self.log_std_range
        )  # TODO: tanh transform?

        return distrax.MultivariateNormalDiag(
            loc=action_mean, scale_diag=jnp.exp(action_log_std)
        )

    def __call__(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        action_log_prob = action_dist.log_prob(action)
        action, log_det_j = self.bij.forward_and_log_det(action)
        action = self.action_loc + action * self.action_scale
        action_log_prob -= log_det_j.sum(axis=-1)
        return action, action_log_prob

    def action_log_prob(self, obs, rng):
        return self(obs, rng)

    def log_prob(self, obs, action, epsilon=1e-6):
        low, high = self.action_range
        action = jnp.clip(action, low + epsilon, high - epsilon)

        action_dist = self._action_dist(obs)
        action = (action - self.action_loc) / self.action_scale
        action, log_det_j = self.bij.inverse_and_log_det(action)
        action_log_prob = action_dist.log_prob(action)
        action_log_prob += log_det_j.sum(axis=-1)
        return action_log_prob

    def act(self, obs, rng):
        action, _ = self(obs, rng)
        return action


class BetaPolicy(nn.Module):
    action_dim: int
    action_range: tuple[float, float]
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    use_bias: bool = True

    @property
    def action_loc(self):
        return self.action_range[0]

    @property
    def action_scale(self):
        return self.action_range[1] - self.action_range[0]

    def __call__(self, obs, rng):
        action, _ = self.action_log_prob(obs, rng)
        return action, *self.log_prob_entropy(obs, action)

    def setup(self):
        self.features = MLP(self.hidden_layer_sizes, self.activation, use_bias=self.use_bias)
        self.alpha = nn.Dense(self.action_dim, use_bias=self.use_bias)
        self.beta = nn.Dense(self.action_dim, use_bias=self.use_bias)

    def _action_dist(self, obs):
        x = self.features(obs)
        alpha = 1 + nn.softplus(self.alpha(x))
        beta = 1 + nn.softplus(self.beta(x))
        return distrax.Beta(alpha, beta)

    def action_log_prob(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        log_prob = action_dist.log_prob(action)
        action = self.action_loc + action * self.action_scale
        return action, log_prob.squeeze(1)

    def act(self, obs, rng):
        action, _ = self.action_log_prob(obs, rng)
        return action

    def log_prob_entropy(self, obs, action):
        action_dist = self._action_dist(obs)
        action = (action - self.action_loc) / self.action_scale
        return action_dist.log_prob(action).squeeze(1), action_dist.entropy()


class DeterministicPolicy(nn.Module):
    action_dim: int
    action_range: tuple[float, float]
    hidden_layer_sizes: tuple[int]
    activation: Callable
    use_bias: bool = True

    @property
    def action_loc(self):
        return (self.action_range[1] + self.action_range[0]) / 2

    @property
    def action_scale(self):
        return (self.action_range[1] - self.action_range[0]) / 2

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_layer_sizes:
            x = nn.Dense(size, use_bias=self.use_bias)(x)
            x = self.activation(x)
        x = nn.Dense(self.action_dim, use_bias=self.use_bias)(x)
        x = jnp.tanh(x)

        action = self.action_loc + x * self.action_scale
        return action

    def act(self, obs, rng):
        action = self(obs)
        return action


# Value networks


class VNetwork(MLP):
    @nn.compact
    def __call__(self, obs):
        x = super().__call__(obs)
        if self.use_orthogonal_init:
            return nn.Dense(
                1,
                use_bias=self.use_bias,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
            )(x).squeeze(1)
        return nn.Dense(1, use_bias=self.use_bias)(x).squeeze(1)


class QNetwork(MLP):
    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs.reshape(obs.shape[0], -1), action], axis=-1)
        x = super().__call__(x)
        return nn.Dense(1, use_bias=self.use_bias)(x).squeeze(1)


class DiscreteQNetwork(MLP):
    # Note: action_dim must have default due to dataclass inheritance rules
    # (parent MLP has use_bias with default)
    action_dim: int = 0

    @nn.compact
    def __call__(self, obs):
        x = super().__call__(obs)
        return nn.Dense(self.action_dim, use_bias=self.use_bias)(x)

    def take(self, obs, action):
        q_values = self(obs)
        return jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(1)


class DuelingQNetwork(nn.Module):
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    action_dim: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, obs):
        x = MLP(self.hidden_layer_sizes, self.activation, use_bias=self.use_bias)(obs)
        value = nn.Dense(1, use_bias=self.use_bias)(x)
        advantage = nn.Dense(self.action_dim, use_bias=self.use_bias)(x)
        advantage = advantage - jnp.mean(advantage, axis=-1, keepdims=True)
        return value + advantage

    def take(self, obs, action):
        q_values = self(obs)
        return jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(1)


class ImplicitQuantileNetwork(nn.Module):
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    action_dim: int
    use_bias: bool = True

    risk_distortion: Callable = lambda tau: tau
    # risk_distortion: Callable = lambda tau: 0.8 * tau
    # Or e.g.: tau ** 0.71 / (tau ** 0.71 + (1 - tau) ** 0.71) ** (1 / 0.71)

    @property
    def embedding_dim(self):
        return self.hidden_layer_sizes[-1]

    @nn.compact
    def __call__(self, obs, rng):
        x = obs.reshape(obs.shape[0], -1)
        psi = MLP(self.hidden_layer_sizes, self.activation, use_bias=self.use_bias)(x)

        tau = distrax.Uniform(0, 1).sample(seed=rng, sample_shape=obs.shape[0])
        tau = self.risk_distortion(tau)
        phi_input = jnp.cos(jnp.pi * jnp.outer(tau, jnp.arange(self.embedding_dim)))
        phi = nn.relu(nn.Dense(self.embedding_dim, use_bias=self.use_bias)(phi_input))

        x = nn.swish(nn.Dense(64, use_bias=self.use_bias)(psi * phi))
        return nn.Dense(self.action_dim, use_bias=self.use_bias)(x), tau

    def q(self, obs, rng, num_samples=32):
        rng = jax.random.split(rng, num_samples)
        zs, _ = jax.vmap(self, in_axes=(None, 0))(obs, rng)
        return zs.mean(axis=0)

    def best_action(self, obs, rng, num_samples=32):
        q = self.q(obs, rng, num_samples)
        best_action = jnp.argmax(q, axis=1)
        return best_action
