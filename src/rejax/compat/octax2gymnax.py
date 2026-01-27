"""
Gymnax wrapper for Octax (CHIP-8 arcade games in JAX).

Uses octax's built-in OctaxGymnaxWrapper and adds a transpose to convert
observations from CHW (4, 32, 64) to HWC (64, 32, 4) format for CNN compatibility.

Reference: arXiv 2510.01764
"""
import warnings
from copy import copy

from jax import numpy as jnp


class HWCObsWrapper:
    """Wrapper that transposes observations from CHW to HWC format.

    Octax's OctaxGymnaxWrapper outputs (frame_skip, H, W) = (4, 32, 64).
    We transpose to (W, H, C) = (64, 32, 4) to match the paper's CNN architecture.

    Note: The paper treats observations as (64 height, 32 width) for the CNN,
    which is the transposed view of the CHIP-8 display (64 wide, 32 tall).
    """

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def default_params(self):
        return self._env.default_params

    def reset(self, key, params):
        obs, state = self._env.reset(key, params)
        # Transpose (4, 32, 64) -> (64, 32, 4) to match paper's CNN input
        obs = jnp.transpose(obs, (2, 1, 0)).astype(jnp.float32)
        return obs, state

    def step(self, key, state, action, params):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        # Transpose (4, 32, 64) -> (64, 32, 4) to match paper's CNN input
        obs = jnp.transpose(obs, (2, 1, 0)).astype(jnp.float32)
        return obs, state, reward, done, info

    def observation_space(self, params):
        from gymnax.environments import spaces
        # (64, 32, 4) to match paper's "(4, 64, 32)" notation with "64 height, 32 width"
        return spaces.Box(low=0.0, high=1.0, shape=(64, 32, 4))

    def action_space(self, params):
        return self._env.action_space(params)

    @property
    def num_actions(self) -> int:
        return self._env.num_actions

    @property
    def name(self) -> str:
        return self._env.name

    def __deepcopy__(self, memo):
        warnings.warn(
            f"Trying to deepcopy {type(self).__name__}, which contains an octax env. "
            "Octax envs may throw an error when deepcopying, so a shallow copy is returned.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return copy(self)


def create_octax(game_name: str, **kwargs):
    """Create an Octax environment wrapped for Gymnax compatibility with HWC observations.

    Args:
        game_name: Name of the Octax game (e.g., "brix", "pong", "tetris", "tank")
        **kwargs: Additional arguments passed to octax.environments.create_environment

    Returns:
        Tuple of (HWCObsWrapper, EnvParams)

    Example:
        env, env_params = create_octax("brix")
        obs, state = env.reset(jax.random.PRNGKey(0), env_params)
        # obs.shape = (64, 32, 4) - matches paper's CNN input format
    """
    from octax.environments import create_environment
    from octax.wrappers import OctaxGymnaxWrapper

    # Create octax environment
    octax_env, metadata = create_environment(game_name, **kwargs)

    # Wrap with Gymnax-compatible wrapper (built-in to octax)
    gymnax_env = OctaxGymnaxWrapper(octax_env)

    # Add HWC transpose wrapper
    hwc_env = HWCObsWrapper(gymnax_env)

    return hwc_env, gymnax_env.default_params
