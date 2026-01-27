"""
Gymnax wrapper for Octax (CHIP-8 arcade games in JAX).

Uses octax's built-in OctaxGymnaxWrapper directly - NO extra transpose!
The paper uses (4, 32, 64) CHW format directly with their CNN.

Reference: arXiv 2510.01764
"""
import warnings
from copy import copy


def create_octax(game_name: str, **kwargs):
    """Create an Octax environment wrapped for Gymnax compatibility.

    NOTE: Returns observations in (4, 32, 64) CHW format - same as paper!
    Do NOT add extra transpose - paper's CNN handles this format directly.

    Args:
        game_name: Name of the Octax game (e.g., "brix", "pong", "tetris", "tank")
        **kwargs: Additional arguments passed to octax.environments.create_environment

    Returns:
        Tuple of (OctaxGymnaxWrapper, EnvParams)

    Example:
        env, env_params = create_octax("brix")
        obs, state = env.reset(jax.random.PRNGKey(0), env_params)
        # obs.shape = (4, 32, 64) - CHW format, same as paper
    """
    from octax.environments import create_environment
    from octax.wrappers import OctaxGymnaxWrapper

    # Create octax environment
    octax_env, metadata = create_environment(game_name, **kwargs)

    # Wrap with Gymnax-compatible wrapper (built-in to octax)
    # NO extra transpose - paper uses (4, 32, 64) CHW format directly!
    gymnax_env = OctaxGymnaxWrapper(octax_env)

    return gymnax_env, gymnax_env.default_params
