from rejax.algos import DQN, IQN, PPO, PPOOctax, PQN, SAC, TD3, Algorithm
from rejax.networks import CNN, DiscreteCNNPolicy, CNNVNetwork


_algos = {
    "dqn": DQN,
    "iqn": IQN,
    "ppo": PPO,
    "ppo_octax": PPOOctax,
    "pqn": PQN,
    "sac": SAC,
    "td3": TD3,
}


def get_algo(algo: str) -> Algorithm:
    """Get an algorithm class."""
    return _algos[algo]


__all__ = [
    "DQN",
    "IQN",
    "PPO",
    "PPOOctax",
    "PQN",
    "SAC",
    "TD3",
    "get_algo",
    # CNN networks for image observations
    "CNN",
    "DiscreteCNNPolicy",
    "CNNVNetwork",
]
