from typing import ClassVar

from stable_baselines3 import DQN
from stable_baselines3.common.policies import BasePolicy

from ..common.off_policy_algorithm import GraphOffPolicyAlgorithm
from .policies import GNNDQNPolicy, MultiInputGNNDQNPolicy


class GraphDQN(GraphOffPolicyAlgorithm, DQN):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "GraphInputPolicy": GNNDQNPolicy,
        "MultiInputPolicy": MultiInputGNNDQNPolicy,
    }
