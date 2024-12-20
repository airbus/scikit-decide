from typing import ClassVar

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

from ..common.on_policy_algorithm import GraphOnPolicyAlgorithm
from ..common.policies import GNNActorCriticPolicy, MultiInputGNNActorCriticPolicy


class GraphPPO(GraphOnPolicyAlgorithm, PPO):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "GraphInputPolicy": GNNActorCriticPolicy,
        "MultiInputPolicy": MultiInputGNNActorCriticPolicy,
    }
