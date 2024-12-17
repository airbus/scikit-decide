from typing import ClassVar

from stable_baselines3 import A2C
from stable_baselines3.common.policies import BasePolicy

from ..common.on_policy_algorithm import GraphOnPolicyAlgorithm
from ..common.policies import GNNActorCriticPolicy, MultiInputGNNActorCriticPolicy


class GraphA2C(GraphOnPolicyAlgorithm, A2C):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "GraphInputPolicy": GNNActorCriticPolicy,
        "MultiInputPolicy": MultiInputGNNActorCriticPolicy,
    }
