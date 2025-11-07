from typing import ClassVar

from stable_baselines3 import A2C
from stable_baselines3.common.policies import BasePolicy

from ..common.on_policy_algorithm import GraphOnPolicyAlgorithm
from ..common.policies import GNNActorCriticPolicy, MultiInputGNNActorCriticPolicy


class GraphA2C(GraphOnPolicyAlgorithm, A2C):
    """Advantage Actor Critic (A2C) with graph observations.

    It is meant to be applied to a gymnasium environment whose observation space is
    - either a `gymnasium.spaces.Graph` => you should use policy="GraphInputPolicy",
    - or a `gymnasium.spaces.Dict` with some subspaces being `gymnasium.spaces.Graph`
      => you should use policy="MultiInputPolicy"

    The policies will use a GNN to extract features from the observation before being plug to an MLP for prediction.

    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "GraphInputPolicy": GNNActorCriticPolicy,
        "MultiInputPolicy": MultiInputGNNActorCriticPolicy,
    }
