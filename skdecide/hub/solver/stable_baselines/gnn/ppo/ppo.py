from typing import ClassVar

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

from ..common.on_policy_algorithm import GraphOnPolicyAlgorithm
from ..common.policies import (
    GNN2NodeActorCriticPolicy,
    GNNActorCriticPolicy,
    MultiInputGNNActorCriticPolicy,
)


class GraphPPO(GraphOnPolicyAlgorithm, PPO):
    """Proximal Policy Optimization algorithm (PPO) with graph observations.

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


class Graph2NodePPO(GraphOnPolicyAlgorithm, PPO):
    """Proximal Policy Optimization algorithm (PPO) with graph observations and node actions.

    It is meant to be applied to a gymnasium environment such that:
    - an observation is a graph represented by a `gymnasium.spaces.GraphInstance`,
    - an action is a node of the observation graph, represented by its index (integer between 0 and number of nodes).

    So the observation space should be a `gymnasium.spaces.Graph`
    and the action space should be a `gymnasium.spaces.Discrete` even though the actual number of actions is derived at
    runtime from the number of nodes in the current observation.

    The policy will (see `GNN2NodeActorCriticPolicy` for further details):
    - to predict the value: extract features with a GNN before applying a MLP
    - to predict the action: use another GNN whose nodes embedding will directly be used as logits associated to each node

    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "GraphInputPolicy": GNN2NodeActorCriticPolicy,
    }
