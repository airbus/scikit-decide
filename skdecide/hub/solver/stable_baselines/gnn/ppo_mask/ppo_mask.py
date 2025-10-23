from typing import ClassVar, Optional, Union

from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv

from ..common.buffers import (
    MaskableDictGraphRolloutBuffer,
    MaskableGraph2NodeRolloutBuffer,
    MaskableGraphRolloutBuffer,
)
from ..common.on_policy_algorithm import GraphOnPolicyAlgorithm
from ..common.policies import (
    MaskableGNN2NodeActorCriticPolicy,
    MaskableGNNActorCriticPolicy,
    MaskableMultiInputGNNActorCriticPolicy,
)


class MaskableGraphPPO(GraphOnPolicyAlgorithm, MaskablePPO):
    """Proximal Policy Optimization algorithm (PPO) with graph observations and action masking.

    It is meant to be applied to a gymnasium environment
    - whose observation space is
      - either a `gymnasium.spaces.Graph` => you should use policy="GraphInputPolicy",
      - or a `gymnasium.spaces.Dict` with some subspaces being `gymnasium.spaces.Graph`
        => you should use policy="MultiInputPolicy"
    - whose action space is enumerable
    - which is exposing a method `action_masks()` returning a numpy array of same length as
      the number of actions, filled with 0's and 1's corresponding to applicability of the action
      (0: not applicable, 1: applicable)

    The policies will use a GNN to extract features from the observation before being plug to a MLP for prediction.

    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "GraphInputPolicy": MaskableGNNActorCriticPolicy,
        "MultiInputPolicy": MaskableMultiInputGNNActorCriticPolicy,
    }

    support_action_masking = True

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: GymEnv,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        **kwargs,
    ):
        # Use proper default rollout buffer class
        if rollout_buffer_class is None:
            if isinstance(env.observation_space, spaces.Graph):
                rollout_buffer_class = MaskableGraphRolloutBuffer
            elif isinstance(env.observation_space, spaces.Dict):
                rollout_buffer_class = MaskableDictGraphRolloutBuffer

        super().__init__(
            policy=policy,
            env=env,
            rollout_buffer_class=rollout_buffer_class,
            **kwargs,
        )


class MaskableGraph2NodePPO(GraphOnPolicyAlgorithm, MaskablePPO):
    """Proximal Policy Optimization algorithm (PPO) with graph observations, node actions, and action masking.

    It is meant to be applied to a gymnasium environment:
    - such that an observation is a graph represented by a `gymnasium.spaces.GraphInstance`,
    - such that an action is a node of the observation graph, represented by its index (integer between 0 and number of nodes).
    - exposing a method `action_masks()` returning a numpy array of same length as the number of nodes,
       filled with 0's and 1's corresponding to applicability of the action (0: not applicable, 1: applicable)

    So the observation space should be a `gymnasium.spaces.Graph`
    and the action space should be a `gymnasium.spaces.Discrete` even though the actual number of actions is derived at
    runtime from the number of nodes in the current observation.

    The policy will (see `MaskableGNN2NodeActorCriticPolicy` for further details):
    - to predict the value: extract features with a GNN before applying a MLP
    - to predict the action: use another GNN whose nodes embedding will directly be used as logits associated to each node

    NB: The number of nodes (and the size of the action mask) may be variable from an observation to another.

    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "GraphInputPolicy": MaskableGNN2NodeActorCriticPolicy,
    }

    support_action_masking = True

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: GymEnv,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        **kwargs,
    ):
        # Use proper default rollout buffer class
        if rollout_buffer_class is None:
            if isinstance(env.observation_space, spaces.Graph):
                rollout_buffer_class = MaskableGraph2NodeRolloutBuffer
            elif isinstance(env.observation_space, spaces.Dict):
                raise NotImplementedError()

        super().__init__(
            policy=policy,
            env=env,
            rollout_buffer_class=rollout_buffer_class,
            **kwargs,
        )
