from typing import ClassVar, Optional, Union

from sb3_contrib import MaskablePPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv

from ..common.buffers import MaskableGraphRolloutBuffer
from ..common.on_policy_algorithm import GraphOnPolicyAlgorithm
from ..common.policies import MaskableGNNActorCriticPolicy


class MaskableGraphPPO(GraphOnPolicyAlgorithm, MaskablePPO):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "GraphInputPolicy": MaskableGNNActorCriticPolicy,
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
            rollout_buffer_class = MaskableGraphRolloutBuffer

        super().__init__(
            policy=policy,
            env=env,
            rollout_buffer_class=rollout_buffer_class,
            **kwargs,
        )
