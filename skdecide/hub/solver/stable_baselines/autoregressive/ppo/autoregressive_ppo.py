from typing import ClassVar, Optional, Union

from sb3_contrib import MaskablePPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv

from skdecide.hub.solver.stable_baselines.autoregressive.common.buffers import (
    ApplicableActionsRolloutBuffer,
)
from skdecide.hub.solver.stable_baselines.autoregressive.common.policies import (
    AutoregressiveActorCriticPolicy,
)


class AutoregressivePPO(MaskablePPO):
    """Proximal Policy Optimization algorithm (PPO) with autoregressive action prediction.

    The action is multidiscrete and each component is predicted by using the observation + the previous components
    (already predicted). A -1 in the components is valid and interpreted as "no component" and we assume that no further
    components have a meaning for this particular action (correspond to actions with a variable number of parameters).

    See the policy `AutoregressiveRestrictedMultiDiscreteActorCriticPolicy` for more details.

    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": AutoregressiveActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: GymEnv,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        **kwargs,
    ):

        # Use proper default rollout buffer class
        if rollout_buffer_class is None:
            rollout_buffer_class = ApplicableActionsRolloutBuffer

        super().__init__(
            policy=policy,
            env=env,
            rollout_buffer_class=rollout_buffer_class,
            **kwargs,
        )
