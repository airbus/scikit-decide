from typing import Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, TensorDict
from stable_baselines3.common.vec_env import VecEnv

from ...common.on_policy_algorithm import SkdecideOnPolicyAlgorithm
from ...gnn.common.utils import obs_as_tensor
from ...gnn.common.vec_env.dummy_vec_env import wrap_graph_env
from .buffers import ApplicableActionsGraphRolloutBuffer, ApplicableActionsRolloutBuffer


class ApplicableActionsOnPolicyAlgorithm(SkdecideOnPolicyAlgorithm):
    """Base class for On-Policy algorithms (ex: A2C/PPO) using list of applicable actions."""

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
                rollout_buffer_class = ApplicableActionsGraphRolloutBuffer
            else:
                rollout_buffer_class = ApplicableActionsRolloutBuffer

        # Check only using dummy VecEnv
        if isinstance(env, VecEnv) and env.num_envs > 1:
            raise NotImplementedError(
                "ApplicableActionsOnPolicyAlgorithm not implemented for real vectorized environment "
                "(ie. with more than 1 wrapped environment)"
            )

        super().__init__(
            policy=policy,
            env=env,
            rollout_buffer_class=rollout_buffer_class,
            **kwargs,
        )


class ApplicableActionsGraphOnPolicyAlgorithm(ApplicableActionsOnPolicyAlgorithm):
    """Base class for On-Policy algorithms (ex: A2C/PPO) using list of applicable actions and graph obs."""

    @staticmethod
    def _wrap_env(
        env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True
    ) -> VecEnv:
        return wrap_graph_env(env, verbose=verbose, monitor_wrapper=monitor_wrapper)

    @staticmethod
    def obs_as_tensor(
        obs: Union[np.ndarray, dict[str, np.ndarray]], device: th.device
    ) -> Union[th.Tensor, TensorDict]:
        return obs_as_tensor(obs=obs, device=device)
