from typing import Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, TensorDict
from stable_baselines3.common.vec_env import VecEnv

from ...common.on_policy_algorithm import SkdecideOnPolicyAlgorithm
from .buffers import DictGraphRolloutBuffer, GraphRolloutBuffer
from .utils import obs_as_tensor
from .vec_env.dummy_vec_env import wrap_graph_env


class GraphOnPolicyAlgorithm(SkdecideOnPolicyAlgorithm):
    """Base class for On-Policy algorithms (ex: A2C/PPO) with graph observations."""

    support_action_masking = False
    """Whether this algorithm supports action masking.

    Useful to share the code between algorithms.

    """

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
                rollout_buffer_class = GraphRolloutBuffer
            elif isinstance(env.observation_space, spaces.Dict):
                rollout_buffer_class = DictGraphRolloutBuffer

        # Check only using dummy VecEnv
        if isinstance(env, VecEnv) and env.num_envs > 1:
            raise NotImplementedError(
                "GraphOnPolicyAlgorithm not implemented for real vectorized environment "
                "(ie. with more than 1 wrapped environment)"
            )

        super().__init__(
            policy=policy,
            env=env,
            rollout_buffer_class=rollout_buffer_class,
            **kwargs,
        )

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
