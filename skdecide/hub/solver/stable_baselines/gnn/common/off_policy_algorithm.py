from typing import Optional, Union

from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv

from .buffers import DictGraphReplayBuffer, GraphReplayBuffer
from .vec_env.dummy_vec_env import wrap_graph_env


class GraphOffPolicyAlgorithm(OffPolicyAlgorithm):
    """Base class for On-Policy algorithms (ex: SAC/TD3) with graph observations."""

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: GymEnv,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        **kwargs,
    ):

        # Use proper default rollout buffer class
        if replay_buffer_class is None:
            if isinstance(env.observation_space, spaces.Graph):
                replay_buffer_class = GraphReplayBuffer
            elif isinstance(env.observation_space, spaces.Dict):
                replay_buffer_class = DictGraphReplayBuffer

        # Use proper VecEnv wrapper for env with Graph spaces
        env = wrap_graph_env(env)
        if env.num_envs > 1:
            raise NotImplementedError(
                "GraphOnPolicyAlgorithm not implemented for real vectorized environment "
                "(ie. with more than 1 wrapped environment)"
            )

        super().__init__(
            policy=policy,
            env=env,
            replay_buffer_class=replay_buffer_class,
            **kwargs,
        )
