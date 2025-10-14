from collections import OrderedDict
from typing import Callable, List, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs
from stable_baselines3.common.vec_env.util import dict_to_obs

from ..utils import copy_np_array_or_list_of_graph_instances

EnvSubObs = Union[np.ndarray, list[gym.spaces.GraphInstance]]
VecEnvObs = Union[EnvSubObs, dict[str, EnvSubObs], tuple[EnvSubObs, ...]]


class GraphDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        super().__init__(env_fns)
        # replace buffers for graph spaces by lists
        obs_space = self.envs[0].observation_space
        if isinstance(obs_space, gym.spaces.Graph):
            self.buf_obs[None] = [None for _ in range(self.num_envs)]
        elif isinstance(obs_space, gym.spaces.Dict):
            for k, space in obs_space.spaces.items():
                if isinstance(space, gym.spaces.Graph):
                    self.buf_obs[k] = [None for _ in range(self.num_envs)]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))


def copy_obs_dict(obs: dict[str, EnvSubObs]) -> dict[str, EnvSubObs]:
    """
    Deep-copy a dict of numpy arrays.

    :param obs: a dict of numpy arrays.
    :return: a dict of copied numpy arrays.
    """
    assert isinstance(obs, OrderedDict), (
        f"unexpected type for observations '{type(obs)}'"
    )
    return OrderedDict(
        [(k, copy_np_array_or_list_of_graph_instances(v)) for k, v in obs.items()]
    )


def wrap_graph_env(
    env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True
) -> VecEnv:
    """Wrap environment with the appropriate wrappers if needed.

    :param env:
    :param verbose: Verbosity level: 0 for no output, 1 for indicating wrappers used
    :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
    :return: The wrapped environment.
    """
    if not isinstance(env, VecEnv):
        if not is_wrapped(env, Monitor) and monitor_wrapper:
            if verbose >= 1:
                print("Wrapping the env with a `Monitor` wrapper")
            env = Monitor(env)
        if verbose >= 1:
            print("Wrapping the env in a DummyVecEnv.")
        # patch: add dummy shape and dtype to graph obs space to avoid issues
        observation_space = env.observation_space
        if isinstance(observation_space, gym.spaces.Graph):
            observation_space._shape = (0,)
            observation_space.dtype = np.float64
        elif isinstance(observation_space, gym.spaces.Dict):
            for subspace in observation_space.spaces.values():
                if isinstance(subspace, gym.spaces.Graph):
                    subspace._shape = (0,)
                    subspace.dtype = np.float64
        env = GraphDummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
    return env
