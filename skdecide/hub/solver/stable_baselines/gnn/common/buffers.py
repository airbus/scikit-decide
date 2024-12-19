from collections.abc import Generator
from typing import Optional, TypeVar, Union

import numpy as np
import torch as th
import torch_geometric as thg
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from .utils import copy_graph_instance, graph_obs_to_thg_data


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[tuple[int, ...], dict[str, tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Graph):
        # Will not be used
        return observation_space.node_space.shape + observation_space.edge_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(
            f"{observation_space} observation space is not supported"
        )


class GraphRolloutBuffer(RolloutBuffer):
    """Rollout buffer used in on-policy algorithms like A2C/PPO with graph observations.

    Handles cases where observation space is:
    - a Graph space
    - a Dict space whose subspaces includes a Graph space

    """

    observations: Union[list[spaces.GraphInstance], list[list[spaces.GraphInstance]]]
    tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

    def __init__(
        self,
        buffer_size: int,
        observation_space: Union[spaces.Graph, spaces.Dict],
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False

        self.reset()

    def reset(self) -> None:
        assert isinstance(
            self.observation_space, spaces.Graph
        ), "GraphRolloutBuffer must be used with Graph obs space only"
        super().reset()
        self.observations = list()

    def add(
        self,
        obs: spaces.GraphInstance,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self._add_obs(obs)

        # Same reshape, for actions
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def _add_obs(self, obs: list[spaces.GraphInstance]) -> None:
        self.observations.append([copy_graph_instance(g) for g in obs])

    def _swap_and_flatten_obs(self) -> None:
        self.observations = _swap_and_flatten_nested_list(self.observations)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            self._swap_and_flatten_obs()
            for tensor in self.tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> RolloutBufferSamples:
        observations = self._get_observations_samples(batch_inds)
        data = (
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(observations, *tuple(map(self.to_torch, data)))

    def _get_observations_samples(self, batch_inds: np.ndarray) -> thg.data.Data:
        return self._graphlist_to_torch(self.observations, batch_inds=batch_inds)

    def _graphlist_to_torch(
        self, graph_list: list[spaces.GraphInstance], batch_inds: np.ndarray
    ) -> thg.data.Data:
        return thg.data.Batch.from_data_list(
            [
                graph_obs_to_thg_data(graph_list[idx], device=self.device)
                for idx in batch_inds
            ]
        )


class DictGraphRolloutBuffer(GraphRolloutBuffer, DictRolloutBuffer):

    observations: dict[
        str,
        Union[
            Union[list[spaces.GraphInstance], list[list[spaces.GraphInstance]]],
            np.ndarray,
        ],
    ]
    obs_shape: dict[str, tuple[int, ...]]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.is_observation_subspace_graph: dict[str, bool] = {
            k: isinstance(space, spaces.Graph)
            for k, space in observation_space.spaces.items()
        }
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        super(GraphRolloutBuffer, self).reset()
        for k, is_graph in self.is_observation_subspace_graph.items():
            if is_graph:
                self.observations[k] = list()

    def _add_obs(
        self, obs: dict[str, Union[np.ndarray, list[spaces.GraphInstance]]]
    ) -> None:
        for key in self.observations.keys():
            if self.is_observation_subspace_graph[key]:
                self.observations[key].append(
                    [copy_graph_instance(g) for g in obs[key]]
                )
            else:
                obs_ = np.array(obs[key])
                # Reshape needed when using multiple envs with discrete observations
                # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
                self.observations[key][self.pos] = obs_

    def _swap_and_flatten_obs(self) -> None:
        for key, obs in self.observations.items():
            if self.is_observation_subspace_graph[key]:
                self.observations[key] = _swap_and_flatten_nested_list(obs)
            else:
                self.observations[key] = self.swap_and_flatten(obs)

    def _get_observations_samples(
        self, batch_inds: np.ndarray
    ) -> dict[str, Union[thg.data.Data, th.Tensor]]:
        return {
            k: self._graphlist_to_torch(obs, batch_inds=batch_inds)
            if self.is_observation_subspace_graph[k]
            else self.to_torch(obs[batch_inds])
            for k, obs in self.observations.items()
        }


T = TypeVar("T")


def _swap_and_flatten_nested_list(obs: list[list[T]]) -> list[T]:
    return [x for subobs in obs for x in subobs]