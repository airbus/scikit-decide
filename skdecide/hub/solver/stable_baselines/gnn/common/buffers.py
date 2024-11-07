from collections.abc import Generator
from typing import Optional, Union

import numpy as np
import torch as th
import torch_geometric as thg
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import (
    DictRolloutBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from .utils import copy_graph_instance, graph_obs_to_thg_data


class GraphRolloutBuffer(RolloutBuffer):
    observations: Union[list[spaces.GraphInstance], list[list[spaces.GraphInstance]]]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Graph,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = (
            observation_space.node_space.shape + observation_space.edge_space.shape
        )
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
        self.observations = list()  # try to use list to save torch_geometric data

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

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # Same reshape, for actions
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations.append([copy_graph_instance(g) for g in obs])
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            # can only be used when venv = 1
            self.raw_observations: list[spaces.GraphInstance] = list()
            for vec_obs in self.observations:
                self.raw_observations.extend(vec_obs)
            self.observations = self.raw_observations
            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
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
        selected_observations = thg.data.Batch.from_data_list(
            [
                graph_obs_to_thg_data(self.observations[idx], device=self.device)
                for idx in batch_inds
            ]
        )
        data = (
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(
            selected_observations, *tuple(map(self.to_torch, data))
        )
