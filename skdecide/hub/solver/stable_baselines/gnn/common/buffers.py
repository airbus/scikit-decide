from typing import Optional, Union

import numpy as np
import torch as th
import torch_geometric as thg
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import (
    MaskableDictRolloutBuffer,
    MaskableRolloutBuffer,
)
from stable_baselines3.common.buffers import (
    BaseBuffer,
    DictReplayBuffer,
    DictRolloutBuffer,
)
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
from torch.nn.utils.rnn import pad_sequence

from ...common.buffers import (
    MaskableScikitDecideRolloutBufferMixin,
    ScikitDecideReplayBuffer,
    ScikitDecideRolloutBuffer,
    swap_and_flatten_nested_list,
)
from .preprocessing import get_obs_shape
from .utils import copy_graph_instance, graph_instance_to_thg_data


class GraphBaseBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: Union[spaces.Graph, spaces.Dict],
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
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

    def _graphlist_to_torch(
        self, graph_list: list[spaces.GraphInstance], batch_inds: np.ndarray
    ) -> thg.data.Data:
        return thg.data.Batch.from_data_list(
            [
                graph_instance_to_thg_data(graph_list[idx], device=self.device)
                for idx in batch_inds
            ]
        )


class GraphRolloutBuffer(ScikitDecideRolloutBuffer, GraphBaseBuffer):
    """Rollout buffer used in on-policy algorithms like A2C/PPO with graph observations.

    Handles cases where observation space is:
    - a Graph space
    - a Dict space whose subspaces includes a Graph space

    """

    observations: Union[list[spaces.GraphInstance], list[list[spaces.GraphInstance]]]

    def reset(self) -> None:
        assert isinstance(self.observation_space, spaces.Graph), (
            "GraphRolloutBuffer must be used with Graph obs space only"
        )
        super().reset()
        self.observations = list()

    def _add_obs(self, obs: list[spaces.GraphInstance]) -> None:
        self.observations.append([copy_graph_instance(g) for g in obs])

    def _swap_and_flatten_obs(self) -> None:
        self.observations = swap_and_flatten_nested_list(self.observations)

    def _get_observations_samples(self, batch_inds: np.ndarray) -> thg.data.Data:
        return self._graphlist_to_torch(self.observations, batch_inds=batch_inds)


class DictGraphRolloutBuffer(GraphRolloutBuffer, DictRolloutBuffer):
    observations: dict[
        str,
        Union[
            Union[list[spaces.GraphInstance], list[list[spaces.GraphInstance]]],
            np.ndarray,
        ],
    ]

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
                self.observations[key] = swap_and_flatten_nested_list(obs)
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


class MaskableGraph2NodeRolloutBufferMixin(MaskableScikitDecideRolloutBufferMixin):
    action_masks: list[np.ndarray]

    def reset(self):
        super().reset()
        self.action_masks = list()

    def _add_action_masks(self, action_masks: Optional[np.ndarray] = None):
        if action_masks is None:
            raise NotImplementedError()
        self.action_masks.append(action_masks.reshape(self.n_envs, -1))

    def _swap_and_flatten_action_masks(self) -> None:
        if self.n_envs > 1:
            raise NotImplementedError()
        else:
            self.action_masks = [a.flatten() for a in self.action_masks]

    def _get_action_masks_samples(self, batch_inds: np.ndarray) -> th.Tensor:
        return pad_sequence(
            [self.to_torch(self.action_masks[idx]) for idx in batch_inds],
            batch_first=True,
        )


class MaskableGraphRolloutBuffer(
    MaskableScikitDecideRolloutBufferMixin, GraphRolloutBuffer, MaskableRolloutBuffer
): ...


class MaskableDictGraphRolloutBuffer(
    MaskableScikitDecideRolloutBufferMixin,
    DictGraphRolloutBuffer,
    MaskableDictRolloutBuffer,
): ...


class MaskableGraph2NodeRolloutBuffer(
    MaskableGraph2NodeRolloutBufferMixin, GraphRolloutBuffer, MaskableRolloutBuffer
): ...


class GraphReplayBuffer(ScikitDecideReplayBuffer, GraphBaseBuffer):
    observations: list[spaces.GraphInstance]
    next_observations: list[spaces.GraphInstance]

    def __init__(
        self,
        buffer_size: int,
        observation_space: Union[spaces.Graph, spaces.Dict],
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        if optimize_memory_usage:
            raise NotImplementedError(
                "No memory usage optimization implemented for GraphReplayBuffer."
            )
        if n_envs > 1:
            raise NotImplementedError(
                "No multiple vectorized environements implemented for GraphReplayBuffer."
            )

        self._init_observations()

    def _init_observations(self):
        self.observations = list()
        self.next_observations = list()

    def _add_obs(
        self, obs: list[spaces.GraphInstance], next_obs: list[spaces.GraphInstance]
    ) -> None:
        self.observations.append(obs)
        self.next_observations.append(next_obs)

    def _get_observations_samples(
        self,
        batch_inds: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> tuple[thg.data.Data, thg.data.Data]:
        if env is not None:
            raise NotImplementedError(
                "observation normalization not yet implemented for graphReplayBuffer."
            )
        observations = self._extract_observations_samples(
            self.observations, batch_inds=batch_inds
        )
        next_observations = self._extract_observations_samples(
            self.next_observations, batch_inds=batch_inds
        )
        return observations, next_observations

    def _extract_observations_samples(
        self, observations: list[spaces.GraphInstance], batch_inds: np.ndarray
    ) -> thg.data.Data:
        return self._graphlist_to_torch(self.observations, batch_inds=batch_inds)


class DictGraphReplayBuffer(GraphReplayBuffer, DictReplayBuffer):
    observations: dict[
        str,
        Union[
            list[spaces.GraphInstance],
            np.ndarray,
        ],
    ]
    next_observations: dict[
        str,
        Union[
            list[spaces.GraphInstance],
            np.ndarray,
        ],
    ]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        self.is_observation_subspace_graph: dict[str, bool] = {
            k: isinstance(space, spaces.Graph)
            for k, space in observation_space.spaces.items()
        }
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )

    def _init_observations(self):
        for k, is_graph in self.is_observation_subspace_graph.items():
            if is_graph:
                self.observations[k] = list()
                self.next_observations[k] = list()

    def _add_obs(
        self,
        obs: dict[str, Union[np.ndarray, list[spaces.GraphInstance]]],
        next_obs: dict[str, Union[np.ndarray, list[spaces.GraphInstance]]],
    ) -> None:
        for key in self.observations.keys():
            if self.is_observation_subspace_graph[key]:
                self.observations[key].append(
                    [copy_graph_instance(g) for g in obs[key]]
                )
                self.next_observations[key].append(
                    [copy_graph_instance(g) for g in next_obs[key]]
                )
            else:
                obs_ = np.array(obs[key])
                next_obs_ = np.array(next_obs[key])
                # Reshape needed when using multiple envs with discrete observations
                # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
                    next_obs_ = next_obs_.reshape((self.n_envs,) + self.obs_shape[key])
                self.observations[key][self.pos] = obs_
                self.next_observations[key][self.pos] = next_obs_

    def _extract_observations_samples(
        self,
        observations: dict[
            str,
            Union[
                list[spaces.GraphInstance],
                np.ndarray,
            ],
        ],
        batch_inds: np.ndarray,
    ) -> dict[str, Union[thg.data.Data, th.Tensor]]:
        return {
            k: self._graphlist_to_torch(obs, batch_inds=batch_inds)
            if self.is_observation_subspace_graph[k]
            else self.to_torch(obs[batch_inds])
            for k, obs in observations.items()
        }
