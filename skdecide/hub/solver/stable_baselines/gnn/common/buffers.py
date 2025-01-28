from collections.abc import Generator
from typing import Any, Dict, List, Optional, TypeVar, Union

import numpy as np
import torch as th
import torch_geometric as thg
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import (
    MaskableDictRolloutBuffer,
    MaskableRolloutBuffer,
    MaskableRolloutBufferSamples,
)
from stable_baselines3.common.buffers import (
    BaseBuffer,
    DictReplayBuffer,
    DictRolloutBuffer,
    ReplayBuffer,
    RolloutBuffer,
)
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from .preprocessing import get_action_dim, get_obs_shape
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


class GraphRolloutBuffer(RolloutBuffer, GraphBaseBuffer):
    """Rollout buffer used in on-policy algorithms like A2C/PPO with graph observations.

    Handles cases where observation space is:
    - a Graph space
    - a Dict space whose subspaces includes a Graph space

    """

    observations: Union[list[spaces.GraphInstance], list[list[spaces.GraphInstance]]]
    tensor_names = ["values", "log_probs", "advantages", "returns"]

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
        self._add_action(action)
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def _add_action(self, action: np.ndarray) -> None:
        action = action.reshape((self.n_envs, self.action_dim))
        self.actions[self.pos] = np.array(action).copy()

    def _add_obs(self, obs: list[spaces.GraphInstance]) -> None:
        self.observations.append([copy_graph_instance(g) for g in obs])

    def _swap_and_flatten_obs(self) -> None:
        self.observations = _swap_and_flatten_nested_list(self.observations)

    def _swap_and_flatten_action(self) -> None:
        self.actions = self.swap_and_flatten(self.actions)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            self._swap_and_flatten_obs()
            self._swap_and_flatten_action()
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
        actions = self._get_actions_samples(batch_inds)
        data = (
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(
            observations, actions, *tuple(map(self.to_torch, data))
        )

    def _get_observations_samples(self, batch_inds: np.ndarray) -> thg.data.Data:
        return self._graphlist_to_torch(self.observations, batch_inds=batch_inds)

    def _get_actions_samples(self, batch_inds: np.ndarray) -> th.Tensor:
        return self.to_torch(self.actions[batch_inds])


class Graph2GraphRolloutBuffer(GraphRolloutBuffer):
    """Rollout buffer when both observations and actions are graphs."""

    actions: Union[list[spaces.GraphInstance], list[list[spaces.GraphInstance]]]

    def reset(self) -> None:
        assert isinstance(
            self.action_space, spaces.Graph
        ), "Graph2GraphRolloutBuffer must be used with Graph action space only"
        super().reset()
        self.actions = list()

    def _add_action(self, action: list[spaces.GraphInstance]) -> None:
        self.actions.append([copy_graph_instance(g) for g in action])

    def _swap_and_flatten_action(self) -> None:
        self.actions = _swap_and_flatten_nested_list(self.actions)

    def _get_actions_samples(self, batch_inds: np.ndarray) -> thg.data.Data:
        return self._graphlist_to_torch(self.actions, batch_inds=batch_inds)


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


class _BaseMaskableRolloutBuffer:

    tensor_names = [
        "values",
        "log_probs",
        "advantages",
        "returns",
        "action_masks",
    ]

    def add(self, *args, action_masks: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape(
                (self.n_envs, self.mask_dims)
            )

        super().add(*args, **kwargs)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> MaskableRolloutBufferSamples:
        samples_wo_action_masks = super()._get_samples(batch_inds=batch_inds, env=env)
        return MaskableRolloutBufferSamples(
            *samples_wo_action_masks,
            action_masks=self.action_masks[batch_inds].reshape(-1, self.mask_dims),
        )


class MaskableGraphRolloutBuffer(
    _BaseMaskableRolloutBuffer, GraphRolloutBuffer, MaskableRolloutBuffer
):
    ...


class MaskableDictGraphRolloutBuffer(
    _BaseMaskableRolloutBuffer, DictGraphRolloutBuffer, MaskableDictRolloutBuffer
):
    ...


class GraphReplayBuffer(ReplayBuffer, GraphBaseBuffer):
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

    def add(
        self,
        obs: list[spaces.GraphInstance],
        next_obs: list[spaces.GraphInstance],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        self._add_obs(obs=obs, next_obs=next_obs)

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _add_obs(
        self, obs: list[spaces.GraphInstance], next_obs: list[spaces.GraphInstance]
    ) -> None:
        self.observations.append(obs)
        self.next_observations.append(next_obs)

    def _get_observations_samples(
        self, observations: list[spaces.GraphInstance], batch_inds: np.ndarray
    ) -> thg.data.Data:
        return self._graphlist_to_torch(observations, batch_inds=batch_inds)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        if env is not None:
            raise NotImplementedError(
                "observation normalization not yet implemented for graphReplayBuffer."
            )
        env_indices = 0  # single env
        return ReplayBufferSamples(
            observations=self._get_observations_samples(
                self.observations, batch_inds=batch_inds
            ),
            actions=self.to_torch(self.actions[batch_inds, env_indices, :]),
            next_observations=self._get_observations_samples(
                self.next_observations, batch_inds=batch_inds
            ),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                (
                    self.dones[batch_inds, env_indices]
                    * (1 - self.timeouts[batch_inds, env_indices])
                ).reshape(-1, 1)
            ),
            rewards=self.to_torch(
                self._normalize_reward(
                    self.rewards[batch_inds, env_indices].reshape(-1, 1), env
                )
            ),
        )


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

    def _get_observations_samples(
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


T = TypeVar("T")


def _swap_and_flatten_nested_list(obs: list[list[T]]) -> list[T]:
    return [x for subobs in obs for x in subobs]
