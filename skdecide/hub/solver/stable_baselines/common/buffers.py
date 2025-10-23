from collections.abc import Generator
from typing import Any, Optional, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import MaskableRolloutBufferSamples
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

T = TypeVar("T")


def swap_and_flatten_nested_list(obs: list[list[T]]) -> list[T]:
    return [x for subobs in obs for x in subobs]


class ScikitDecideRolloutBuffer(RolloutBuffer):
    """Base class for scikit-decide customized RolloutBuffer."""

    tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

    def add(
        self,
        obs: np.ndarray,
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

    def _add_obs(self, obs: np.ndarray) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        self.observations[self.pos] = np.array(obs)

    def _swap_and_flatten_obs(self) -> None:
        self.observations = self.swap_and_flatten(self.observations)

    def _swap_and_flatten_action_masks(self) -> None:
        """Method to override in buffers meant to be used with action masks."""
        # by default, no action masks
        ...

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            self._swap_and_flatten_obs()
            self._swap_and_flatten_action_masks()
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

    def _get_observations_samples(self, batch_inds: np.ndarray) -> th.Tensor:
        return self.to_torch(self.observations[batch_inds])


class MaskableScikitDecideRolloutBufferMixin:
    """Mixin to add to ScikitDecideRolloutBuffer for maskable buffers."""

    def add(self, *args, action_masks: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """

        self._add_action_masks(action_masks=action_masks)
        super().add(*args, **kwargs)

    def _add_action_masks(self, action_masks: Optional[np.ndarray] = None):
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape(
                (self.n_envs, self.mask_dims)
            )

    def _swap_and_flatten_action_masks(self) -> None:
        self.action_masks = self.swap_and_flatten(self.action_masks)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> MaskableRolloutBufferSamples:
        samples_wo_action_masks = super()._get_samples(batch_inds=batch_inds, env=env)
        action_masks = self._get_action_masks_samples(batch_inds=batch_inds)
        return MaskableRolloutBufferSamples(
            *samples_wo_action_masks,
            action_masks=action_masks,
        )

    def _get_action_masks_samples(self, batch_inds: np.ndarray) -> np.ndarray:
        return self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims))


class ScikitDecideReplayBuffer(ReplayBuffer):
    def add(
        self,
        obs: list[spaces.GraphInstance],
        next_obs: list[spaces.GraphInstance],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
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
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

    def _get_observations_samples(
        self,
        batch_inds: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        obs = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )
        return self.to_torch(obs), self.to_torch(next_obs)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        observations, next_observations = self._get_observations_samples(
            batch_inds=batch_inds, env_indices=env_indices, env=env
        )

        return ReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices, :]),
            next_observations=next_observations,
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
