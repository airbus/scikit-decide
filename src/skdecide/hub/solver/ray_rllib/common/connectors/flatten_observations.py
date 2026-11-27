#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from abc import abstractmethod
from math import prod
from typing import Any, Optional

import gymnasium as gym
from ray.rllib.utils.numpy import flatten_inputs_to_1d_tensor
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import TensorType

from skdecide.hub.solver.ray_rllib.common.connectors.observation_preprocessor import (
    PerAgentObservationPreprocessor,
)

SpaceBaseStruct = (
    gym.spaces.Space | dict[str, "SpaceBaseStruct"] | tuple["SpaceBaseStruct"]
)


class PerAgentBaseFlattenObservations(PerAgentObservationPreprocessor):
    """A connector piece that apply a flattening to each agent observation."""

    _input_mutiagent_obs_space_base_struct: dict[str, SpaceBaseStruct]
    """base struct of input observation multiagent space.

    Dict spaces are replaced by plain dict and Tuple spaces by plain tuple.

    """

    @abstractmethod
    def flatten_agent_observation(
        self,
        observation: Any,
        observation_space: gym.Space,
        observation_space_base_struct: SpaceBaseStruct,
        output_observation_space: gym.Space,
    ) -> Any:
        """Flatten agent observation

        # Parameters
        observation: agent observation to flatten
        observation_space: agent observation space
        observation_space_base_struct: base struct of the agent observation space
        output_observation_space: expected flattened observation space

        # Returns
        The flattened agent observation space

        """
        ...

    @abstractmethod
    def flatten_agent_observation_space(
        self,
        observation_space: gym.Space,
        observation_space_base_struct: SpaceBaseStruct,
    ) -> gym.Space:
        """Flatten agent observation space

        # Parameters
        observation_space: agent observation space to flatten
        observation_space_base_struct: base struct of the agent observation space

        # Returns
        The flattened agent observation space

        """
        ...

    def preprocess_agent_obs(self, observation: Any, agent_id: str) -> Any:
        return self.flatten_agent_observation(
            observation=observation,
            observation_space=self.input_observation_space[agent_id],
            observation_space_base_struct=self._input_mutiagent_obs_space_base_struct[
                agent_id
            ],
            output_observation_space=self.observation_space[agent_id],
        )

    def compute_output_agent_observation_space(
        self,
        input_agent_observation_space: gym.Space,
        input_action_space: gym.Space,
        agent_id: str,
    ) -> gym.Space:
        return self.flatten_agent_observation_space(
            observation_space=input_agent_observation_space,
            observation_space_base_struct=self._input_mutiagent_obs_space_base_struct[
                agent_id
            ],
        )

    def recompute_output_observation_space(
        self, input_observation_space: gym.spaces.Dict, input_action_space: gym.Space
    ) -> gym.Space:
        self._input_mutiagent_obs_space_base_struct = get_base_struct_from_space(
            input_observation_space
        )
        return super().recompute_output_observation_space(
            input_observation_space, input_action_space
        )


class FlattenMultiagentObservations(PerAgentBaseFlattenObservations):
    def __init__(
        self,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        flatten_3d_obs: bool = True,
        **kwargs,
    ):
        super().__init__(input_observation_space, input_action_space, **kwargs)
        self.flatten_3d_obs = flatten_3d_obs

    def flatten_agent_observation(
        self,
        observation: Any,
        observation_space: gym.Space,
        observation_space_base_struct: SpaceBaseStruct,
        output_observation_space: gym.Space,
    ) -> Any:
        return flatten_ordinary_obs(
            observation=observation,
            space_base_struct=observation_space_base_struct,
            flatten_3d_obs=self.flatten_3d_obs,
        )

    def flatten_agent_observation_space(
        self,
        observation_space: gym.Space,
        observation_space_base_struct: SpaceBaseStruct,
    ) -> gym.Space:
        return flatten_ordinary_space(
            space=observation_space,
            space_base_struct=observation_space_base_struct,
            flatten_3d_obs=self.flatten_3d_obs,
        )


def flatten_ordinary_space(
    space: gym.spaces.Space,
    space_base_struct: SpaceBaseStruct,
    flatten_3d_obs: bool = True,
) -> gym.spaces.Box:
    """Flatten "ordinary" space

    # Parameters
    space: space to flatten
    space_base_struct: base struct representing the space (to avoid recomputing it if already done)
        Dict spaces are replaced by plain dict and Tuple spaces by plain tuple.
    flatten_3d_obs: whether to also flatten 3d box spaces

    # Returns
    The flattened space

    """
    if isinstance(space, gym.spaces.Box):
        if not flatten_3d_obs and len(space.shape) == 3:
            # avoid flattening 3d box
            return space

        if prod(space.shape) == 0:
            # special case of "no feature"
            return gym.spaces.Box(low=0, high=0, shape=(0,), dtype=space.dtype)
        low = space.low.min()
        high = space.high.max()
    elif isinstance(space, gym.spaces.Discrete) or isinstance(
        space, gym.spaces.MultiDiscrete
    ):
        low = 0
        high = 1
    else:
        low = float("-inf")
        high = float("inf")
    sample = flatten_inputs_to_1d_tensor(
        space.sample(), space_base_struct, batch_axis=False
    )
    return gym.spaces.Box(low=low, high=high, shape=sample.shape, dtype=sample.dtype)


TensorStruct = TensorType | dict[str, "TensorStruct"] | tuple["TensorStruct"]


def flatten_ordinary_obs(
    observation: TensorStruct,
    space_base_struct: SpaceBaseStruct,
    flatten_3d_obs: bool = True,
) -> dict[str, Any]:
    if (
        not flatten_3d_obs
        and isinstance(space_base_struct, gym.spaces.Box)
        and len(space_base_struct.shape) == 3
    ):
        # avoid flattening 3d obs
        return observation
    return flatten_inputs_to_1d_tensor(
        inputs=observation,
        spaces_struct=space_base_struct,
        batch_axis=False,
    )
