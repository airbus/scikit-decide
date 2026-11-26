#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from abc import abstractmethod
from typing import Any

import gymnasium as gym
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space

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
