#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from typing import Any

import gymnasium as gym
from ray.rllib.connectors.env_to_module.observation_preprocessor import (
    MultiAgentObservationPreprocessor,
)
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode


class PerAgentObservationPreprocessor(MultiAgentObservationPreprocessor):
    """Preprocessor applied to each agent observation."""

    @abstractmethod
    def preprocess_agent_obs(self, observation: Any, agent_id: str) -> Any:
        """Override to implement the preprocessing logic per agent.

        # Parameters
        observation: An observation from an agent to be preprocessed
        agent_id

        # Returns
        The new agent observation
        """
        ...

    @abstractmethod
    def compute_output_agent_observation_space(
        self,
        input_agent_observation_space: gym.Space,
        input_action_space: gym.Space,
        agent_id: str,
    ) -> gym.Space:
        """Compute each agent output observation space.

        # Parameters
        input_agent_observation_space
        input_action_space
        agent_id

        # Returns

        """
        ...

    def recompute_output_observation_space(
        self, input_observation_space: gym.spaces.Dict, input_action_space: gym.Space
    ) -> gym.Space:
        return gym.spaces.Dict(
            {
                agent_id: self.compute_output_agent_observation_space(
                    input_agent_observation_space=space,
                    input_action_space=input_action_space,
                    agent_id=agent_id,
                )
                for agent_id, space in input_observation_space.items()
            }
        )

    def preprocess(
        self, observations: dict[str, Any], episode: MultiAgentEpisode
    ) -> dict[str, Any]:
        return {
            agent_id: self.preprocess_agent_obs(
                observation=agent_observation,
                agent_id=agent_id,
            )
            for agent_id, agent_observation in observations.items()
        }
