#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import tree
from ray.rllib.connectors.env_to_module.observation_preprocessor import (
    MultiAgentObservationPreprocessor,
)
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.utils.numpy import flatten_inputs_to_1d_tensor
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space

from skdecide.hub.solver.ray_rllib.action_masking.utils.spaces.space_utils import (
    ACTION_MASK,
    TRUE_OBS,
)


class FlattenMultiagentMaskedObservations(MultiAgentObservationPreprocessor):
    """A connector piece that flattens "true" observation components into a 1D array, and keep action mask.

        It assumes that observation structure is as  follows:
        ```
        obs = {
            agent_id: {
                true_obs_key: true_obs
                action_mask_key: action_mask
            },
        }
        ```
        where `action_mask` is the mask already flattened and `true_obs` is the original observation.
    0000000000

    """

    def __init__(
        self,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        true_obs_key: str = TRUE_OBS,
        action_mask_key: str = ACTION_MASK,
        **kwargs,
    ):
        super().__init__(input_observation_space, input_action_space, **kwargs)
        self.true_obs_key = true_obs_key
        self.action_mask_key = action_mask_key

    def preprocess(
        self, observations: dict[str, Any], episode: MultiAgentEpisode
    ) -> dict[str, Any]:
        return {
            agent: {
                # flatten "true" obs
                self.true_obs_key: flatten_inputs_to_1d_tensor(
                    inputs=masked_observation[self.true_obs_key],
                    spaces_struct=self._input_obs_base_struct[agent][self.true_obs_key],
                    # Our items are individual observations (no batch axis present).
                    batch_axis=False,
                ),
                # keep mask unchanged
                self.action_mask_key: masked_observation[self.action_mask_key],
            }
            for agent, masked_observation in observations.items()
        }

    def recompute_output_observation_space(
        self, input_observation_space: gym.Space, input_action_space: gym.Space
    ) -> gym.Space:
        self._input_obs_base_struct = get_base_struct_from_space(
            self.input_observation_space
        )
        assert isinstance(input_observation_space, gym.spaces.Dict), (
            f"To flatten a Multi-Agent observation, it is expected that observation space is a dictionary, its actual type is {type(input_observation_space)}"
        )
        spaces = {}
        for agent_id, space in input_observation_space.items():
            assert isinstance(space, gym.spaces.Dict), (
                f"Agent masked observation space is expected to be a dict with keys {self.true_obs_key} and {self.action_mask_key}, its actual type is {type(space)}"
            )
            assert set(space.spaces) == {self.true_obs_key, self.action_mask_key}, (
                f"Agent masked observation space is expected to be a dict with keys {self.true_obs_key} and {self.action_mask_key}, its actual keys are {set(space.spaces)}"
            )
            sample = flatten_inputs_to_1d_tensor(
                tree.map_structure(
                    lambda s: s.sample(),
                    self._input_obs_base_struct[agent_id][self.true_obs_key],
                ),
                self._input_obs_base_struct[agent_id][self.true_obs_key],
                batch_axis=False,
            )
            flattened_true_observation_space = gym.spaces.Box(
                float("-inf"), float("inf"), (len(sample),), np.float32
            )
            action_mask_space = space[self.action_mask_key]

            spaces[agent_id] = gym.spaces.Dict(
                {
                    self.true_obs_key: flattened_true_observation_space,
                    self.action_mask_key: action_mask_space,
                }
            )
        return gym.spaces.Dict(spaces)
