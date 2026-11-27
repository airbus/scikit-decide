#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Optional

import gymnasium as gym

from skdecide.hub.solver.ray_rllib.action_masking.connectors.flatten_observations import (
    flatten_unmasked_obs,
    flatten_unmasked_space,
)
from skdecide.hub.solver.ray_rllib.autoregressive.utils.spaces.space_utils import (
    APPLICABLE_ACTIONS,
    TRUE_OBS,
)
from skdecide.hub.solver.ray_rllib.common.connectors.flatten_observations import (
    PerAgentBaseFlattenObservations,
    SpaceBaseStruct,
)


class FlattenMultiagentAutoregressiveActionObservations(
    PerAgentBaseFlattenObservations
):
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

    """

    def __init__(
        self,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        is_graph_dict: bool = False,
        is_multiinput: bool = False,
        flatten_3d_obs: bool = True,
        **kwargs,
    ):
        self.is_graph_dict = is_graph_dict
        self.is_multiinput = is_multiinput
        self.flatten_3d_obs = flatten_3d_obs
        super().__init__(input_observation_space, input_action_space, **kwargs)

    def flatten_agent_observation(
        self,
        observation: Any,
        observation_space: gym.Space,
        observation_space_base_struct: SpaceBaseStruct,
        output_observation_space: gym.Space,
    ) -> Any:
        return {
            TRUE_OBS: flatten_unmasked_obs(
                observation=observation[TRUE_OBS],
                input_observation_space=observation_space[TRUE_OBS],
                output_observation_space=output_observation_space[TRUE_OBS],
                space_base_struct=observation_space_base_struct[TRUE_OBS],
                is_graph_dict=self.is_graph_dict,
                is_multiinput=self.is_multiinput,
                flatten_3d_obs=self.flatten_3d_obs,
            ),
            APPLICABLE_ACTIONS: observation[APPLICABLE_ACTIONS],
        }

    def flatten_agent_observation_space(
        self,
        observation_space: gym.Space,
        observation_space_base_struct: SpaceBaseStruct,
    ) -> gym.Space:
        assert isinstance(observation_space, gym.spaces.Dict)
        return gym.spaces.Dict(
            {
                TRUE_OBS: flatten_unmasked_space(
                    space=observation_space[TRUE_OBS],
                    space_base_struct=observation_space_base_struct[TRUE_OBS],
                    is_graph_dict=self.is_graph_dict,
                    is_multiinput=self.is_multiinput,
                    flatten_3d_obs=self.flatten_3d_obs,
                ),
                APPLICABLE_ACTIONS: observation_space[APPLICABLE_ACTIONS],
            }
        )
