#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Optional

import gymnasium as gym

from skdecide.hub.solver.ray_rllib.action_masking.utils.spaces.space_utils import (
    ACTION_MASK,
    TRUE_OBS,
)
from skdecide.hub.solver.ray_rllib.common.connectors.flatten_observations import (
    PerAgentBaseFlattenObservations,
    SpaceBaseStruct,
)
from skdecide.hub.solver.ray_rllib.gnn.connectors.flatten_observations import (
    flatten_graph_dict_obs,
    flatten_graph_dict_space,
    flatten_multiinput_obs,
    flatten_multiinput_space,
    flatten_ordinary_obs,
    flatten_ordinary_space,
)


class FlattenMultiagentMaskedObservations(PerAgentBaseFlattenObservations):
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
        **kwargs,
    ):
        self.is_graph_dict = is_graph_dict
        self.is_multiinput = is_multiinput
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
            ),
            ACTION_MASK: observation[ACTION_MASK],
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
                ),
                ACTION_MASK: observation_space[ACTION_MASK],
            }
        )


def flatten_unmasked_space(
    space: gym.Space,
    space_base_struct: SpaceBaseStruct,
    is_graph_dict: bool,
    is_multiinput: bool,
):
    """Flatten unmasked space

    Can be either:
    - graph dict space
    - multiinput (including graph dict) space
    - "ordinary" space

    # Parameters
    space: space to flatten
    space_base_struct: base struct representing the space (to avoid recomputing it if already done)
        Dict spaces are replaced by plain dict and Tuple spaces by plain tuple.
    is_graph_dict: input space is seen as a graph dict space (overrides `is_multiinput`)
    is_multiinput: input space is seen as a multiinput space

    # Returns
    The flattened space

    """
    if is_graph_dict:
        return flatten_graph_dict_space(
            space=space, space_base_struct=space_base_struct
        )
    elif is_multiinput:
        return flatten_multiinput_space(
            space=space, space_base_struct=space_base_struct
        )
    else:
        return flatten_ordinary_space(space=space, space_base_struct=space_base_struct)


def flatten_unmasked_obs(
    observation: Any,
    input_observation_space: gym.spaces.Dict,
    output_observation_space: gym.spaces.Dict,
    space_base_struct: dict[str, SpaceBaseStruct],
    is_graph_dict: bool,
    is_multiinput: bool,
):
    """Flatten unmasked observation

    Can be either:
    - graph dict
    - multiinput (including graph dict)
    - "ordinary"

    # Returns
    The flattened space

    """

    if is_graph_dict:
        return flatten_graph_dict_obs(
            observation=observation,
            output_observation_space=output_observation_space,
            space_base_struct=space_base_struct,
        )
    elif is_multiinput:
        return flatten_multiinput_obs(
            observation=observation,
            input_observation_space=input_observation_space,
            output_observation_space=output_observation_space,
            space_base_struct=space_base_struct,
        )
    else:
        return flatten_ordinary_obs(
            observation=observation, space_base_struct=space_base_struct
        )
