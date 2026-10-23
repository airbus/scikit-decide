#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from ray.rllib.utils.numpy import flatten_inputs_to_1d_tensor
from ray.rllib.utils.spaces.repeated import Repeated

from skdecide.hub.solver.ray_rllib.common.connectors.flatten_observations import (
    PerAgentBaseFlattenObservations,
    SpaceBaseStruct,
    flatten_ordinary_obs,
    flatten_ordinary_space,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    DEFAULT_N_EDGES,
    DEFAULT_N_NODES,
    EDGE_LINKS,
    EDGES,
    NODES,
    is_graph_dict_space,
)


class FlattenMultiagentGraphObservations(PerAgentBaseFlattenObservations):
    """A connector piece that flattens node and edge features into a 1D arrays.

    It assumes that observation structure is as  follows:
    ```
    obs = {
        "nodes": node features
        "edges": edge features
        "edge_links": list of connected nodes
    }
    ```

    """

    def flatten_agent_observation(
        self,
        observation: Any,
        observation_space: gym.Space,
        observation_space_base_struct: SpaceBaseStruct,
        output_observation_space: gym.Space,
    ) -> Any:
        return flatten_graph_dict_obs(
            observation=observation,
            space_base_struct=observation_space_base_struct,
            output_observation_space=output_observation_space,
        )

    def flatten_agent_observation_space(
        self,
        observation_space: gym.Space,
        observation_space_base_struct: SpaceBaseStruct,
    ) -> gym.Space:
        return flatten_graph_dict_space(
            space=observation_space, space_base_struct=observation_space_base_struct
        )


class FlattenMultiagentMultiinputObservations(PerAgentBaseFlattenObservations):
    """A connector piece that flattens each subobs from a dict obs (containing graphs)."""

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
        return flatten_multiinput_obs(
            observation=observation,
            input_observation_space=observation_space,
            output_observation_space=output_observation_space,
            space_base_struct=observation_space_base_struct,
            flatten_3d_obs=self.flatten_3d_obs,
        )

    def flatten_agent_observation_space(
        self,
        observation_space: gym.Space,
        observation_space_base_struct: SpaceBaseStruct,
    ) -> gym.Space:
        return flatten_multiinput_space(
            space=observation_space,
            space_base_struct=observation_space_base_struct,
            flatten_3d_obs=self.flatten_3d_obs,
        )


def flatten_multiinput_space(
    space: gym.spaces.Dict,
    space_base_struct: SpaceBaseStruct,
    flatten_3d_obs: bool = True,
) -> gym.spaces.Dict:
    """Flatten multiinput space

    A multiinput space is a dict space whose subspaces will be flattened, but not the wrapping dict space itself.
    Some of the subspaces can be graph dict spaces.

    # Parameters
    space: space to flatten
    space_base_struct: base struct representing the space (to avoid recomputing it if already done)
        Dict spaces are replaced by plain dict and Tuple spaces by plain tuple.
    flatten_3d_obs: whether to also flatten 3d box spaces

    # Returns
    The flattened space

    """
    return gym.spaces.Dict(
        {
            key: (
                flatten_graph_dict_space(
                    subspace, space_base_struct=space_base_struct[key]
                )
                if is_graph_dict_space(subspace)
                else flatten_ordinary_space(
                    subspace, space_base_struct=space_base_struct[key]
                )
            )
            for key, subspace in space.items()
        }
    )


def flatten_graph_dict_space(
    space: gym.spaces.Dict, space_base_struct: SpaceBaseStruct
) -> gym.spaces.Dict:
    """Flatten graph dict space

    # Parameters
    space: space to flatten
    space_base_struct: base struct representing the space (to avoid recomputing it if already done)
        Dict spaces are replaced by plain dict and Tuple spaces by plain tuple.

    # Returns
    The flattened space

    """
    return gym.spaces.Dict(
        {
            NODES: Repeated(
                flatten_ordinary_space(
                    space[NODES].child_space, space_base_struct=space_base_struct[NODES]
                ),
                max_len=DEFAULT_N_NODES,
            ),
            EDGES: Repeated(
                flatten_ordinary_space(
                    space[EDGES].child_space, space_base_struct=space_base_struct[EDGES]
                ),
                max_len=DEFAULT_N_EDGES,
            ),
            EDGE_LINKS: space[EDGE_LINKS],
        }
    )


def flatten_graph_dict_obs(
    observation: dict[str, Any],
    output_observation_space: gym.spaces.Dict,
    space_base_struct: dict[str, SpaceBaseStruct],
) -> dict[str, Any]:
    return {
        NODES: flatten_inputs_to_1d_tensor(
            inputs=observation[NODES],
            spaces_struct=space_base_struct[NODES].child_space,
            # The node id dim is like a batch axis
            batch_axis=True,
        ),
        EDGES: flatten_inputs_to_1d_tensor(
            inputs=edges,
            spaces_struct=space_base_struct[EDGES].child_space,
            # The edge id dim is like a batch axis
            batch_axis=True,
        )
        if (edges := observation[EDGES]).size > 0
        else np.empty(
            (len(edges),)
            + (edge_output_space := output_observation_space[EDGES].child_space).shape,
            dtype=edge_output_space.dtype,
        ),
        EDGE_LINKS: observation[EDGE_LINKS],
    }


def flatten_multiinput_obs(
    observation: dict[str, Any],
    input_observation_space: gym.spaces.Dict,
    output_observation_space: gym.spaces.Dict,
    space_base_struct: dict[str, SpaceBaseStruct],
    flatten_3d_obs: bool = True,
) -> dict[str, Any]:
    return {
        key: (
            flatten_graph_dict_obs(
                observation=obs,
                space_base_struct=space_base_struct[key],
                output_observation_space=output_observation_space[key],
            )
            if is_graph_dict_space(input_observation_space[key])
            else flatten_ordinary_obs(
                observation=obs,
                space_base_struct=space_base_struct[key],
                flatten_3d_obs=flatten_3d_obs,
            )
        )
        for key, obs in observation.items()
    }
