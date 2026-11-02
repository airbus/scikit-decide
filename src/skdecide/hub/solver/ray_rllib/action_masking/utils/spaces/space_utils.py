from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from skdecide import EnumerableSpace
from skdecide.hub.solver.ray_rllib.common.utils.spaces.utils import (
    get_item,
    pad_axis,
    set_item,
)

TRUE_OBS = "observations"
ACTION_MASK = "action_mask"


def create_agent_action_mask_space(
    agent_action_space: EnumerableSpace, graph2node: bool
):
    if graph2node:
        # When converting obs space (Graph) to a dict, we artificially specify a number of nodes/edges
        # We need to use the same number of nodes for the dimension of the action mask space,
        # for dummy samples generation purposes
        # NB: local import to avoid circular import
        from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
            DEFAULT_N_NODES,
        )

        action_mask_dim = DEFAULT_N_NODES
    else:
        # we derive the action mask dim from actual action space size
        action_mask_dim = len(agent_action_space.get_elements())
    return gym.spaces.Box(
        0,
        1,
        shape=(action_mask_dim,),
        dtype=np.int8,
    )


def is_masked_obs(x: Any) -> bool:
    return isinstance(x, dict) and len(x) == 2 and TRUE_OBS in x and ACTION_MASK in x


def is_masked_obs_space(x: gym.spaces.Space) -> bool:
    return (
        isinstance(x, gym.spaces.Dict)
        and len(x.spaces) == 2
        and TRUE_OBS in x.spaces
        and ACTION_MASK in x.spaces
    )


def pad_sample_batches_action_mask(
    samples: list[dict[str, Any]], keys: tuple[str, ...], has_batch_dim: bool = True
) -> None:
    node_edge_id_dim = 1 if has_batch_dim else 0
    n_nodes_per_sample = tuple(
        get_item(s, keys).shape[node_edge_id_dim] for s in samples
    )
    if len(set(n_nodes_per_sample)) > 1:
        # different number of nodes => padding
        max_n_nodes = max(n_nodes_per_sample)
        for s in samples:
            action_mask = get_item(s, keys)
            set_item(
                s,
                keys=keys,
                value=pad_axis(
                    action_mask, max_n_nodes, axis=node_edge_id_dim, value=0
                ).view(
                    type(action_mask)  # keep BatchedNDArray wrapper if present
                ),
            )
