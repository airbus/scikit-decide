from typing import Any

import gymnasium as gym
import numpy as np

from skdecide import EnumerableSpace
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import DEFAULT_N_NODES

TRUE_OBS = "true_obs"
ACTION_MASK = "valid_avail_actions_mask"


def create_agent_action_mask_space(
    agent_action_space: EnumerableSpace, graph2node: bool
):
    if graph2node:
        # When converting obs space (Graph) to a dict, we artificially specify a number of nodes/edges
        # We need to use the same number of nodes for the dimension of the action mask space,
        # for dummy samples generation purposes
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
