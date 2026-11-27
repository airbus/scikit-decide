from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from ray.rllib.utils.spaces.repeated import Repeated

from skdecide.hub.solver.ray_rllib.common.utils.spaces.utils import (
    get_item,
    pad_axis,
    set_item,
)
from skdecide.hub.space.gym import MultiDiscreteSpace

TRUE_OBS = "observations"
APPLICABLE_ACTIONS = "applicable_actions"

DEFAULT_N_APPLICABLE_ACTIONS = 2
"""Initial applicable actions number to be used by dummy samples generation by rllib"""


def create_agent_applicable_actions_space(agent_action_space: MultiDiscreteSpace):
    return Repeated(
        child_space=agent_action_space.unwrapped(),
        max_len=DEFAULT_N_APPLICABLE_ACTIONS,
    )


def is_applicable_actions_obs(x: Any) -> bool:
    return (
        isinstance(x, dict)
        and len(x) == 2
        and TRUE_OBS in x
        and APPLICABLE_ACTIONS in x
    )


def is_applicable_actions_obs_space(x: gym.spaces.Space) -> bool:
    return (
        isinstance(x, gym.spaces.Dict)
        and len(x.spaces) == 2
        and TRUE_OBS in x.spaces
        and APPLICABLE_ACTIONS in x.spaces
    )


def pad_sample_batches_applicable_actions(
    samples: list[dict[str, Any]], keys: tuple[str, ...], has_batch_dim: bool = True
) -> None:
    applicable_action_id_dim = 1 if has_batch_dim else 0
    node_edge_id_dim = applicable_action_id_dim + 1
    n_nodes_per_sample = tuple(
        get_item(s, keys).shape[node_edge_id_dim] for s in samples
    )
    n_applicable_actions_per_sample = tuple(
        get_item(s, keys).shape[applicable_action_id_dim] for s in samples
    )
    if (
        len(set(n_nodes_per_sample)) > 1
        or len(set(n_applicable_actions_per_sample)) > 1
    ):
        # different number of nodes and/or number applicable actions => padding
        max_n_nodes = max(n_nodes_per_sample)
        max_n_applicable_actions = max(n_applicable_actions_per_sample)
        for s in samples:
            applicable_actions = get_item(s, keys)
            set_item(
                s,
                keys=keys,
                value=pad_axis(
                    pad_axis(
                        applicable_actions,
                        max_n_nodes,
                        axis=node_edge_id_dim,
                        value=-1,  # negative component -> invalid component
                    ),
                    max_n_applicable_actions,
                    axis=applicable_action_id_dim,
                    value=-1,  # negative values => invalid action
                ).view(
                    type(applicable_actions)  # keep BatchedNDArray wrapper if present
                ),
            )


class GraphNodeMultiDiscrete(gym.spaces.MultiDiscrete):
    """Gym space for actions with graph node components.

    More precisely, some components are actually ids of observation graph nodes

    # Attributes
    nb_graph_independent_components: number of action components that are independent of the graph.
        It will be the first components, the next ones are assumed to be graph nodes (identified by the node index).

    """

    def __init__(
        self,
        nvec: NDArray[np.integer[Any]] | list[int],
        nb_graph_independent_components: int = 1,
        dtype: str | type[np.integer[Any]] = np.int64,
        seed: int | np.random.Generator | None = None,
        start: NDArray[np.integer[Any]] | list[int] | None = None,
    ):
        super().__init__(nvec=nvec, dtype=dtype, seed=seed, start=start)
        self.nb_graph_independent_components = nb_graph_independent_components

    @property
    def nb_graph2node_components(self) -> int:
        return len(self.nvec) - self.nb_graph_independent_components

    def is_graphnode(self, i_component: int) -> bool:
        return i_component > self.nb_graph_independent_components


class HeteroGraphNodeMultiDiscrete(GraphNodeMultiDiscrete):
    """Gym space multidiscrete actions with heterogeneous graph node components.

    More precisely:
     - some components are actually ids of observation graph nodes
     - the graph nodes can only be a subpart of the graph nodes corresponding to a given type of node.

    To node types are specified via binary node features. Each component can correspond to a different type of node,
    and thus will have a specific binary feature specified via its index among node features.

    # Attributes
    action_components_node_flag_indices: node feature index, component by component, so that each corresponding action
        component will be the index of the chosen node among the nodes of the proper type,
        i.e. for which the specified node feature is equal to 1.
        For graph independent nodes, the node feature index must be None. (It must be the first components.)

    """

    def __init__(
        self,
        nvec: NDArray[np.integer[Any]] | list[int],
        action_components_node_flag_indices: list[int | None],
        dtype: str | type[np.integer[Any]] = np.int64,
        seed: int | np.random.Generator | None = None,
        start: NDArray[np.integer[Any]] | list[int] | None = None,
    ):
        nb_graph_independent_components = sum(
            index is None for index in action_components_node_flag_indices
        )
        assert all(
            isinstance(index, int)
            for index in action_components_node_flag_indices[
                nb_graph_independent_components:
            ]
        ), (
            "Graph independent components (i.e. with `None` value in action_components_node_flag_indices) "
            "must be the first components."
        )
        super().__init__(
            nvec=nvec,
            dtype=dtype,
            seed=seed,
            start=start,
            nb_graph_independent_components=nb_graph_independent_components,
        )
        self.action_components_node_flag_indices = action_components_node_flag_indices


def create_agent_autoregressive_action_space(
    agent_action_space: MultiDiscreteSpace,
    action_components_node_flag_indices: list[int | None] | None = None,
    nb_graph_independent_components: int | None = None,
) -> gym.spaces.MultiDiscrete:
    """Return a space to be used as action space by RL-modules."""
    unwrapped_agent_action_space: gym.spaces.MultiDiscrete = (
        agent_action_space.unwrapped()
    )
    if action_components_node_flag_indices is not None:
        if nb_graph_independent_components is not None:
            assert nb_graph_independent_components == sum(
                index is None for index in action_components_node_flag_indices
            ), (
                "`nb_graph_independent_components` should correspond to the number of `None` in `action_components_node_flag_indices`."
            )
        return HeteroGraphNodeMultiDiscrete(
            nvec=unwrapped_agent_action_space.nvec,
            action_components_node_flag_indices=action_components_node_flag_indices,
        )
    elif (
        nb_graph_independent_components is not None
        and nb_graph_independent_components < len(unwrapped_agent_action_space.nvec)
    ):
        return GraphNodeMultiDiscrete(
            nvec=unwrapped_agent_action_space.nvec,
            nb_graph_independent_components=nb_graph_independent_components,
        )
    else:
        return gym.spaces.MultiDiscrete(nvec=unwrapped_agent_action_space.nvec)
