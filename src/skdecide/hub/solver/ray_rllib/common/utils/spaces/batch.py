#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import tree
from ray.rllib.utils.spaces.space_utils import BatchedNdArray

from skdecide.hub.solver.ray_rllib.action_masking.utils.spaces.space_utils import (
    ACTION_MASK,
    TRUE_OBS,
    is_masked_obs,
    pad_sample_batches_action_mask,
)
from skdecide.hub.solver.ray_rllib.autoregressive.utils.spaces.space_utils import (
    APPLICABLE_ACTIONS,
    is_applicable_actions_obs,
    pad_sample_batches_applicable_actions,
)
from skdecide.hub.solver.ray_rllib.autoregressive.utils.spaces.space_utils import (
    TRUE_OBS as AUTOREGRESSIVE_TRUE_OBS,
)
from skdecide.hub.solver.ray_rllib.common.utils.spaces.batch_original_code import (
    original_batch,
)
from skdecide.hub.solver.ray_rllib.common.utils.spaces.utils import get_item
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    EDGE_LINKS,
    EDGES,
    NODES,
    is_graph_dict,
    is_graph_dict_multiinput,
    pad_batched_graph_dict,
)


def prepare_for_batch(
    list_of_structs: list[Any],
    individual_items_already_have_batch_dim: bool | str = False,
) -> None:
    individual_items_already_have_batch_dim = guess_if_individual_items_already_have_batch_dim(
        list_of_structs,
        individual_items_already_have_batch_dim=individual_items_already_have_batch_dim,
    )
    if len(list_of_structs) > 0:
        pad_sample_batches_obs(
            list_of_structs,
            keys=tuple(),
            has_batch_dim=individual_items_already_have_batch_dim,
        )


def guess_if_individual_items_already_have_batch_dim(
    list_of_structs: list[Any],
    individual_items_already_have_batch_dim: bool | str = False,
) -> bool:
    if individual_items_already_have_batch_dim == "auto":
        first = list_of_structs[0]
        # Nested structures (dict/tuple) require tree traversal; leaves do not.
        is_nested = isinstance(first, (dict, tuple))
        if isinstance(first, BatchedNdArray):
            individual_items_already_have_batch_dim = True
        elif is_nested:
            flat = tree.flatten(first)
            individual_items_already_have_batch_dim = isinstance(
                flat[0], BatchedNdArray
            )
        else:
            individual_items_already_have_batch_dim = False

    return bool(individual_items_already_have_batch_dim)


def batch_skdecide(
    list_of_structs: list[Any],
    *,
    individual_items_already_have_batch_dim: bool | str = False,
) -> Any:
    """Enhance `ray.rllib.utils.spaces.space_utils.batch` to manage graphs."""
    prepare_for_batch(
        list_of_structs,
        individual_items_already_have_batch_dim=individual_items_already_have_batch_dim,
    )
    return original_batch(
        list_of_structs=list_of_structs,
        individual_items_already_have_batch_dim=individual_items_already_have_batch_dim,
    )


def pad_sample_batches_obs(
    samples: list[Any], keys: tuple[str, ...], has_batch_dim: bool = True
) -> None:
    node_edge_id_dim = 1 if has_batch_dim else 0
    first_subobs = get_item(samples[0], keys)
    if is_graph_dict(first_subobs):
        graph_dicts = tuple(get_item(s, keys) for s in samples)
        nb_nodes_per_graph = tuple(
            graph_dict[NODES].shape[node_edge_id_dim] for graph_dict in graph_dicts
        )
        nb_edges_per_graph = tuple(
            graph_dict[EDGES].shape[node_edge_id_dim] for graph_dict in graph_dicts
        )
        # We compute also nb_edge_links_per_graph as it can be different from nb_edges if some graphs have been padded
        # (=> nb_edge_links = nb_edges +1) and some not (=> nb_edge_links = nb_edges).
        nb_edge_links_per_graph = tuple(
            graph_dict[EDGE_LINKS].shape[node_edge_id_dim] for graph_dict in graph_dicts
        )
        if (
            len(set(nb_nodes_per_graph)) > 1
            or len(set(nb_edges_per_graph)) > 1
            or len(set(nb_edge_links_per_graph)) > 1
        ):
            # different number of nodes or edges => padding
            # different of nb of edge_links means some graph padded other not => padding
            max_n_nodes = max(nb_nodes_per_graph)
            max_n_edges = max(nb_edges_per_graph)
            for i_sample, s in enumerate(samples):
                if (
                    (nb_nodes_per_graph[i_sample] < max_n_nodes)  # nodes to be padded
                    or (
                        nb_edges_per_graph[i_sample] < max_n_edges
                    )  # edges to be padded
                    or (
                        nb_edge_links_per_graph[i_sample]
                        <= nb_edges_per_graph[i_sample]
                    )  # graph not yet padded
                ):
                    pad_batched_graph_dict(
                        s,
                        keys=keys,
                        max_n_nodes=max_n_nodes,
                        max_n_edges=max_n_edges,
                        has_batch_dim=has_batch_dim,
                    )
    elif is_masked_obs(first_subobs):
        # pad "true_obs" part
        pad_sample_batches_obs(
            samples=samples, keys=keys + (TRUE_OBS,), has_batch_dim=has_batch_dim
        )
        # pad action mask
        pad_sample_batches_action_mask(
            samples=samples, keys=keys + (ACTION_MASK,), has_batch_dim=has_batch_dim
        )
    elif is_applicable_actions_obs(first_subobs):
        # pad "true_obs" part
        pad_sample_batches_obs(
            samples=samples,
            keys=keys + (AUTOREGRESSIVE_TRUE_OBS,),
            has_batch_dim=has_batch_dim,
        )
        # pad applicable actions
        pad_sample_batches_applicable_actions(
            samples=samples,
            keys=keys + (APPLICABLE_ACTIONS,),
            has_batch_dim=has_batch_dim,
        )

    elif is_graph_dict_multiinput(first_subobs):
        # pad each subobs (that are graphs)
        for subkey in first_subobs:
            pad_sample_batches_obs(
                samples=samples, keys=keys + (subkey,), has_batch_dim=has_batch_dim
            )
    else:
        # not a graph => nothing to pad
        ...
