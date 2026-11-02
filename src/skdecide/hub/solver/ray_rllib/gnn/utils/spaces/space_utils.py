from __future__ import annotations

from typing import Any, Union

import gymnasium as gym
import numpy as np
import tree
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.spaces.space_utils import BatchedNdArray

from skdecide.hub.solver.ray_rllib.action_masking.utils.spaces.space_utils import (
    ACTION_MASK,
    TRUE_OBS,
    is_masked_obs,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.batch_original_code import (
    original_batch,
)

NODES = "nodes"
EDGES = "edges"
EDGE_LINKS = "edge_links"

DEFAULT_N_NODES = (
    2  # initial nodes number to be used by dummy samples generation by rllib
)
DEFAULT_N_EDGES = (
    1  # initial edges number to be used by dummy samples generation by rllib
)


def convert_graph_space_to_dict_space(space: gym.spaces.Graph) -> gym.spaces.Dict:
    converted_node_space = Repeated(space.node_space, max_len=DEFAULT_N_NODES)
    if space.edge_space is None:
        converted_edge_space = Repeated(
            gym.spaces.Box(low=0, high=0, shape=(0,)), max_len=DEFAULT_N_EDGES
        )
    else:
        converted_edge_space = Repeated(space.edge_space, max_len=DEFAULT_N_EDGES)
    converted_edge_links_space = Repeated(
        gym.spaces.Box(low=0, high=DEFAULT_N_NODES - 1, shape=(2,), dtype=np.int_),
        max_len=DEFAULT_N_EDGES,
    )
    # add shapes for `get_dummy_batch_for_space()`
    _add_dummy_shape_to_repeated_space(converted_node_space)
    _add_dummy_shape_to_repeated_space(converted_edge_space)
    _add_dummy_shape_to_repeated_space(converted_edge_links_space)
    return gym.spaces.Dict(
        dict(
            nodes=converted_node_space,
            edges=converted_edge_space,
            edge_links=converted_edge_links_space,
        )
    )


def _add_dummy_shape_to_repeated_space(space: Repeated) -> None:
    n_rep = space.max_len
    if isinstance(space.child_space, gym.spaces.Box):
        space._shape = (n_rep,) + space.child_space.shape
    elif isinstance(space.child_space, gym.spaces.Discrete):
        space._shape = (n_rep, 1)
    else:
        raise NotImplementedError()


def pad_axis(
    x: np.ndarray, max_dim: int, value: Union[int, float] = 0, axis: int = 0
) -> np.ndarray:
    actual_dim = x.shape[axis]
    pad_width = np.zeros((len(x.shape), 2), dtype=int)
    pad_width[axis, 1] = max_dim - actual_dim
    return np.pad(
        x,
        pad_width=pad_width,
        constant_values=value,
    )


def convert_graph_to_dict(
    x: gym.spaces.GraphInstance, max_n_nodes: int = 0, max_n_edges: int = 0
) -> dict[str, np.ndarray]:
    nodes = x.nodes
    edges = x.edges
    edge_links = x.edge_links

    # no edge features? convert None into empty array tractable by rllib
    if edges is None:
        edges = np.empty(shape=(0,), dtype=nodes.dtype)

    # pad arrays? (necessary for rllib buffers that assume identical shapes)
    padding = max_n_nodes > 0 and max_n_edges > 0
    if padding:
        nodes, edges, edge_links = pad_graph(
            nodes=nodes,
            edges=edges,
            edge_links=edge_links,
            max_n_nodes=max_n_nodes,
            max_n_edges=max_n_edges,
        )

    return dict(
        nodes=nodes,
        edges=edges,
        edge_links=edge_links,
    )


def pad_graph(
    nodes: np.ndarray,
    edges: np.ndarray,
    edge_links: np.ndarray,
    max_n_nodes: int,
    max_n_edges: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual_n_nodes = len(nodes)
    # pad to max number of nodes + edges and encode actual number of edges and nodes in edge_links
    nodes = pad_axis(nodes, max_n_nodes)
    edges = pad_axis(edges, max_n_edges)
    # pad edge_links to max number of edges + 1 last "edge" encoding actual nodes number
    edge_links = pad_axis(
        edge_links, max_n_edges, value=-1
    )  # negative values => fake edges
    edge_links = np.vstack(
        (
            edge_links,
            (0, -actual_n_nodes),
        )  # actual nodes number (negative value => fake edge)
    )
    return nodes, edges, edge_links


def pad_batched_graph_dict(
    sample: dict[str, np.ndarray],
    max_n_nodes: int,
    max_n_edges: int,
    keys: tuple[str, ...] | None = None,
    has_batch_dim: bool = True,
) -> None:
    """Pad inplace graph dicts to same number of nodes and edges.

    Encode actual number of nodes in edge links.

    """
    if keys is None:
        keys = tuple()
    node_edge_id_dim = 1 if has_batch_dim else 0
    x = get_item(sample, keys)
    nodes, edges, edge_links = x[NODES], x[EDGES], x[EDGE_LINKS]
    assert isinstance(edge_links, np.ndarray) and len(edge_links.shape) == (
        int(has_batch_dim) + 2
    )  # batch, nb_edges, edge_nodes

    # edge links: padding with -1 (easy to recognize fake edges) + last edge encoding actual node numbers
    if (edge_links < 0).any():
        # already padded => keep last fake edge at last position (encoding node number)
        assert edge_links.shape[node_edge_id_dim] == edges.shape[node_edge_id_dim] + 1
        if has_batch_dim:
            encoding_n_nodes_edge_links = edge_links[
                :, None, -1, :
            ]  # keep all dimensions
            edge_links = edge_links[:, :-1, :]  # drop last edge
        else:
            encoding_n_nodes_edge_links = edge_links[None, -1, :]  # keep all dimensions
            edge_links = edge_links[:-1, :]  # drop last edge
    else:
        # not padded => all graphs have same number of nodes (and edges)
        assert edge_links.shape[node_edge_id_dim] == edges.shape[node_edge_id_dim]
        actual_n_nodes = nodes.shape[node_edge_id_dim]
        if has_batch_dim:
            encoding_n_nodes_edge_links = np.zeros(
                (edge_links.shape[0], 1, 2), dtype=int
            )
            encoding_n_nodes_edge_links[:, :, -1] = -actual_n_nodes
        else:
            encoding_n_nodes_edge_links = np.zeros((1, 2), dtype=int)
            encoding_n_nodes_edge_links[:, -1] = -actual_n_nodes

    edge_links = np.concatenate(
        (
            pad_axis(edge_links, max_n_edges, value=-1, axis=node_edge_id_dim),
            encoding_n_nodes_edge_links,
        ),
        axis=node_edge_id_dim,
    ).view(type(edge_links))  # keep BatchedNDArray wrapper if present

    # nodes and edges: pad with 0
    nodes = pad_axis(nodes, max_n_nodes, axis=node_edge_id_dim).view(
        type(nodes)
    )  # keep BatchedNDArray wrapper if present
    edges = pad_axis(edges, max_n_edges, axis=node_edge_id_dim).view(
        type(edges)
    )  # keep BatchedNDArray wrapper if present

    # set new padded nodes, edges and edge_links
    x[NODES] = nodes
    x[EDGES] = edges
    x[EDGE_LINKS] = edge_links


def convert_dict_space_to_graph_space(space: gym.spaces.Dict) -> gym.spaces.Graph:
    return gym.spaces.Graph(
        node_space=space.spaces[NODES].child_space,
        edge_space=space.spaces[EDGES].child_space,
    )


def convert_dict_to_graph(x: dict[str, np.ndarray]) -> gym.spaces.GraphInstance:
    nodes = x[NODES]
    edges = x[EDGES]
    edge_links = x[EDGE_LINKS]

    # was padded?
    if (
        len(edge_links) > 0 and edge_links[-1, 1] < 0
    ):  # represents -n_nodes when padding
        # get actual number of nodes and edges
        n_nodes = -int(edge_links[-1, 1])
        n_edges = sum((edge_links >= 0).all(axis=1))
        # extract true nodes, edges, edge_links
        nodes = nodes[:n_nodes]
        edges = edges[:n_edges]
        edge_links = edge_links[:n_edges]

    return gym.spaces.GraphInstance(nodes=nodes, edges=edges, edge_links=edge_links)


def is_graph_dict(x: Any) -> bool:
    return (
        isinstance(x, dict)
        and len(x) == 3
        and NODES in x
        and EDGES in x
        and EDGE_LINKS in x
    )


def is_graph_dict_space(space: gym.spaces.Space) -> bool:
    return (
        isinstance(space, gym.spaces.Dict)
        and len(space.spaces) == 3
        and NODES in space.spaces
        and EDGES in space.spaces
        and EDGE_LINKS in space.spaces
    )


def is_graph_dict_multiinput(x: Any) -> bool:
    return isinstance(x, dict) and any([is_graph_dict(v) for v in x.values()])


def is_graph_dict_multiinput_space(space: gym.spaces.Space) -> bool:
    return isinstance(space, gym.spaces.Dict) and any(
        [is_graph_dict_space(subspace) for subspace in space.values()]
    )


def prepare_for_batch_graph(
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


def batch_graph(
    list_of_structs: list[Any],
    *,
    individual_items_already_have_batch_dim: bool | str = False,
) -> Any:
    """Enhance `ray.rllib.utils.spaces.space_utils.batch` to manage graphs."""
    prepare_for_batch_graph(
        list_of_structs,
        individual_items_already_have_batch_dim=individual_items_already_have_batch_dim,
    )
    return original_batch(
        list_of_structs=list_of_structs,
        individual_items_already_have_batch_dim=individual_items_already_have_batch_dim,
    )


def get_item(s: dict[str, Any], keys: tuple[str, ...]) -> Any:
    if len(keys) == 0:
        return s
    else:
        return get_item(s[keys[0]], keys[1:])


def set_item(s: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    if len(keys) == 0:
        raise ValueError("keys must of len >=1")
    elif len(keys) == 1:
        s[keys[0]] = value
    else:
        set_item(s[keys[0]], keys=keys[1:], value=value)


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
    elif is_graph_dict_multiinput(first_subobs):
        # pad each subobs (that are graphs)
        for subkey in first_subobs:
            pad_sample_batches_obs(
                samples=samples, keys=keys + (subkey,), has_batch_dim=has_batch_dim
            )
    else:
        # not a graph => nothing to pad
        ...


def pad_sample_batches_action_mask(
    samples: list[dict[str, Any]], keys: tuple[str, ...], has_batch_dim: bool = True
) -> None:
    node_edge_id_dim = 1 if has_batch_dim else 0
    if len(set(get_item(s, keys).shape[node_edge_id_dim] for s in samples)) > 1:
        # different number of nodes => padding
        max_n_nodes = max(get_item(s, keys).shape[node_edge_id_dim] for s in samples)
        for s in samples:
            action_mask = get_item(s, keys)
            set_item(
                s,
                keys=keys,
                value=pad_axis(action_mask, max_n_nodes, axis=node_edge_id_dim).view(
                    type(action_mask)
                ),  # keep BatchedNDArray wrapper if present
            )
