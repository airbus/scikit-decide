from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated

from skdecide.hub.solver.ray_rllib.common.utils.spaces.utils import get_item, pad_axis
from skdecide.hub.solver.utils.gnn.space_utils import (
    add_onehot_encoded_discrete_feature_to_box_space,
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


def add_onehot_encoded_discrete_feature_to_graph_dict_space(
    graph_dict_space: gym.spaces.Dict, new_dim: int
) -> gym.spaces.Dict:
    node_space_repeated: Repeated = graph_dict_space[NODES]
    node_space = node_space_repeated.child_space
    if isinstance(node_space, gym.spaces.Box):
        enriched_node_space = add_onehot_encoded_discrete_feature_to_box_space(
            node_space, new_dim
        )
    else:
        raise NotImplementedError()
    return gym.spaces.Dict(
        {
            NODES: Repeated(
                child_space=enriched_node_space, max_len=node_space_repeated.max_len
            ),
            EDGES: graph_dict_space[EDGES],
            EDGE_LINKS: graph_dict_space[EDGE_LINKS],
        }
    )
