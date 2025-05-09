from typing import Any, Union

import gymnasium as gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated

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
    # pad arrays? (necessary for rllib buffers that assume identical shapes)
    padding = max_n_nodes > 0 and max_n_edges > 0

    if padding:
        nodes, edges, edge_links = pad_graph(
            nodes=x.nodes,
            edges=x.edges,
            edge_links=x.edge_links,
            max_n_nodes=max_n_nodes,
            max_n_edges=max_n_edges,
        )
    else:
        # no padding
        nodes = x.nodes
        edges = x.edges
        edge_links = x.edge_links

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
    x: dict[str, np.ndarray], max_n_nodes: int, max_n_edges: int
) -> dict[str, np.ndarray]:
    nodes, edges, edge_links = x[NODES], x[EDGES], x[EDGE_LINKS]
    assert (
        isinstance(edge_links, np.ndarray) and len(edge_links.shape) == 3
    )  # batch, nb_edges, edge_nodes

    # edge links: padding with -1 (easy to recognize fake edges) + last edge encoding actual node numbers
    if (edge_links < 0).any():
        # already padded => keep last fake edge at last position (encoding node number)
        assert edge_links.shape[1] == edges.shape[1] + 1
        encoding_n_nodes_edge_links = edge_links[:, None, -1, :]  # keep all dimensions
        edge_links = edge_links[:, :-1, :]  # drop last edge
    else:
        # not padded => all graphs have same number of nodes (and edges)
        assert edge_links.shape[1] == edges.shape[1]
        actual_n_nodes = nodes.shape[1]
        encoding_n_nodes_edge_links = np.zeros((edge_links.shape[0], 1, 2), dtype=int)
        encoding_n_nodes_edge_links[:, :, -1] = -actual_n_nodes

    edge_links = np.concatenate(
        (
            pad_axis(edge_links, max_n_edges, value=-1, axis=1),
            encoding_n_nodes_edge_links,
        ),
        axis=1,
    )

    # nodes and edges: pad with 0 second axis (first axis = batch)
    nodes = pad_axis(nodes, max_n_nodes, axis=1)
    edges = pad_axis(edges, max_n_edges, axis=1)

    return dict(
        nodes=nodes,
        edges=edges,
        edge_links=edge_links,
    )


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


def extract_graph_dict_from_batched_graph_dict(
    batched_graph_dict: dict[str, np.ndarray], index: int
) -> dict[str, np.ndarray]:
    return {k: v[index, :] for k, v in batched_graph_dict.items()}
