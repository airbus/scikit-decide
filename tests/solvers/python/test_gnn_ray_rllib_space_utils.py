import gymnasium as gym
import numpy as np
import pytest

from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    convert_dict_to_graph,
    convert_graph_to_dict,
    pad_batched_graph_dict,
)


def batch_graph_dicts(graph_dicts):
    return dict(
        nodes=np.stack(tuple(g["nodes"] for g in graph_dicts), axis=0),
        edges=np.stack(tuple(g["edges"] for g in graph_dicts), axis=0),
        edge_links=np.stack(tuple(g["edge_links"] for g in graph_dicts), axis=0),
    )


def unbatch_graph_dict(batched_graph_dict):
    n_batch = len(batched_graph_dict["nodes"])
    return [
        dict(
            nodes=batched_graph_dict["nodes"][i_batch],
            edges=batched_graph_dict["edges"][i_batch],
            edge_links=batched_graph_dict["edge_links"][i_batch],
        )
        for i_batch in range(n_batch)
    ]


def are_equal_graphs(
    g1: gym.spaces.GraphInstance, g2: gym.spaces.GraphInstance
) -> bool:
    return all(np.all(a == b) for a, b in zip(g1, g2))


@pytest.fixture
def graphs():
    return [
        gym.spaces.GraphInstance(
            nodes=np.reshape(range(6), (2, 3)),
            edges=np.reshape([1.0], (1, 1)),
            edge_links=np.reshape([0, 1], (1, 2)),
        ),
        gym.spaces.GraphInstance(
            nodes=np.reshape(range(9), (3, 3)),
            edges=np.reshape([1.0, 2.0], (2, 1)),
            edge_links=np.reshape([[0, 1], [1, 2]], (2, 2)),
        ),
    ]


@pytest.fixture
def samestruct_graphs():
    return [
        gym.spaces.GraphInstance(
            nodes=np.reshape(range(6, 0, -1), (2, 3)),
            edges=np.reshape([1.0], (1, 1)),
            edge_links=np.reshape([0, 1], (1, 2)),
        ),
        gym.spaces.GraphInstance(
            nodes=np.reshape(range(6), (2, 3)),
            edges=np.reshape([1.0], (1, 1)),
            edge_links=np.reshape([[0, 1]], (1, 2)),
        ),
    ]


def test_convert_graph_dict_wo_padding(graphs):
    assert all(
        are_equal_graphs(g, convert_dict_to_graph(convert_graph_to_dict(g)))
        for g in graphs
    )


def test_convert_graph_dict_with_padding(graphs):
    max_n_nodes = 4
    max_n_edges = 3
    node_space_shape = graphs[0].nodes.shape[1:]
    edge_space_shape = graphs[0].edges.shape[1:]
    padded_graph_dicts = [
        convert_graph_to_dict(g, max_n_edges=max_n_edges, max_n_nodes=max_n_nodes)
        for g in graphs
    ]
    for gd in padded_graph_dicts:
        assert gd["nodes"].shape == (max_n_nodes,) + node_space_shape
        assert gd["edges"].shape == (max_n_edges,) + edge_space_shape
        assert gd["edge_links"].shape == (max_n_edges + 1, 2)
    assert all(
        are_equal_graphs(g, convert_dict_to_graph(gd))
        for g, gd in zip(graphs, padded_graph_dicts)
    )


def test_pad_batched_dict_graph_wo_prepadding(samestruct_graphs):
    graphs = samestruct_graphs
    n_batch = len(graphs)
    max_n_nodes = 5
    max_n_edges = 5
    node_space_shape = graphs[0].nodes.shape[1:]
    edge_space_shape = graphs[0].edges.shape[1:]

    batched_dict_graph = batch_graph_dicts([convert_graph_to_dict(g) for g in graphs])
    padded_batched_graph_dict = pad_batched_graph_dict(
        batched_dict_graph, max_n_nodes=max_n_nodes, max_n_edges=max_n_edges
    )

    assert (
        padded_batched_graph_dict["nodes"].shape
        == (n_batch, max_n_nodes) + node_space_shape
    )
    assert (
        padded_batched_graph_dict["edges"].shape
        == (n_batch, max_n_edges) + edge_space_shape
    )
    assert padded_batched_graph_dict["edge_links"].shape == (
        n_batch,
        max_n_edges + 1,
    ) + (2,)

    assert all(
        are_equal_graphs(g, convert_dict_to_graph(gd))
        for g, gd in zip(graphs, unbatch_graph_dict(padded_batched_graph_dict))
    )


def test_pad_batched_dict_graph_with_prepadding(graphs):
    n_batch = len(graphs)
    max_n_nodes_prepadding = 4
    max_n_edges_prepadding = 3
    max_n_nodes = 5
    max_n_edges = 5
    node_space_shape = graphs[0].nodes.shape[1:]
    edge_space_shape = graphs[0].edges.shape[1:]

    batched_dict_graph = batch_graph_dicts(
        [
            convert_graph_to_dict(
                g,
                max_n_edges=max_n_edges_prepadding,
                max_n_nodes=max_n_nodes_prepadding,
            )
            for g in graphs
        ]
    )
    padded_batched_graph_dict = pad_batched_graph_dict(
        batched_dict_graph, max_n_nodes=max_n_nodes, max_n_edges=max_n_edges
    )

    assert (
        padded_batched_graph_dict["nodes"].shape
        == (n_batch, max_n_nodes) + node_space_shape
    )
    assert (
        padded_batched_graph_dict["edges"].shape
        == (n_batch, max_n_edges) + edge_space_shape
    )
    assert padded_batched_graph_dict["edge_links"].shape == (
        n_batch,
        max_n_edges + 1,
    ) + (2,)

    assert all(
        are_equal_graphs(g, convert_dict_to_graph(gd))
        for g, gd in zip(graphs, unbatch_graph_dict(padded_batched_graph_dict))
    )
