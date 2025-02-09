from typing import Any, Union

import gymnasium as gym
import numpy as np


def convert_graph_space_to_dict_space(space: gym.spaces.Graph) -> gym.spaces.Dict:
    # artificially decide of 2 nodes and 1 edge (for dummy samples auto-generated by ray.rllib)
    return gym.spaces.Dict(
        dict(
            nodes=repeat_space(space.node_space, n_rep=2),
            edges=repeat_space(space.edge_space, n_rep=1),
            edge_links=gym.spaces.Box(low=0, high=2, shape=(1, 2), dtype=np.int_),
        )
    )


def repeat_space_box(space: gym.spaces.Box, n_rep: int) -> gym.spaces.Box:
    rep_low = np.repeat(space.low[None, :], n_rep, axis=0)
    rep_high = np.repeat(space.high[None, :], n_rep, axis=0)
    rep_shape = (n_rep,) + space.shape
    return gym.spaces.Box(
        low=rep_low,
        high=rep_high,
        shape=rep_shape,
        dtype=space.dtype,
    )


def repeat_space_discrete(space: gym.spaces.Discrete, n_rep: int) -> gym.spaces.Box:
    return gym.spaces.Box(
        low=space.start,
        high=space.start + space.n - 1,
        shape=(n_rep, 1),
        dtype=space.dtype,
    )


def repeat_space(space: Union[gym.spaces.Box, gym.spaces.Discrete], n_rep: int):
    if isinstance(space, gym.spaces.Box):
        return repeat_space_box(space=space, n_rep=n_rep)
    elif isinstance(space, gym.spaces.Discrete):
        return repeat_space_discrete(space=space, n_rep=n_rep)
    else:
        raise NotImplementedError()


def remove_first_axis_space(space: gym.spaces.Box) -> gym.spaces.Box:
    return gym.spaces.Box(
        low=space.low[0, :],
        high=space.high[0, :],
        shape=space.shape[1:],
        dtype=space.dtype,
    )


def convert_graph_to_dict(x: gym.spaces.GraphInstance) -> dict[str, np.ndarray]:
    return dict(
        nodes=x.nodes,
        edges=x.edges,
        edge_links=x.edge_links,
    )


def convert_dict_space_to_graph_space(space: gym.spaces.Dict) -> gym.spaces.Graph:
    return gym.spaces.Graph(
        node_space=remove_first_axis_space(space.spaces["nodes"]),
        edge_space=remove_first_axis_space(space.spaces["edges"]),
    )


def convert_dict_to_graph(x: dict[str, np.ndarray]) -> gym.spaces.GraphInstance:
    return gym.spaces.GraphInstance(
        nodes=x["nodes"], edges=x["edges"], edge_links=x["edge_links"]
    )


def is_graph_dict(x: Any) -> bool:
    return (
        isinstance(x, dict)
        and len(x) == 3
        and "nodes" in x
        and "edges" in x
        and "edge_links" in x
    )


def is_graph_dict_space(x: gym.spaces.Space) -> bool:
    return (
        isinstance(x, gym.spaces.Dict)
        and len(x.spaces) == 3
        and "nodes" in x.spaces
        and "edges" in x.spaces
        and "edge_links" in x.spaces
    )


def is_graph_dict_multiinput(x: Any) -> bool:
    return isinstance(x, dict) and any([is_graph_dict(v) for v in x.values()])


def is_graph_dict_multiinput_space(x: gym.spaces.Space) -> bool:
    return isinstance(x, gym.spaces.Dict) and any(
        [is_graph_dict_space(subspace) for subspace in x.values()]
    )


def is_masked_obs(x: Any) -> bool:
    return (
        isinstance(x, dict)
        and len(x) == 2
        and "true_obs" in x
        and "valid_avail_actions_mask" in x
    )


def is_masked_obs_space(x: gym.spaces.Space) -> bool:
    return (
        isinstance(x, gym.spaces.Dict)
        and len(x.spaces) == 2
        and "true_obs" in x.spaces
        and "valid_avail_actions_mask" in x.spaces
    )


def extract_graph_dict_from_batched_graph_dict(
    batched_graph_dict: dict[str, np.ndarray], index: int
) -> dict[str, np.ndarray]:
    return {k: v[index, :] for k, v in batched_graph_dict.items()}
