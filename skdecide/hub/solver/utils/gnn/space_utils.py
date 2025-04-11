import gymnasium as gym
import numpy as np


def add_onehot_encoded_discrete_feature_to_graph_space(
    graph_space: gym.spaces.Graph, new_dim: int
) -> gym.spaces.Graph:
    node_space = graph_space.node_space
    if isinstance(node_space, gym.spaces.Box):
        node_space = add_onehot_encoded_discrete_feature_to_box_space(
            node_space, new_dim
        )
    else:
        raise NotImplementedError()
    return gym.spaces.Graph(
        node_space=node_space,
        edge_space=graph_space.edge_space,
    )


def add_onehot_encoded_discrete_feature_to_box_space(
    box_space: gym.spaces.Box, new_dim: int
) -> gym.spaces.Box:
    low = np.concatenate((box_space.low.flatten(), np.array([0] * new_dim)))
    high = np.concatenate((box_space.high.flatten(), np.array([1] * new_dim)))
    return gym.spaces.Box(low=low, high=high, dtype=box_space.dtype)
