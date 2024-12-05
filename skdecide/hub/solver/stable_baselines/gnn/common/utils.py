from typing import Union

import gymnasium as gym
import numpy as np
import torch as th
import torch_geometric as thg


def copy_graph_instance(g: gym.spaces.GraphInstance) -> gym.spaces.GraphInstance:
    return gym.spaces.GraphInstance(
        nodes=np.copy(g.nodes), edges=np.copy(g.edges), edge_links=np.copy(g.edge_links)
    )


def copy_np_array_or_list_of_graph_instances(
    obs: Union[np.ndarray, list[gym.spaces.GraphInstance]]
) -> Union[np.ndarray, list[gym.spaces.GraphInstance]]:
    if isinstance(obs[0], gym.spaces.GraphInstance):
        return [copy_graph_instance(g) for g in obs]
    else:
        return np.copy(obs)


def graph_obs_to_thg_data(
    obs: gym.spaces.GraphInstance, device: th.device
) -> thg.data.Data:
    # Node features
    flatten_node_features = obs.nodes.reshape((len(obs.nodes), -1))
    x = th.tensor(flatten_node_features).float()
    # Edge features
    if obs.edges is None:
        edge_attr = None
    else:
        flatten_edge_features = obs.edges.reshape((len(obs.edges), -1))
        edge_attr = th.tensor(flatten_edge_features).float()
    edge_index = th.tensor(obs.edge_links, dtype=th.long).t().contiguous().view(2, -1)
    return thg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)
