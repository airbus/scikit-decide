from typing import Optional

import gymnasium as gym
import torch as th
import torch_geometric as thg


def graph_obs_to_thg_data(
    obs: gym.spaces.GraphInstance,
    device: Optional[th.device] = None,
    pin_memory: bool = False,
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
    # thg.Data
    data = thg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # Pin the tensor's memory (for faster transfer to GPU later).
    if pin_memory and th.cuda.is_available():
        data.pin_memory()

    return data if device is None else data.to(device)
