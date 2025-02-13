from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn
import torch_geometric as thg


def graph_instance_to_thg_data(
    graph: gym.spaces.GraphInstance,
    device: Optional[th.device] = None,
    pin_memory: bool = False,
) -> thg.data.Data:
    # Node features
    flatten_node_features = graph.nodes.reshape((len(graph.nodes), -1))
    x = th.tensor(flatten_node_features).float()
    # Edge features
    if graph.edges is None:
        edge_attr = None
    else:
        flatten_edge_features = graph.edges.reshape(
            (len(graph.edges), int(np.prod(graph.edges.shape[1:])))
        )
        edge_attr = th.tensor(flatten_edge_features).float()
    edge_index = th.tensor(graph.edge_links, dtype=th.long).t().contiguous().view(2, -1)
    # thg.Data
    data = thg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # Pin the tensor's memory (for faster transfer to GPU later).
    if pin_memory and th.cuda.is_available():
        data.pin_memory()

    return data if device is None else data.to(device)


def thg_data_to_graph_instance(
    data: thg.data.Data, space: gym.spaces.Graph, vectorized=True
) -> Union[gym.spaces.GraphInstance, list[gym.spaces.GraphInstance]]:
    nodes = data.x.cpu().numpy().reshape((-1, *space.node_space.shape))
    edges = data.edge_attr.cpu().numpy().reshape((-1, *space.edge_space.shape))
    edge_links = data.edge_index.cpu().numpy().transpose()
    batch = data.batch
    if batch is None:
        graph = gym.spaces.GraphInstance(
            nodes=nodes, edges=edges, edge_links=edge_links
        )
        if vectorized:
            return [graph]
        else:
            return graph
    else:
        raise NotImplementedError()


def extract_module_parameters_values(m: torch.nn.Module) -> dict[str, np.ndarray]:
    return {
        name: np.array(param.data.cpu().numpy()) for name, param in m.named_parameters()
    }
