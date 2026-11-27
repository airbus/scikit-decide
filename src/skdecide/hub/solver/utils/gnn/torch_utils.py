from math import prod
from typing import Optional, Union

import gymnasium as gym
import torch as th
import torch_geometric as thg
from torch.nn.functional import pad


def graph_instance_to_thg_data(
    graph: gym.spaces.GraphInstance,
    device: Optional[th.device] = None,
    pin_memory: bool = False,
) -> thg.data.Data:
    return torch_graph_tensors_to_thg_data(
        nodes=th.tensor(graph.nodes, dtype=th.float),
        edges=None if graph.edges is None else th.tensor(graph.edges, dtype=th.float),
        edge_links=th.tensor(
            [] if graph.edge_links is None else graph.edge_links, dtype=th.long
        ),
        device=device,
        pin_memory=pin_memory,
    )


def torch_graph_tensors_to_thg_data(
    nodes: th.Tensor,
    edges: th.Tensor | None,
    edge_links: th.Tensor,
    device: Optional[th.device] = None,
    pin_memory: bool = False,
) -> thg.data.Data:
    x = nodes.reshape((len(nodes), -1)).float()
    if edges is None:
        edge_attr = None
    else:
        edge_attr = edges.reshape((len(edges), int(prod(edges.shape[1:])))).float()
    edge_index = edge_links.long().t().contiguous().view(2, -1)
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


def unbatch_node_logits(
    data: thg.data.Data,
    nodes_to_keep: Optional[tuple[th.Tensor, ...]] = None,
    max_n_nodes: int | None = None,
) -> th.Tensor:
    """Extract node logits from graph embeddings as batched (padded) logits

    # Parameters
    data: graph with node embeddings
    nodes_to_keep: indices of filtered nodes
    max_n_nodes: max number of nodes in the resulting tensor.

    # Returns
    logits of shape (nb_sample, max_n_nodes)
    if the number of graph nodes differ for each sample, we pad with -infty (or rather the lowest float value avialable)

    """
    x, batch = data.x, data.batch
    if nodes_to_keep is not None:
        # use only some nodes according to nodes_to_keep
        x = x[nodes_to_keep]
        if batch is not None:
            batch = batch[nodes_to_keep]
    if batch is None:
        node_logits = x.flatten()
        if max_n_nodes is not None and len(node_logits) < max_n_nodes:
            minus_infty_approx = th.finfo(x.dtype).min
            node_logits = pad(
                node_logits,
                (0, max_n_nodes - len(node_logits)),
                value=minus_infty_approx,
            )
    else:
        x_split = thg.utils.unbatch(x.flatten(), batch)
        max_n_nodes_x = max(len(xx) for xx in x_split)
        if max_n_nodes is None:
            max_n_nodes = max_n_nodes_x
        else:
            max_n_nodes = max(max_n_nodes_x, max_n_nodes)
        # we pad with -inf the logits (to avoid sampling node index higher than actual node number)
        # for stability issues (in particular in backprop), we approximate -inf with min float
        minus_infty_approx = th.finfo(x.dtype).min
        node_logits = th.stack(
            tuple(
                pad(xx, (0, max_n_nodes - len(xx)), value=minus_infty_approx)
                if len(xx) < max_n_nodes
                else xx
                for xx in x_split
            )
        )
    return node_logits
