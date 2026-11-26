from __future__ import annotations

from typing import Optional

import torch as th
import torch_geometric as thg

from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    EDGE_LINKS,
    EDGES,
    NODES,
)
from skdecide.hub.solver.utils.gnn.torch_utils import (
    torch_graph_tensors_to_thg_data,
)


def batched_torch_graph_dict_to_thg_data(
    batched_graph_dict: dict[str, th.Tensor],
    device: Optional[str] = None,
    pin_memory: bool = False,
) -> thg.data.Data:
    return thg.data.Batch.from_data_list(
        [
            torch_graph_tensors_to_thg_data(
                *unpad_torch_graph_tensors(
                    nodes=batched_graph_dict[NODES][index, :],
                    edges=batched_graph_dict[EDGES][index, :],
                    edge_links=batched_graph_dict[EDGE_LINKS][index, :],
                ),
                device=device,
                pin_memory=pin_memory,
            )
            for index in range(len(batched_graph_dict[NODES]))
        ]
    )


def unpad_torch_graph_tensors(
    nodes: th.Tensor,
    edges: th.Tensor | None,
    edge_links: th.Tensor,
) -> tuple[th.Tensor, th.Tensor | None, th.Tensor]:
    # was padded?
    if (
        len(edge_links) > 0 and edge_links[-1, 1] < 0
    ):  # represents -n_nodes when padding
        # get actual number of nodes and edges
        n_nodes = -int(edge_links[-1, 1])
        n_edges = sum((edge_links >= 0).all(axis=1))
        # extract true nodes, edges, edge_links
        nodes = nodes[:n_nodes]
        if edges is not None:
            edges = edges[:n_edges]
        edge_links = edge_links[:n_edges]

    return (nodes, edges, edge_links)
