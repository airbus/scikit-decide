from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch as th
import torch_geometric as thg
from torch import nn
from torch_geometric.nn import global_max_pool


class GraphFeaturesExtractor(nn.Module):
    """Graph feature extractor for Graph observation spaces.

    Will chain a gnn with a reduction layer to extract a fixed number of features.
    The user can specify both the gnn and reduction layer.

    By default, we use:
    - gnn: a 2-layers GCN
    - reduction layer: global_max_pool + linear layer + relu

    Args:
        observation_space:
        features_dim: Number of extracted features
            - If reduction_layer_class is given, should match the output of this network.
            - If reduction_layer is None, will be used by the default network as its output dimension.
        gnn_out_dim: dimension of the node embedding in gnn output
            - If gnn is given, should not be None and should match the output of gnn
            - If gnn is not given, will be used to generate it. By default, gnn_out_dim = 2 * features_dim
        gnn_class: GNN network class (for instance chosen from `torch_geometric.nn.models` used to embed the graph observations)
        gnn_kwargs: used by `gnn_class.__init__()`. Without effect if `gnn_class` is None.
        reduction_layer_class: network class to be plugged after the gnn to get a fixed number of features.
        reduction_layer_kwargs: used by `reduction_layer_class.__init__()`. Without effect if `reduction_layer_class` is None.

    """

    def __init__(
        self,
        observation_space: gym.spaces.Graph,
        features_dim: int = 64,
        gnn_out_dim: Optional[int] = None,
        gnn_class: Optional[type[nn.Module]] = None,
        gnn_kwargs: Optional[dict[str, Any]] = None,
        reduction_layer_class: Optional[type[nn.Module]] = None,
        reduction_layer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.features_dim = features_dim

        if gnn_out_dim is None:
            if gnn_class is None:
                gnn_out_dim = 2 * features_dim
            else:
                raise ValueError(
                    "`gnn_out_dim` cannot be None if `gnn` is not None, "
                    "and should match `gnn` output."
                )

        if gnn_class is None:
            node_features_dim = int(np.prod(observation_space.node_space.shape))
            self.gnn = thg.nn.models.GCN(
                in_channels=node_features_dim,
                hidden_channels=gnn_out_dim,
                num_layers=2,
                dropout=0.2,
            )
        else:
            if gnn_kwargs is None:
                gnn_kwargs = {}
            self.gnn = gnn_class(**gnn_kwargs)

        if reduction_layer_class is None:
            self.reduction_layer = _DefaultReductionLayer(
                gnn_out_dim=gnn_out_dim, features_dim=features_dim
            )
        else:
            if reduction_layer_kwargs is None:
                reduction_layer_kwargs = {}
            self.reduction_layer = reduction_layer_class(**reduction_layer_kwargs)

    def forward(self, observations) -> th.Tensor:
        x, edge_index, edge_attr, batch = (
            observations.x,
            observations.edge_index,
            observations.edge_attr,
            observations.batch,
        )
        # construct edge weights, for GNNs needing it, as the first edge feature
        edge_weight = edge_attr[:, 0]
        h = self.gnn(
            x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr
        )
        embedded_observations = thg.data.Data(
            x=h, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )
        h = self.reduction_layer(embedded_observations=embedded_observations)
        return h


class _DefaultReductionLayer(nn.Module):
    def __init__(self, gnn_out_dim: int, features_dim: int):
        super().__init__()
        self.gnn_out_dim = gnn_out_dim
        self.features_dim = features_dim
        self.linear_layer = nn.Linear(gnn_out_dim, features_dim)

    def forward(self, embedded_observations: thg.data.Data) -> th.Tensor:
        x, edge_index, batch = (
            embedded_observations.x,
            embedded_observations.edge_index,
            embedded_observations.batch,
        )
        h = global_max_pool(x, batch)
        h = self.linear_layer(h).relu()
        return h
