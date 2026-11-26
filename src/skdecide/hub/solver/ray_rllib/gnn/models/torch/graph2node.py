#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import TYPE_CHECKING

from ray.rllib.core import Columns
from ray.rllib.core.models.torch.base import TorchModel

from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    NODES,
    convert_dict_space_to_graph_space,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.torch_utils import (
    batched_torch_graph_dict_to_thg_data,
)
from skdecide.hub.solver.utils.gnn.torch_layers import Graph2NodeLayer
from skdecide.hub.solver.utils.gnn.torch_utils import unbatch_node_logits

if TYPE_CHECKING:
    from skdecide.hub.solver.ray_rllib.gnn.models.configs import Graph2NodeConfig

GRAPH_EMBEDDINGS = "graph_embeddings"


class TorchGraph2Node(TorchModel):
    config: Graph2NodeConfig

    def __init__(self, config: Graph2NodeConfig):
        super().__init__(config)
        observation_space = convert_dict_space_to_graph_space(
            self.config.observation_space
        )
        self.layer = Graph2NodeLayer(
            observation_space=observation_space, **self.config.graph2node_kwargs
        )
        self.node_flag_index = self.config.node_flag_index

    def _forward(
        self, input_dict: dict, i_component: int | None = None, **kwargs
    ) -> dict:
        observations = input_dict[Columns.OBS]
        # convert to thg.data.Batch
        graph_observations = batched_torch_graph_dict_to_thg_data(observations)
        # apply gnn
        graph_embeddings = self.layer.forward(graph_observations)
        # filter nodes to keep
        if self.node_flag_index is not None:
            nodes_to_keep = graph_observations.x[:, self.node_flag_index].nonzero(
                as_tuple=True
            )[0]
        else:
            nodes_to_keep = None
        # pad node logits to match previously computed logits if any (e.g. in rl_module.forward_train)
        if Columns.ACTION_DIST_INPUTS in input_dict:
            if isinstance(i_component, int):
                # autoregressive setup, graph node component
                action_dist_inputs_per_component, action_component_masks = input_dict[
                    Columns.ACTION_DIST_INPUTS
                ]
                max_n_nodes = action_dist_inputs_per_component.shape[-1]
            else:
                # graph observation with graph node action
                max_n_nodes = input_dict[Columns.ACTION_DIST_INPUTS].shape[-1]
        else:
            max_n_nodes = None
        node_logits = unbatch_node_logits(
            graph_embeddings, max_n_nodes=max_n_nodes, nodes_to_keep=nodes_to_keep
        )
        output = {
            NODES: node_logits,
        }
        if self.config.keep_graph_embedding:
            output[GRAPH_EMBEDDINGS] = graph_embeddings
        return output
