#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.base import Encoder, Model
from ray.rllib.core.models.catalog import Catalog

from skdecide.hub.solver.ray_rllib.gnn.models.catalog import GraphCatalog
from skdecide.hub.solver.ray_rllib.gnn.models.configs import Graph2NodeConfig


class GraphPPOCatalog(GraphCatalog, PPOCatalog): ...


class Graph2NodePPOCatalog(GraphPPOCatalog):
    def build_encoder(self, framework: str) -> Encoder:
        """Build encoder.

        Will only be used for critic as actor directly use a GNN via the action_net.

        """
        return Catalog.build_encoder(self, framework=framework)

    def build_action_net(self, framework: str) -> Model:
        graph2node_kwargs = self._model_config_dict.get(
            "graph2node_action_net_kwargs", {}
        )
        self.action_net_config = Graph2NodeConfig(
            observation_space=self.observation_space,
            graph2node_kwargs=graph2node_kwargs,
        )
        return self.action_net_config.build(framework)
