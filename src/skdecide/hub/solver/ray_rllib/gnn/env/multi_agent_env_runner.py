#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner

from skdecide.hub.solver.ray_rllib.gnn.utils.monkey_patch import (
    monkey_patch_rllib_for_graph,
)


class GraphMultiAgentEnvRunner(MultiAgentEnvRunner):
    graph2node: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        monkey_patch_rllib_for_graph(graph2node=self.graph2node)


class Graph2NodeMultiAgentEnvRunner(GraphMultiAgentEnvRunner):
    graph2node = True
