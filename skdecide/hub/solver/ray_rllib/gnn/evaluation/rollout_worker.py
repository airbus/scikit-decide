from ray.rllib import RolloutWorker

from skdecide.hub.solver.ray_rllib.gnn.utils.monkey_patch import (
    monkey_patch_rllib_for_graph,
)


class GraphRolloutWorker(RolloutWorker):
    graph2node: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        monkey_patch_rllib_for_graph(graph2node=self.graph2node)


class Graph2NodeRolloutWorker(GraphRolloutWorker):
    graph2node = True
