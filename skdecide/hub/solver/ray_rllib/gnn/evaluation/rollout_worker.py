from ray.rllib import RolloutWorker

from .collectors.agent_collector import monkey_patch_agent_collector
from .collectors.simple_list_collector import monkey_patch_policy_collector


class GraphRolloutWorker(RolloutWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        monkey_patch_agent_collector()
        monkey_patch_policy_collector()


class Graph2NodeRolloutWorker(GraphRolloutWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        monkey_patch_agent_collector(graph2node=True)
        monkey_patch_policy_collector(graph2node=True)
