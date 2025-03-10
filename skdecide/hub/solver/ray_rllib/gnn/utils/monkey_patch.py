from skdecide.hub.solver.ray_rllib.gnn.evaluation.collectors.agent_collector import (
    monkey_patch_agent_collector,
    unmonkey_patch_agent_collector,
)
from skdecide.hub.solver.ray_rllib.gnn.policy.sample_batch_monkey_patch import (
    monkey_patch_concat_samples,
    unmonkey_patch_concat_samples,
)


def monkey_patch_rllib_for_graph(graph2node: bool = False):
    monkey_patch_agent_collector(graph2node=graph2node)
    monkey_patch_concat_samples(graph2node=graph2node)


def unmonkey_patch_rllib_for_graph():
    unmonkey_patch_agent_collector()
    unmonkey_patch_concat_samples()
