from skdecide.hub.solver.ray_rllib.gnn.env.single_agent_episode import (
    monkey_patch_single_agent_episode_for_graph2node,
    unmonkey_patch_single_agent_episode,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.batch_monkey_patch import (
    monkey_patch_batch,
    unmonkey_patch_batch,
)


def monkey_patch_rllib_for_graph(graph2node: bool = False):
    monkey_patch_batch(graph2node=graph2node)
    if graph2node:
        monkey_patch_single_agent_episode_for_graph2node()


def unmonkey_patch_rllib_for_graph():
    unmonkey_patch_batch()
    unmonkey_patch_single_agent_episode()
