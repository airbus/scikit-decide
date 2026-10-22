#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from ray.rllib.env.single_agent_episode import SingleAgentEpisode

from skdecide.hub.solver.ray_rllib.autoregressive.env.single_agent_episode import (
    autoregressive_graph2node_single_agent_episode_to_numpy,
)
from skdecide.hub.solver.ray_rllib.gnn.env.single_agent_episode import (
    graph2node_single_agent_episode_to_numpy,
)


def monkey_patch_single_agent_episode(
    autoregressive_action: bool, graph2node: bool
) -> None:
    if graph2node:
        if not hasattr(SingleAgentEpisode, "_to_numpy_unpatched"):
            # store true method only if not already done
            SingleAgentEpisode._to_numpy_unpatched = SingleAgentEpisode.to_numpy
        if autoregressive_action:
            SingleAgentEpisode.to_numpy = (
                autoregressive_graph2node_single_agent_episode_to_numpy
            )
        else:
            SingleAgentEpisode.to_numpy = graph2node_single_agent_episode_to_numpy


def unmonkey_patch_single_agent_episode() -> None:
    if hasattr(SingleAgentEpisode, "_to_numpy_unpatched"):
        SingleAgentEpisode.to_numpy = SingleAgentEpisode._to_numpy_unpatched
        del SingleAgentEpisode._to_numpy_unpatched
