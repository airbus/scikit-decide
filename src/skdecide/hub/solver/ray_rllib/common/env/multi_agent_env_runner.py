#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner

from skdecide.hub.solver.ray_rllib.common.utils.monkey_patch import (
    monkey_patch_rllib_for_skdecide,
)


class SkdecideMultiAgentEnvRunner(MultiAgentEnvRunner):
    graph2node: bool = False
    autoregressive_action: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        monkey_patch_rllib_for_skdecide(
            graph2node=self.graph2node, autoregressive_action=self.autoregressive_action
        )


class Graph2NodeMultiAgentEnvRunner(SkdecideMultiAgentEnvRunner):
    graph2node = True


class AutoregressiveMultiAgentEnvRunner(SkdecideMultiAgentEnvRunner):
    autoregressive_action = True


class AutoregressiveGraph2NodeMultiAgentEnvRunner(
    AutoregressiveMultiAgentEnvRunner, Graph2NodeMultiAgentEnvRunner
): ...
