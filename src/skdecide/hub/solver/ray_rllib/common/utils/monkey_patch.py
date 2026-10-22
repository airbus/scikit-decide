#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from skdecide.hub.solver.ray_rllib.common.env.single_agent_episode import (
    monkey_patch_single_agent_episode,
    unmonkey_patch_single_agent_episode,
)
from skdecide.hub.solver.ray_rllib.common.utils.spaces.batch_monkey_patch import (
    monkey_patch_batch,
    unmonkey_patch_batch,
)


def monkey_patch_rllib_for_skdecide(
    graph2node: bool = False, autoregressive_action: bool = False
):
    monkey_patch_batch()
    monkey_patch_single_agent_episode(
        graph2node=graph2node, autoregressive_action=autoregressive_action
    )


def unmonkey_patch_rllib_for_skdecide():
    unmonkey_patch_batch()
    unmonkey_patch_single_agent_episode()
