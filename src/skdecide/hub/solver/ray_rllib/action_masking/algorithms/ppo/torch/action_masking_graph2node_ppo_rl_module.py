#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from skdecide.hub.solver.ray_rllib.action_masking.algorithms.ppo.torch.action_masking_ppo_torch_rl_module import (
    ActionMaskingPPOTorchRLModule,
)
from skdecide.hub.solver.ray_rllib.gnn.algorithms.ppo.torch.graph2node_ppo_torch_rl_module import (
    Graph2NodePPOTorchRLModule,
)


class ActionMaskingGraph2NodePPOTorchRLModule(
    ActionMaskingPPOTorchRLModule, Graph2NodePPOTorchRLModule
): ...
