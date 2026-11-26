#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Type

from ray.rllib.core.distribution.torch.torch_distribution import TorchDistribution
from ray.rllib.core.rl_module.torch import TorchRLModule

from skdecide.hub.solver.ray_rllib.common.constants import TORCH_FRAMEWORK
from skdecide.hub.solver.ray_rllib.gnn.models.catalog import get_action_dist_cls


class Graph2NodeTorchRLModule(TorchRLModule):
    def get_inference_action_dist_cls(self) -> Type[TorchDistribution]:
        if self.action_dist_cls is not None:
            return self.action_dist_cls
        else:
            return get_action_dist_cls(
                framework=TORCH_FRAMEWORK, action_space=self.action_space
            )
