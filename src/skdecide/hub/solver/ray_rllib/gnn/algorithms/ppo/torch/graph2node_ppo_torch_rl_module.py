#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Dict

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core import Columns
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.utils.typing import TensorType

from skdecide.hub.solver.ray_rllib.gnn.algorithms.ppo.ppo_catalog import (
    Graph2NodePPOCatalog,
)
from skdecide.hub.solver.ray_rllib.gnn.rl_module.torch.graph2node_torch_rl_module import (
    Graph2NodeTorchRLModule,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import NODES


class Graph2NodePPOTorchRLModule(DefaultPPOTorchRLModule, Graph2NodeTorchRLModule):
    catalog: Graph2NodePPOCatalog

    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = Graph2NodePPOCatalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

    def setup(self):
        if not self.inference_only:
            # encoder used for critic only so not for inference
            self.encoder = self.catalog.build_encoder(framework=self.framework)
            self.vf = self.catalog.build_vf_head(framework=self.framework)
        self.action_net = self.catalog.build_action_net(framework=self.framework)

    def get_non_inference_attributes(self) -> list[str]:
        return ["encoder", "vf"]

    def _forward(self, batch: dict[str, TensorType], **kwargs) -> dict[str, TensorType]:
        return _forward_graph2node_ppo(self, batch, **kwargs)

    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Cannot simply call `self._forward()`
        # Else, when derived in `ActionMaskingGraph2NodePPOTorchRLModule`,
        # `super()._forward_train() will call `self._forward()` which will be `ActionMaskingPPOTorchRLModule._forward()`
        # instead of `Graph2NodePPOTorchRLModule._forward()`
        return _forward_graph2node_ppo(self, batch, **kwargs)

    def compute_values(self, batch: dict[str, TensorType], embeddings=None):
        if embeddings is None:
            embeddings = self.encoder(batch)[ENCODER_OUT]
        # Call the super's method to compute values for GAE.
        return super().compute_values(batch, embeddings)


def _forward_graph2node_ppo(
    self: Graph2NodePPOTorchRLModule, batch: dict[str, TensorType], **kwargs
) -> dict[str, TensorType]:
    output = {}
    action_net_outs = self.action_net(batch)
    output[Columns.ACTION_DIST_INPUTS] = action_net_outs[NODES]
    return output
