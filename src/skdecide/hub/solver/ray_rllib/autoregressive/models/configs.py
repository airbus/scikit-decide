#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

from ray.rllib.core.models.configs import ModelConfig

from skdecide.hub.solver.ray_rllib.autoregressive.models.torch.encoder import (
    TorchOneHotEncoder,
)
from skdecide.hub.solver.ray_rllib.common.constants import TORCH_FRAMEWORK


@dataclass(kw_only=True)
class OneHotEncoderConfig(ModelConfig):
    """Configuration for one-hot encoder."""

    nb_classes: int

    def build(self, framework: str = TORCH_FRAMEWORK) -> TorchOneHotEncoder:
        if framework == TORCH_FRAMEWORK:
            return TorchOneHotEncoder(self)
        else:
            raise NotImplementedError()

    @property
    def output_dims(self) -> tuple[int, ...]:
        return (self.nb_classes,)
