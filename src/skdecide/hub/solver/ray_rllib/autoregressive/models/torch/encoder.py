#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from ray.rllib.core import Columns
from ray.rllib.core.models.base import ENCODER_OUT, Encoder
from ray.rllib.core.models.torch.base import TorchModel

if TYPE_CHECKING:
    from skdecide.hub.solver.ray_rllib.autoregressive.models.configs import (
        OneHotEncoderConfig,
    )


class TorchOneHotEncoder(TorchModel, Encoder):
    config: OneHotEncoderConfig

    def __init__(self, config: OneHotEncoderConfig):
        super().__init__(config)
        self.nb_classes = config.nb_classes

    def _forward(self, input_dict: dict, **kwargs) -> dict:
        observations = input_dict[Columns.OBS]
        # we want to map -1 (invalid obs) to (0, 0, ..., 0)
        # so we add a class that we remove afterwards
        return {
            ENCODER_OUT: F.one_hot(
                observations.long() + 1,  # -1 -> mapped to 0
                num_classes=self.nb_classes + 1,  # new category: -1
            )[:, 1:]  # remove new category
        }
