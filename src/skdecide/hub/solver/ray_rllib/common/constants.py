#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum

TORCH_FRAMEWORK = "torch"


class ForwardPhase(Enum):
    TRAIN = "train"
    INFERENCE = "inference"
    EXPLORATION = "exploration"
