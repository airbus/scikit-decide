#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog

from skdecide.hub.solver.ray_rllib.autoregressive.models.catalog import (
    AutoregressiveCatalog,
)
from skdecide.hub.solver.ray_rllib.gnn.algorithms.ppo.ppo_catalog import (
    GraphPPOCatalog,
)


class AutoregressivePPOCatalog(AutoregressiveCatalog, PPOCatalog): ...


class AutoregressiveGraphPPOCatalog(AutoregressivePPOCatalog, GraphPPOCatalog): ...
