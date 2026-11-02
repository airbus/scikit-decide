#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from ray.rllib.utils.spaces.space_utils import batch

from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.batch_original_code import (
    original_batch,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import batch_graph


def monkey_patch_batch(graph2node: bool = False) -> None:
    """Monkey patch rllib so that batch() pad graph arrays if necessary.

    Note we need to update function's
    - `__code__`:  bytecode
    - `__globals__`: namespace, immutable attribute, which is actually
       the namespace of the function's modules

    That's why
    - we put `original_batch` in a dedicated module so that its namespace can be updated by
      batch.__globals__ without side effects
    - we only add to batch.__globals__ the necessary names
      ("original_batch" and "prepare_for_batch_graph2node" or "prepare_for_batch_graph")

    """
    if not hasattr(original_batch, "_original_globals"):
        # store original function code only if not already done
        original_batch.__code__ = batch.__code__
        original_batch._original_globals = dict(batch.__globals__)
        original_batch.__globals__.update(batch.__globals__)

    new_batch = batch_graph
    batch.__code__ = new_batch.__code__
    for name in batch.__code__.co_names:
        batch.__globals__[name] = new_batch.__globals__[name]


def unmonkey_patch_batch() -> None:
    if hasattr(original_batch, "_original_globals"):
        batch.__code__ = original_batch.__code__
        for k in list(batch.__globals__):
            batch.__globals__.pop(k)
        batch.__globals__.update(original_batch._original_globals)
        del original_batch._original_globals
