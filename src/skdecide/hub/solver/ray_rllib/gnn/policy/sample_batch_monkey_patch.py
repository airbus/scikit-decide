from __future__ import annotations

from ray.rllib.policy.sample_batch import concat_samples

from skdecide.hub.solver.ray_rllib.gnn.policy.sample_batch import (
    concat_samples_graph,
    concat_samples_graph2node,
)
from skdecide.hub.solver.ray_rllib.gnn.policy.sample_batch_original_code import (
    original_concat_samples,
)


def monkey_patch_concat_samples(graph2node: bool = False) -> None:
    """Monkey patch rllib so that concat_samples pad graph arrays if necessary.

    Note we need to update functions
    - `__code__`:  bytecode
    - `__globals__`: namespace, immutable attribute, which is actually
       the namespace of the functions modules

    That's why
    - we put `original_concat_samples` in a dedicated module so that its namespace can be updated by
      concat_samples.__globals__ without side effects
    - we only add to concat_samples.__globals__ the necessary names
      ("original_concat_samples" and "prepare_for_concat_samples_graph2node" or "prepare_for_concat_samples_graph")
      as it is skdecide.hub.solver.ray_rllib.gnn.policy.sample_batch namespace.

    """
    if not hasattr(original_concat_samples, "_original_globals"):
        # store original function code only if not already done
        original_concat_samples.__code__ = concat_samples.__code__
        original_concat_samples._original_globals = dict(concat_samples.__globals__)
        original_concat_samples.__globals__.update(concat_samples.__globals__)

    if graph2node:
        new_concat_samples = concat_samples_graph2node
    else:
        new_concat_samples = concat_samples_graph
    concat_samples.__code__ = new_concat_samples.__code__
    for name in concat_samples.__code__.co_names:
        concat_samples.__globals__[name] = new_concat_samples.__globals__[name]


def unmonkey_patch_concat_samples() -> None:
    if hasattr(original_concat_samples, "_original_globals"):
        concat_samples.__code__ = original_concat_samples.__code__
        for k in list(concat_samples.__globals__):
            concat_samples.__globals__.pop(k)
        concat_samples.__globals__.update(original_concat_samples._original_globals)
        del original_concat_samples._original_globals
