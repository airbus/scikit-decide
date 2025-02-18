from typing import Any

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import concat_samples as original_concat_samples
from ray.rllib.utils.typing import SampleBatchType

from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    EDGES,
    NODES,
    TRUE_OBS,
    is_graph_dict,
    is_graph_dict_multiinput,
    is_masked_obs,
    pad_batched_graph_dict,
)


def concat_samples(samples: list[SampleBatchType]) -> SampleBatchType:
    if all(isinstance(s, SampleBatch) for s in samples) and all(
        s.get_interceptor is None for s in samples
    ):
        s: SampleBatch = samples[0]
        for key in (SampleBatch.OBS, SampleBatch.NEXT_OBS):
            # pad the obs (check inside the function if necessary or not)
            pad_sample_batches_obs(samples, keys=(key,))

    return original_concat_samples(samples)


def get_item(s: dict[str, Any], keys: tuple[str, ...]) -> Any:
    if len(keys) == 0:
        return s
    else:
        return get_item(s[keys[0]], keys[1:])


def set_item(s: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    if len(keys) == 0:
        raise ValueError("keys must of len >=1")
    elif len(keys) == 1:
        s[keys[0]] = value
    else:
        set_item(s[keys[0]], keys=keys[1:], value=value)


def pad_sample_batches_obs(samples: list[SampleBatch], keys: tuple[str, ...]) -> None:
    first_subobs = get_item(samples[0], keys)

    if is_graph_dict(first_subobs):
        if (
            len(set(get_item(s, keys)[NODES].shape[1] for s in samples)) > 1
            or len(set(get_item(s, keys)[EDGES].shape[1] for s in samples)) > 1
        ):
            # different number of nodes or edges => padding
            max_n_nodes = max(get_item(s, keys)[NODES].shape[1] for s in samples)
            max_n_edges = max(get_item(s, keys)[EDGES].shape[1] for s in samples)
            for s in samples:
                set_item(
                    s,
                    keys=keys,
                    value=pad_batched_graph_dict(
                        get_item(s, keys),
                        max_n_nodes=max_n_nodes,
                        max_n_edges=max_n_edges,
                    ),
                )
    elif is_masked_obs(first_subobs):
        # pad "true_obs" part
        pad_sample_batches_obs(samples=samples, keys=keys + (TRUE_OBS,))
    elif is_graph_dict_multiinput(first_subobs):
        # pad each subobs (that are graphs)
        for subkey in first_subobs:
            pad_sample_batches_obs(samples=samples, keys=keys + (subkey,))
    else:
        # not a graph => nothing to pad
        ...
