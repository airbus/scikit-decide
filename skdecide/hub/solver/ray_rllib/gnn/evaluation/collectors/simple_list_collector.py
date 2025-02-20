from typing import Any

import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.evaluation.collectors.simple_list_collector import _PolicyCollector
from ray.rllib.policy.sample_batch import concat_samples
from ray.rllib.utils.typing import SampleBatchType

from skdecide.hub.solver.ray_rllib.action_masking.utils.spaces.space_utils import (
    ACTION_MASK,
    TRUE_OBS,
    is_masked_obs,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    EDGES,
    NODES,
    is_graph_dict,
    is_graph_dict_multiinput,
    pad_axis,
    pad_batched_graph_dict,
)


def graph_policy_collector_build(self: _PolicyCollector) -> SampleBatch:
    # Create batch from our buffers.
    prepare_for_concat_samples_graph(self.batches)  # pad graph samples if necessary
    batch = concat_samples(self.batches)
    # Clear batches for future samples.
    self.batches = []
    # Reset agent steps to 0.
    self.agent_steps = 0
    # Add num_grad_updates counter to the policy's batch.
    batch.num_grad_updates = self.policy.num_grad_updates

    return batch


def graph2node_policy_collector_build(self: _PolicyCollector) -> SampleBatch:
    # Create batch from our buffers.
    prepare_for_concat_samples_graph2node(
        self.batches
    )  # pad graph samples if necessary
    batch = concat_samples(self.batches)
    # Clear batches for future samples.
    self.batches = []
    # Reset agent steps to 0.
    self.agent_steps = 0
    # Add num_grad_updates counter to the policy's batch.
    batch.num_grad_updates = self.policy.num_grad_updates

    return batch


def monkey_patch_policy_collector(graph2node: bool = False) -> None:
    """Monkey patch rllib so that concat_samples pad graph arrays if necessary."""
    if graph2node:
        _PolicyCollector.build = graph2node_policy_collector_build
    else:
        _PolicyCollector.build = graph_policy_collector_build


def prepare_for_concat_samples_graph2node(samples: list[SampleBatchType]) -> None:
    if all(isinstance(s, SampleBatch) for s in samples) and all(
        s.get_interceptor is None for s in samples
    ):
        # prepare graphs obs
        prepare_for_concat_samples_graph(samples)
        # pad also action logits
        key = SampleBatch.ACTION_DIST_INPUTS
        max_nodes = max(s[key].shape[1] for s in samples)
        minus_infty_approx = np.finfo(samples[0][key].dtype).min
        for s in samples:
            s[key] = pad_axis(
                s[key], max_dim=max_nodes, axis=1, value=minus_infty_approx
            )


def prepare_for_concat_samples_graph(samples: list[SampleBatchType]) -> None:
    if all(isinstance(s, SampleBatch) for s in samples) and all(
        s.get_interceptor is None for s in samples
    ):
        for key in (SampleBatch.OBS, SampleBatch.NEXT_OBS):
            # pad the obs (check inside the function if necessary or not)
            pad_sample_batches_obs(samples, keys=(key,))


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
        # pad action mask
        pad_sample_batches_action_mask(samples=samples, keys=keys + (ACTION_MASK,))
    elif is_graph_dict_multiinput(first_subobs):
        # pad each subobs (that are graphs)
        for subkey in first_subobs:
            pad_sample_batches_obs(samples=samples, keys=keys + (subkey,))
    else:
        # not a graph => nothing to pad
        ...


def pad_sample_batches_action_mask(
    samples: list[SampleBatch], keys: tuple[str, ...]
) -> None:
    if len(set(get_item(s, keys).shape[1] for s in samples)) > 1:
        # different number of nodes => padding
        max_n_nodes = max(get_item(s, keys).shape[1] for s in samples)
        for s in samples:
            set_item(
                s,
                keys=keys,
                value=pad_axis(get_item(s, keys), max_n_nodes, axis=1),
            )
