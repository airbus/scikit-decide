from numbers import Number
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import tree
from ray.rllib import SampleBatch
from ray.rllib.policy.sample_batch import attempt_count_timesteps, tf, torch
from ray.rllib.utils.typing import ViewRequirementsDict


def _pop_graph_items(
    full_dict: dict[Any, Any]
) -> dict[Any, Union[gym.spaces.GraphInstance, list[gym.spaces.GraphInstance]]]:
    graph_dict = {}
    for k, v in full_dict.items():
        if isinstance(v, gym.spaces.GraphInstance) or (
            isinstance(v, list) and isinstance(v[0], gym.spaces.GraphInstance)
        ):
            graph_dict[k] = v
    for k in graph_dict:
        full_dict.pop(k)
    return graph_dict


def _split_graph_requirements(
    full_dict: ViewRequirementsDict,
) -> tuple[ViewRequirementsDict, ViewRequirementsDict]:
    graph_dict = {}
    for k, v in full_dict.items():
        if isinstance(v.space, gym.spaces.Graph):
            graph_dict[k] = v
    wo_graph_dict = {k: v for k, v in full_dict.items() if k not in graph_dict}
    return graph_dict, wo_graph_dict


class GraphSampleBatch(SampleBatch):
    def __init__(self, *args, **kwargs):
        """Constructs a sample batch with possibly graph obs.

        See `ray.rllib.SampleBatch` for more information.

        """
        # split graph samples from others.
        dict_graphs = _pop_graph_items(kwargs)
        dict_from_args = dict(*args)
        dict_graphs.update(_pop_graph_items(dict_from_args))

        super().__init__(dict_from_args, **kwargs)
        super().update(dict_graphs)

    def copy(self, shallow: bool = False) -> "SampleBatch":
        """Creates a deep or shallow copy of this SampleBatch and returns it.

        Args:
            shallow: Whether the copying should be done shallowly.

        Returns:
            A deep or shallow copy of this SampleBatch object.
        """
        copy_ = dict(self)
        data = tree.map_structure(
            lambda v: (
                np.array(v, copy=not shallow) if isinstance(v, np.ndarray) else v
            ),
            copy_,
        )
        copy_ = GraphSampleBatch(
            data,
            _time_major=self.time_major,
            _zero_padded=self.zero_padded,
            _max_seq_len=self.max_seq_len,
            _num_grad_updates=self.num_grad_updates,
        )
        copy_.set_get_interceptor(self.get_interceptor)
        copy_.added_keys = self.added_keys
        copy_.deleted_keys = self.deleted_keys
        copy_.accessed_keys = self.accessed_keys
        return copy_

    def get_single_step_input_dict(
        self,
        view_requirements: ViewRequirementsDict,
        index: Union[str, int] = "last",
    ) -> "SampleBatch":
        (
            view_requirements_graphs,
            view_requirements_wo_graphs,
        ) = _split_graph_requirements(view_requirements)
        # w/o graphs
        sample = GraphSampleBatch(
            super().get_single_step_input_dict(view_requirements_wo_graphs, index)
        )
        # handle graphs
        last_mappings = {
            SampleBatch.OBS: SampleBatch.NEXT_OBS,
            SampleBatch.PREV_ACTIONS: SampleBatch.ACTIONS,
            SampleBatch.PREV_REWARDS: SampleBatch.REWARDS,
        }
        for view_col, view_req in view_requirements_graphs.items():
            if view_req.used_for_compute_actions is False:
                continue

            # Create batches of size 1 (single-agent input-dict).
            data_col = view_req.data_col or view_col
            if index == "last":
                data_col = last_mappings.get(data_col, data_col)
                if view_req.shift_from is not None:
                    raise NotImplementedError()
                else:
                    sample[view_col] = self[data_col][-1:]
            else:
                sample[view_col] = self[data_col][
                    index : index + 1 if index != -1 else None
                ]
        return sample
