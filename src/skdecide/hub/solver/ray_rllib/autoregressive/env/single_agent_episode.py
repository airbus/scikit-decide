#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ray.rllib.core import Columns
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.utils.infinite_lookback_buffer import InfiniteLookbackBuffer

from skdecide.hub.solver.ray_rllib.common.utils.spaces.utils import pad_axis


def autoregressive_graph2node_single_agent_episode_to_numpy(
    self: SingleAgentEpisode,
    original_to_numpy: Callable[[], SingleAgentEpisode] | None = None,
) -> SingleAgentEpisode:
    if original_to_numpy is None:
        original_to_numpy = self._to_numpy_unpatched

    # pad action logits if necessary
    key = Columns.ACTION_DIST_INPUTS
    if key in self.extra_model_outputs:
        autoregressive_dist_inputs_seq: InfiniteLookbackBuffer = (
            self.extra_model_outputs[key]
        )
        if not autoregressive_dist_inputs_seq.finalized:
            first_autoregressive_dist_inputs = autoregressive_dist_inputs_seq[0]
            if (
                isinstance(first_autoregressive_dist_inputs, tuple)
                and len(first_autoregressive_dist_inputs) == 2
                and isinstance(first_autoregressive_dist_inputs[0], tuple)
                and isinstance(first_autoregressive_dist_inputs[1], tuple)
            ):
                # nb action components
                max_nb_components = len(first_autoregressive_dist_inputs[0])
                # size of discrete action component space per sample (could be variable in case of graph node action components)
                dim_action_seq_per_component = [[] for _ in range(max_nb_components)]
                for (
                    action_dist_inputs_per_component,
                    action_component_masks,
                ) in autoregressive_dist_inputs_seq:
                    for i_component, (
                        action_dist_component_inputs,
                        action_component_mask,
                    ) in enumerate(
                        zip(action_dist_inputs_per_component, action_component_masks)
                    ):
                        dim_action_seq_per_component[i_component].append(
                            max(
                                len(action_dist_component_inputs),
                                len(action_component_mask),
                            )
                        )
                if any(
                    len(set(dim_action_component_seq)) > 1
                    for dim_action_component_seq in dim_action_seq_per_component
                ):
                    max_dim_action_per_component = [
                        max(dim_action_component_seq)
                        for dim_action_component_seq in dim_action_seq_per_component
                    ]
                    minus_infty_approx = np.finfo(
                        first_autoregressive_dist_inputs[0].dtype
                    ).min
                    autoregressive_dist_inputs_seq.data = [
                        (
                            tuple(
                                pad_axis(
                                    logits_component,
                                    max_dim=max_dim_action_per_component[i_component],
                                    value=minus_infty_approx,
                                )
                                if len(logits_component)
                                < max_dim_action_per_component[i_component]
                                else logits_component
                                for i_component, logits_component in enumerate(
                                    action_dist_inputs_per_component
                                )
                            ),
                            tuple(
                                pad_axis(
                                    action_component_mask,
                                    max_dim=max_dim_action_per_component[i_component],
                                    value=0,
                                )
                                if len(action_component_mask)
                                < max_dim_action_per_component[i_component]
                                else action_component_mask
                                for i_component, action_component_mask in enumerate(
                                    action_component_masks
                                )
                            ),
                        )
                        for action_dist_inputs_per_component, action_component_masks in autoregressive_dist_inputs_seq
                    ]
    return original_to_numpy()
