#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ray.rllib.core import Columns
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.utils.infinite_lookback_buffer import InfiniteLookbackBuffer

from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import pad_axis


def monkey_patch_single_agent_episode_for_graph2node() -> None:
    if not hasattr(SingleAgentEpisode, "_to_numpy_unpatched"):
        # store true method only if not already done
        SingleAgentEpisode._to_numpy_unpatched = SingleAgentEpisode.to_numpy
    SingleAgentEpisode.to_numpy = graph2node_single_agent_episode_to_numpy


def unmonkey_patch_single_agent_episode() -> None:
    if hasattr(SingleAgentEpisode, "_to_numpy_unpatched"):
        SingleAgentEpisode.to_numpy = SingleAgentEpisode._to_numpy_unpatched
        del SingleAgentEpisode._to_numpy_unpatched


def graph2node_single_agent_episode_to_numpy(
    self: SingleAgentEpisode,
    original_to_numpy: Callable[[], SingleAgentEpisode] | None = None,
) -> SingleAgentEpisode:
    if original_to_numpy is None:
        original_to_numpy = self._to_numpy_unpatched

    # pad action logits if necessary
    key = Columns.ACTION_DIST_INPUTS
    if key in self.extra_model_outputs:
        logits_seq: InfiniteLookbackBuffer = self.extra_model_outputs[key]
        if len(set(len(logits) for logits in logits_seq)) > 1:
            assert not logits_seq.finalized
            minus_infty_approx = np.finfo(logits_seq[0].dtype).min
            max_n_nodes = max(len(logits) for logits in logits_seq)
            logits_seq.data = [
                pad_axis(logits, max_dim=max_n_nodes, value=minus_infty_approx)
                for logits in logits_seq
            ]
    return original_to_numpy()
