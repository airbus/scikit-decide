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
        n_nodes_seq = tuple((len(logits) for logits in logits_seq))
        if len(set(n_nodes_seq)) > 1:
            assert not logits_seq.finalized
            minus_infty_approx = np.finfo(logits_seq[0].dtype).min
            max_n_nodes = max(n_nodes_seq)
            logits_seq.data = [
                pad_axis(logits, max_dim=max_n_nodes, value=minus_infty_approx)
                for logits in logits_seq
            ]
    return original_to_numpy()
