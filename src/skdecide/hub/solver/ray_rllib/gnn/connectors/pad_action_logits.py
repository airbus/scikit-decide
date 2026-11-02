#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional

import numpy as np
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType

from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import pad_axis


class PadGraph2NodeActionLogits(ConnectorV2):
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        metrics: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> Any:
        # We pad action logits directly in single agent episodes
        # It is done before episodes are put into batch by default connectors
        has_action_logits = all(
            Columns.ACTION_DIST_INPUTS in sa_episode.extra_model_outputs
            for sa_episode in self.single_agent_episode_iterator(episodes)
        )
        if has_action_logits:
            n_nodes_per_episode = set(
                sa_episode.extra_model_outputs[Columns.ACTION_DIST_INPUTS].data.shape[1]
                for sa_episode in self.single_agent_episode_iterator(episodes)
            )
            if len(n_nodes_per_episode) > 1:
                max_n_nodes = max(n_nodes_per_episode)
                for sa_episode in self.single_agent_episode_iterator(episodes):
                    logits = sa_episode.extra_model_outputs[
                        Columns.ACTION_DIST_INPUTS
                    ].data
                    if logits.shape[1] < max_n_nodes:
                        minus_infty_approx = np.finfo(logits.dtype).min
                        sa_episode.extra_model_outputs[
                            Columns.ACTION_DIST_INPUTS
                        ].data = pad_axis(
                            logits,
                            max_dim=max_n_nodes,
                            value=minus_infty_approx,
                            axis=1,
                        )
        return batch
