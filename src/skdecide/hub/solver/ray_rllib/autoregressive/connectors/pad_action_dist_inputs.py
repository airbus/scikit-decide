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

from skdecide.hub.solver.ray_rllib.common.utils.spaces.utils import pad_axis


class PadAutoregressiveGraph2NodeActionDistributionInputs(ConnectorV2):
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
        # We pad action component logits and mask directly in single agent episodes
        # It is done before episodes are put into batch by default connectors
        has_action_dist_inputs = all(
            Columns.ACTION_DIST_INPUTS in sa_episode.extra_model_outputs
            for sa_episode in self.single_agent_episode_iterator(episodes)
        )
        if has_action_dist_inputs:
            action_component_dim_per_component_per_episode = set(
                tuple(
                    action_component_mask.shape[1]
                    for action_component_mask in sa_episode.extra_model_outputs[
                        Columns.ACTION_DIST_INPUTS
                    ].data[1]
                )
                for sa_episode in self.single_agent_episode_iterator(episodes)
            )
            if len(action_component_dim_per_component_per_episode) > 1:
                action_component_dim_per_component = tuple(
                    max(action_component_dim_seq)
                    for action_component_dim_seq in zip(
                        *action_component_dim_per_component_per_episode
                    )
                )
                for sa_episode in self.single_agent_episode_iterator(episodes):
                    action_dist_inputs_per_component, action_component_masks = (
                        sa_episode.extra_model_outputs[Columns.ACTION_DIST_INPUTS].data
                    )
                    sa_episode.extra_model_outputs[Columns.ACTION_DIST_INPUTS].data = (
                        tuple(
                            pad_axis(
                                logits_component,
                                max_dim=action_component_dim_per_component[i_component],
                                value=np.finfo(logits_component.dtype).min,
                                axis=1,
                            )
                            if len(logits_component)
                            < action_component_dim_per_component[i_component]
                            else logits_component
                            for i_component, logits_component in enumerate(
                                action_dist_inputs_per_component
                            )
                        ),
                        tuple(
                            pad_axis(
                                action_component_mask,
                                max_dim=action_component_dim_per_component[i_component],
                                value=0,
                                axis=1,
                            )
                            if len(action_component_mask)
                            < action_component_dim_per_component[i_component]
                            else action_component_mask
                            for i_component, action_component_mask in enumerate(
                                action_component_masks
                            )
                        ),
                    )
        return batch
