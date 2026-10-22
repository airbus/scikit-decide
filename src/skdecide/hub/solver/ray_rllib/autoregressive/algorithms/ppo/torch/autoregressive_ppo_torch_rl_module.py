#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import copy
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch as th
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.core.models.configs import RecurrentEncoderConfig
from torch import TensorType
from torch.nn import ModuleList
from torch.nn.functional import pad

from skdecide.hub.solver.ray_rllib.action_masking.distribution.mask import (
    mask_categorical_logits,
)
from skdecide.hub.solver.ray_rllib.autoregressive.algorithms.ppo.ppo_catalog import (
    AutoregressivePPOCatalog,
)
from skdecide.hub.solver.ray_rllib.autoregressive.distribution.autoregressive import (
    AutoregressiveDistribution,
)
from skdecide.hub.solver.ray_rllib.autoregressive.rl_module.torch.autoregressive_torch_rl_module import (
    AutoregressiveTorchRLModule,
)
from skdecide.hub.solver.ray_rllib.autoregressive.utils.spaces.space_utils import (
    APPLICABLE_ACTIONS,
)
from skdecide.hub.solver.ray_rllib.common.constants import ForwardPhase
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import NODES
from skdecide.hub.solver.utils.autoregressive.torch_utils import (
    add_position_feature_to_node_features,
    batched_extract_action_component_mask,
    extract_action_component_mask,
)


class AutoregressivePPOTorchRLModule(
    AutoregressiveTorchRLModule, DefaultPPOTorchRLModule
):
    catalog: AutoregressivePPOCatalog
    action_space: gym.spaces.MultiDiscrete

    def setup(self):
        # reset observation space to be the actual one for this module
        self.observation_space = self.observation_space_with_applicable_actions

        is_stateful = isinstance(
            self.catalog.actor_critic_encoder_config.base_encoder_config,
            RecurrentEncoderConfig,
        )
        if is_stateful:
            self.inference_only = False
        # If this is an `inference_only` Module, we'll have to pass this information
        # to the encoder config as well.
        if self.inference_only and self.framework == "torch":
            self.catalog.actor_critic_encoder_config.inference_only = True

        # Build models from catalog.
        self.encoder = self.catalog.build_actor_critic_encoder(framework=self.framework)
        self.vf = self.catalog.build_vf_head(framework=self.framework)
        self.action_encoder_per_component = ModuleList(
            self.catalog.build_action_encoder_per_component(framework=self.framework)
        )
        self.pi_per_component = ModuleList(
            self.catalog.build_pi_head_per_component(framework=self.framework)
        )

    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward(batch, phase=ForwardPhase.EXPLORATION, **kwargs)

    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward(batch, phase=ForwardPhase.INFERENCE, **kwargs)

    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward(batch, phase=ForwardPhase.TRAIN, **kwargs)

    def _forward(
        self,
        batch: dict[str, TensorType],
        phase: ForwardPhase = ForwardPhase.INFERENCE,
        **kwargs,
    ) -> dict[str, TensorType]:
        # Check, if the action_mask has been extracted from observations or not.
        # (e.g. in forward_train(), it can already have been extracted by compute_values())
        if APPLICABLE_ACTIONS not in batch:
            # Preprocess the batch to extract the `observations` to `Columns.OBS`.
            applicable_actions, batch = self.preprocess_batch(batch)
            batch[APPLICABLE_ACTIONS] = applicable_actions
        # Pass to appropriate method
        if phase == ForwardPhase.TRAIN:
            return self._forward_with_already_sampled_actions(
                batch=batch, phase=phase, **kwargs
            )
        else:
            return self._forward_with_action_sampling(
                batch=batch, phase=phase, **kwargs
            )

    def _forward_with_action_sampling(
        self, batch: Dict[str, Any], phase: ForwardPhase, **kwargs
    ) -> Dict[str, Any]:
        deterministic = phase == ForwardPhase.INFERENCE
        output = {}

        # extract applicable actions
        applicable_actions = batch[APPLICABLE_ACTIONS]
        # check that batch shape is (1,) and remove this dimension
        assert len(applicable_actions.shape) == 3 and len(applicable_actions) == 1, (
            "applicable_actions are supposed to be batched with only one sample"
        )
        applicable_actions = applicable_actions.reshape(
            (-1, len(self.action_space.nvec))
        )  # remove batch dimension

        if self.catalog.nb_graph_independent_components > 0:
            # Encode observations into features if any "regular" action component
            encoder_outs = self.encoder(batch)
            features = encoder_outs[ENCODER_OUT][ACTOR]
            device = features.device
            # Stateful encoder?
            if Columns.STATE_OUT in encoder_outs:
                output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]

        if self.catalog.nb_graph2node_components > 0:
            # Prepare enriched graph if graph observation + any graph node action components
            enriched_obs = copy.copy(batch[Columns.OBS])
            device = enriched_obs[NODES].device

        # pi heads for each component
        action_dist_inputs_per_component = []
        action_components = th.as_tensor([], device=device, dtype=int)
        action_component_masks = []
        action_dist_cls_per_component = self.get_action_dist_cls_per_component(
            phase=phase
        )
        action_dist_per_components = []
        for i_action_component, pi in enumerate(self.pi_per_component):
            # mask for current action component considered
            action_component_mask = extract_action_component_mask(
                action_components=action_components,
                applicable_actions=applicable_actions,
                i_action_component=i_action_component,
                action_component_dim=self.action_space.nvec[i_action_component],
            )

            # more action_component allowed?
            if action_component_mask.sum() == 0:
                # no more components available
                break

            action_component_mask = action_component_mask[None]  # add batch dimension
            action_component_masks.append(action_component_mask)

            # compute component logits
            if i_action_component < self.catalog.nb_graph_independent_components:
                # regular (graph independent) component
                logits_component = pi(features)
            else:
                # graph node component
                if i_action_component == self.catalog.nb_graph_independent_components:
                    # first graph node component: init node features encoding position of each node among graph node components
                    (
                        nb_node_features_before_position_feature,
                        enriched_obs[NODES],
                    ) = add_position_feature_to_node_features(
                        node_features=enriched_obs[NODES],
                        n_graph2node_components=self.catalog.nb_graph2node_components,
                    )
                logits_component = pi(
                    {Columns.OBS: enriched_obs}, i_component=i_action_component
                )[NODES]

            logits_component = mask_categorical_logits(
                logits=logits_component, mask=action_component_mask
            )
            action_dist_inputs_per_component.append(logits_component)

            # sample action component
            action_dist_component = action_dist_cls_per_component[
                i_action_component
            ].from_logits(logits_component)
            action_dist_per_components.append(action_dist_component)
            if deterministic:
                action_component = action_dist_component.to_deterministic().sample()
            else:
                action_component = action_dist_component.sample()
            action_components = th.cat((action_components, action_component), -1)

            # add encoded action component to features for next component logits
            if i_action_component < self.nb_action_components - 1:
                if i_action_component < self.catalog.nb_graph_independent_components:
                    # "regular" (graph independent) component => one-hot encode it
                    encoded_action_component = self.action_encoder_per_component[
                        i_action_component
                    ]({Columns.OBS: action_component})[ENCODER_OUT]

                    if (
                        i_action_component
                        < self.catalog.nb_graph_independent_components - 1
                    ):
                        # add to features for next "regular" (graph independent) components
                        features = th.cat((features, encoded_action_component), dim=-1)

                    if self.catalog.nb_graph2node_components > 0:
                        # add to node features to be used for graph node components
                        enriched_obs[NODES] = th.cat(
                            (
                                enriched_obs[NODES],
                                encoded_action_component.expand(
                                    # expand to have same shape as node features:
                                    # new features will be added to each node
                                    enriched_obs[NODES].shape[:-1] + (-1,)
                                ),
                            ),
                            -1,
                        )

                else:
                    # graph node component

                    # add position among graph node components
                    # i.e. a 1 for corresponding node (row) at corresponding position (column)
                    position = (
                        i_action_component
                        - self.catalog.nb_graph_independent_components
                    )
                    # heterogeneous?
                    if self.catalog.action_components_node_flag_indices is None:
                        # no => action component corresponds to the actual node index
                        chosen_node_id = action_component
                    else:
                        # yes => action component is the index *among filtered nodes*
                        node_flag_index = (
                            self.catalog.action_components_node_flag_indices[
                                i_action_component
                            ]
                        )
                        nodes_to_keep = batch[Columns.OBS][NODES][
                            :, :, node_flag_index
                        ].nonzero(as_tuple=True)[-1]
                        chosen_node_id = nodes_to_keep[action_component]
                    enriched_obs[NODES][
                        :,
                        chosen_node_id,
                        nb_node_features_before_position_feature + position,
                    ] = 1

        actions = action_components[None]  # add batch dimension
        action_dist_cls: AutoregressiveDistribution = self.get_action_dist_cls(
            phase=phase
        )
        action_dist = action_dist_cls(
            action_dist_per_components, masks=action_component_masks
        )

        output[Columns.ACTION_LOGP] = action_dist.logp(actions)
        output[Columns.ACTIONS] = pad(
            # pad actions with -1 in case of early break of the loop (no more components)
            actions,
            pad=(0, self.nb_action_components - actions.shape[-1]),
            value=-1,  # -1 = no component
        )
        # pad action dist inputs with dummy logits + masks to have max number of components (helps storage)
        mask_dtype = action_component_masks[-1].dtype
        logits_dtype = action_dist_inputs_per_component[-1].dtype
        for i_action_component in range(
            len(action_component_masks), self.nb_action_components
        ):
            action_component_dim = self.action_space.nvec[i_action_component]
            action_dist_inputs_per_component.append(
                th.zeros(
                    (1, action_component_dim),
                    device=device,
                    dtype=logits_dtype,
                )
            )
            action_component_masks.append(
                th.zeros(
                    (1, action_component_dim),
                    device=device,
                    dtype=mask_dtype,
                )
            )
        action_dist_inputs = (
            tuple(action_dist_inputs_per_component),
            tuple(action_component_masks),
        )
        output[Columns.ACTION_DIST_INPUTS] = action_dist_inputs

        return output

    def _forward_with_already_sampled_actions(
        self, batch: Dict[str, Any], phase: ForwardPhase, **kwargs
    ) -> Dict[str, Any]:
        assert phase == ForwardPhase.TRAIN

        output = {}

        # extract applicable actions
        applicable_actions = batch[APPLICABLE_ACTIONS]

        # extract sampled actions
        actions = batch[Columns.ACTIONS]

        # Encode observations into features
        encoder_outs = self.encoder(batch)
        features = encoder_outs[ENCODER_OUT][ACTOR]
        # hyp: features are flat
        assert len(features.shape) == 2
        # save encoding for compute_values()
        output[Columns.EMBEDDINGS] = encoder_outs[ENCODER_OUT][CRITIC]

        # batch_dim
        batch_dim = len(features)
        assert len(applicable_actions == batch_dim)

        # full features: extracted features + encoded action components (all at once for efficiency)
        encoded_graph_independent_action_components = tuple(
            action_encoder_component({Columns.OBS: actions[:, i_action_component]})[
                ENCODER_OUT
            ]
            for i_action_component, action_encoder_component in enumerate(
                self.action_encoder_per_component
            )
        )
        full_features = th.cat(
            (features,) + encoded_graph_independent_action_components,
            dim=-1,
        )
        n_features_by_component = features.shape[-1] + np.cumsum(
            [0]
            + list(
                config.output_dims[0]
                for config in self.catalog.action_encoder_per_component_configs
            )
        )
        # node features (original graph obs features + graph node components)
        if self.catalog.nb_graph2node_components > 0:
            enriched_obs = copy.copy(batch[Columns.OBS])
            # + onehot encoded (graph independent) action components (concatenate once for efficiency)
            enriched_obs[NODES] = th.cat(
                (enriched_obs[NODES],)
                + tuple(
                    encoded_component[
                        :,
                        None,  # add a dimension corresponding to node id as we add features to all nodes
                    ].expand(
                        # expand to have same shape as node features:
                        # new features will be added to each node
                        enriched_obs[NODES].shape[:-1] + (-1,)
                    )
                    for encoded_component in encoded_graph_independent_action_components
                ),
                dim=-1,
            )
            # + onehot encoded position of graph nodes in graph node action components (only zeros for now)
            (
                n_node_features_before_position_feature,
                enriched_obs[NODES],
            ) = add_position_feature_to_node_features(
                node_features=enriched_obs[NODES],
                n_graph2node_components=self.catalog.nb_graph2node_components,
            )
        action_dist_inputs_per_component = []
        action_component_masks = []
        for i_action_component, pi in enumerate(self.pi_per_component):
            # all (batchwise) components for this idx are <0 => no more true components
            if (actions[:, i_action_component] < 0).all():
                # no more computation needed
                break

            # compute logits only for samples with >=0 component
            mask_samples_with_component = actions[:, i_action_component] >= 0
            if mask_samples_with_component.all():
                # all (batchwise) components needed
                compute_all_samples = True
                indices_samples_with_component = None
                mask_samples_with_component = None
            else:
                # some components do need to be computed
                compute_all_samples = False
                indices_samples_with_component = (
                    mask_samples_with_component.nonzero().flatten()
                )

            # mask for current action component considered
            action_component_mask = batched_extract_action_component_mask(
                action_components=actions,
                applicable_actions=applicable_actions,
                i_action_component=i_action_component,
                action_component_dim=self.action_space.nvec[i_action_component],
                nonzeromask_samples_mask=mask_samples_with_component,
            )
            action_component_masks.append(action_component_mask)

            # compute logits only for relevant samples
            if compute_all_samples:
                # all samples are relevant
                if i_action_component < self.catalog.nb_graph_independent_components:
                    # regular (graph independent) component
                    action_component_full_logits = pi(
                        full_features[:, : n_features_by_component[i_action_component]]
                    )
                else:
                    # graph node component
                    action_component_full_logits = pi(
                        {Columns.OBS: enriched_obs}, i_component=i_action_component
                    )[NODES]
            else:
                # only some samples are relevant
                if i_action_component < self.catalog.nb_graph_independent_components:
                    # regular (graph independent) component
                    action_component_logits = pi(
                        full_features[
                            indices_samples_with_component,
                            : n_features_by_component[i_action_component],
                        ]
                    )
                else:
                    # graph node component
                    action_component_logits = pi(
                        {
                            Columns.OBS: {
                                # extract subgraph dict corresponding to relevant samples
                                k: v[indices_samples_with_component]
                                for k, v in enriched_obs.items()
                            }
                        },
                        i_component=i_action_component,
                    )[NODES]
                # fill logits with 0 for irrelevant samples  (will be masked afterwards)
                action_component_full_logits = th.zeros_like(
                    action_component_mask, dtype=action_component_logits.dtype
                )
                action_component_full_logits[indices_samples_with_component, :] = (
                    action_component_logits
                )
            # mask logits
            action_component_full_logits = mask_categorical_logits(
                logits=action_component_full_logits, mask=action_component_mask
            )
            action_dist_inputs_per_component.append(action_component_full_logits)

            # add position of current graph node component for next prediction
            if (
                i_action_component >= self.catalog.nb_graph_independent_components
                and i_action_component < self.nb_action_components - 1
            ):
                position = (
                    # position of component among graph node components:
                    i_action_component - self.catalog.nb_graph_independent_components
                )
                action_component = actions[:, i_action_component]
                # skip samples with -1 components
                mask = action_component >= 0
                batch_ids = th.as_tensor(range(len(action_component)))[mask]
                chosen_node_ids = _extract_node_ids_from_action_component(
                    x=enriched_obs[NODES][mask],
                    action_component=action_component[mask],
                    node_flag_index=self.catalog.action_components_node_flag_indices[
                        i_action_component
                    ]
                    if self.catalog.action_components_node_flag_indices is not None
                    else None,
                )
                enriched_obs[NODES][
                    batch_ids,
                    chosen_node_ids,
                    n_node_features_before_position_feature + position,
                ] = 1

        # action_dist_inputs
        action_dist_inputs = (
            tuple(action_dist_inputs_per_component),
            tuple(action_component_masks),
        )
        output[Columns.ACTION_DIST_INPUTS] = action_dist_inputs

        # no need to add actions and logp to output not used by forward_train output

        return output

    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        # Check, if the action_mask has been extracted from observations or not.
        if APPLICABLE_ACTIONS not in batch:
            # Preprocess the batch to extract the `observations` to `Columns.OBS`.
            applicable_actions, batch = self.preprocess_batch(batch)
            batch[APPLICABLE_ACTIONS] = applicable_actions
        # Call the super's method to compute values for GAE.
        return super().compute_values(batch, embeddings)


def _extract_node_ids_from_action_component_heterogeneous_graph(
    x: th.Tensor, node_flag_index: int, action_component: th.Tensor
) -> th.Tensor:
    """

    # Parameters
    x: node features, shape (n_samples, n_nodes, n_features)
    action_component: node indices among filtered nodes, sample by sample, shape (n_samples,)
    node_flag_index: index of the binary node flag used to filter nodes

    # Returns
    node indices among all nodes, shape (n_samples,)

    """
    # 1-based rank of filtered nodes
    filtered_nodes_rank = x[:, :, node_flag_index].bool().cumsum(dim=1)
    # convert action_component (0-based rank) into 1-based target rank, with shape broadcastable to filtered_nodes_rank
    target_rank = action_component.unsqueeze(1) + 1
    # global node index corresponds to first match between ranks and target for each sample
    chosen_node_ids = (filtered_nodes_rank == target_rank).long().argmax(dim=1)
    return chosen_node_ids


def _extract_node_ids_from_action_component(
    x: th.Tensor,
    action_component: th.Tensor,
    node_flag_index: int | None,
) -> th.Tensor:
    """

    # Parameters
    x: node features, shape (n_samples, n_nodes, n_features)
    action_component: node indices among filtered nodes, sample by sample, shape (n_samples,)
    node_flag_index: index of the binary node flag used to filter nodes. If None, no filter is applies.

    # Returns
    node indices among all nodes, shape (n_samples,)

    """
    if node_flag_index is None:
        return action_component.long()
    else:
        return _extract_node_ids_from_action_component_heterogeneous_graph(
            x=x,
            action_component=action_component,
            node_flag_index=node_flag_index,
        )
