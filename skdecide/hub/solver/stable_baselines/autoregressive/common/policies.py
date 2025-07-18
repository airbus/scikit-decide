import copy
from functools import partial
from typing import Any, Optional, Union

import numpy as np
import torch as th
import torch_geometric as thg
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from torch.nn import ModuleList
from torch.nn import functional as F

from skdecide.hub.solver.stable_baselines.autoregressive.common.distributions import (
    MultiMaskableCategoricalDistribution,
)
from skdecide.hub.solver.stable_baselines.gnn.common.policies import (
    BaseGNNActorCriticPolicy,
)
from skdecide.hub.solver.utils.autoregressive.torch_utils import (
    batched_extract_action_component_mask,
    extract_action_component_mask,
)
from skdecide.hub.solver.utils.gnn.space_utils import (
    add_onehot_encoded_discrete_feature_to_graph_space,
)
from skdecide.hub.solver.utils.gnn.torch_layers import (
    Graph2NodeLayer,
    GraphFeaturesExtractor,
)
from skdecide.hub.solver.utils.gnn.torch_utils import unbatch_node_logits

ACTION_NET_ARCH_KEY = "pi"
VALUE_NET_ARCH_KEY = "vf"

ObsType = Union[th.Tensor, thg.data.Data]


class AutoregressiveActorCriticPolicy(MaskableActorCriticPolicy):
    """Policy with autoregressive action prediction for multidiscrete restricted actions.

    Hypotheses:
    - action space multidiscrete, but with possibility of components at -1 to encode variable length actions.
      (e.g. parametric actions whose arity depends on the action type)
    - not all actions are available depending on environment state
    - the current applicable actions are given via the argument `action_masks`, from `MaskableActorCriticPolicy` API,
      as a variable-length numpy.array's (for a single sample) or a list of such numpy.array's when evaluating
      several samples at once.

    Notes:
        The code already take into account `n_graph2node_components` even though will be only useful for
        `AutoregressiveGraph2NodeActorCriticPolicy`, but allows to share most of the code between the 2 classes.

    """

    action_space: spaces.MultiDiscrete
    # By default no graph2node components (not even sure to have graph obs) and thus no graph->node gnn
    n_graph2node_components: int = 0
    action_gnn_class: Optional[type[nn.Module]] = None
    action_gnn_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[dict[str, list[int]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        # Check action space
        if not isinstance(action_space, spaces.MultiDiscrete):
            raise ValueError("action_space must be a multidiscrete space.")
        if not len(np.array(action_space.nvec).shape) == 1:
            raise ValueError(
                "action_space must be a *flat* multidiscrete space. "
                "(i.e action_space.nvec is a 1-d array)."
            )

        if not share_features_extractor:
            raise NotImplementedError()

        # default list of flags for action components graph nodes (not relevant if not in
        # AutoregressiveHeteroGraph2NodeActorCriticPolicy)
        self.action_components_node_flag_indices: list[Optional[int]] = [None] * len(
            action_space.nvec
        )

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self._reset_action_dist()

    @property
    def _n_graphindependent_components(self) -> int:
        return len(self.action_space.nvec) - self.n_graph2node_components

    def _reset_action_dist(self) -> None:
        """Reset action distribution to ensure all marginal are reset.

        Overwrite also default action distribution (which assume independent marginals)

        """
        self.action_dist = MultiMaskableCategoricalDistribution(
            distributions=[
                MaskableCategoricalDistribution(action_dim=int(action_dim))
                for action_dim in self.action_space.nvec
            ]
        )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # mlp_extractor for value only
        net_arch_value = {VALUE_NET_ARCH_KEY: self.net_arch.get(VALUE_NET_ARCH_KEY, [])}
        self.mlp_extractor_value = MlpExtractor(
            self.features_dim,
            net_arch=net_arch_value,
            activation_fn=self.activation_fn,
            device=self.device,
        )
        self.mlp_extractor = self.mlp_extractor_value  # alias for predict_values()

        # mlp_extractors for each action components not directly depending on observation graph
        net_arch_action = {
            ACTION_NET_ARCH_KEY: self.net_arch.get(ACTION_NET_ARCH_KEY, [])
        }
        previous_action_components_features_dims = np.cumsum(
            [0] + list(self.action_space.nvec[:-1])
        )
        self.mlp_extractors_action = ModuleList(
            [
                MlpExtractor(
                    self.features_dim + previous_action_components_features_dim,
                    net_arch=net_arch_action,
                    activation_fn=self.activation_fn,
                    device=self.device,
                )
                if i_component < self._n_graphindependent_components
                else None
                for i_component, previous_action_components_features_dim in enumerate(
                    previous_action_components_features_dims
                )
            ]
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        # value net
        self.value_net = nn.Linear(self.mlp_extractor_value.latent_dim_vf, 1)

        # observation space used by Graph2Node GNN's: enriched with
        #   - previous graph-independent components as global features repeated for each node
        #   - previous graph_node components as a node feature (onehot) encoding the position of the node among
        #     the graph node action components
        if self.n_graph2node_components > 0:
            enriched_observation_space = self.observation_space
            # graph-independent action components
            for new_dim in self.action_space.nvec[
                : self._n_graphindependent_components
            ]:
                enriched_observation_space = (
                    add_onehot_encoded_discrete_feature_to_graph_space(
                        enriched_observation_space, new_dim=new_dim
                    )
                )
            # graph node action components position
            enriched_observation_space = (
                add_onehot_encoded_discrete_feature_to_graph_space(
                    enriched_observation_space, new_dim=self.n_graph2node_components
                )
            )

        # action nets
        self.action_nets = ModuleList(
            [
                nn.Linear(
                    mlp_extractor_action.latent_dim_pi,
                    self.action_space.nvec[i_component],
                )
                if i_component < self._n_graphindependent_components
                else Graph2NodeLayer(
                    observation_space=enriched_observation_space,
                    gnn_class=self.action_gnn_class,
                    gnn_kwargs=self.action_gnn_kwargs,
                )
                for i_component, mlp_extractor_action in enumerate(
                    self.mlp_extractors_action,
                )
            ]
        )

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor_value: np.sqrt(2),
                **{
                    mlp_extractor: np.sqrt(2)
                    for mlp_extractor in self.mlp_extractors_action
                },
                self.value_net: 1,
                **{
                    action_net: np.sqrt(2)
                    if i_component >= self._n_graphindependent_components
                    else 0.01
                    for i_component, action_net in enumerate(self.action_nets)
                },
            }
            for module, gain in module_gains.items():
                if module is not None:
                    module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )  # type: ignore[call-arg]

    def forward(
        self,
        obs: ObsType,
        deterministic: bool = False,
        action_masks: Optional[Union[th.Tensor, np.ndarray]] = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # features
        features = self.extract_features(obs)

        # value
        latent_vf = self.mlp_extractor_value.forward_critic(features)
        values = self.value_net(latent_vf)

        # actions + distribution update
        actions = self._compute_distribution_and_sample_action(
            features=features,
            observation=obs,
            deterministic=deterministic,
            action_masks=action_masks,
        )

        # logp computation
        log_prob = self.action_dist.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: ObsType,
        actions: th.Tensor,
        action_masks: Optional[list[th.Tensor]] = None,
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        # hyp: actions are flat multidiscrete
        assert len(actions.shape) == 2

        # features
        features = self.extract_features(obs)
        # hyp: features are flat
        assert len(features.shape) == 2

        # batch_dim?
        if features is None:
            if obs.batch is None:  # obs: thg.data.Data
                batch_dim = 1
            else:  # obs: thg.data.Batch
                assert isinstance(obs, thg.data.Batch)
                batch_dim = obs.num_graphs
        else:
            batch_dim = len(features)

        # applicable actions
        # hyp: applicable actions given via action_masks, sample by sample
        applicable_actions = action_masks
        assert (
            applicable_actions is not None
            and len(applicable_actions) == batch_dim
            and all(
                len(applicable_actions_by_sample) > 0
                for applicable_actions_by_sample in applicable_actions
            )
        )

        # features + onehot encoded (graph independent) action components (concatenate once for efficiency)
        if self.n_graph2node_components > 0:
            n_components_to_encode = self._n_graphindependent_components
        else:
            n_components_to_encode = self._n_graphindependent_components - 1
        encoded_graph_independent_action_components = (
            _encode_actions_first_components_for_features(
                actions=actions,
                action_space=self.action_space,
                n_components=n_components_to_encode,
            )
        )
        full_features = th.cat(
            (features, encoded_graph_independent_action_components), -1
        )
        n_features_by_component = features.shape[-1] + np.cumsum(
            [0]
            + list(self.action_space.nvec[: self._n_graphindependent_components - 1])
        )

        if self.n_graph2node_components > 0:
            # node features
            # + onehot encoded (graph independent) action components (concatenate once for efficiency)
            # + onehot encoded position of graph nodes in graph node action components (only zeros for now)
            enriched_obs = copy.copy(obs)
            enriched_obs.x = th.cat(
                (
                    enriched_obs.x,
                    encoded_graph_independent_action_components[enriched_obs.batch, :],
                    th.zeros(
                        (len(enriched_obs.x), self.n_graph2node_components),
                        device=self.device,
                        dtype=enriched_obs.x.dtype,
                    ),
                ),
                -1,
            )
            n_node_features_before_position_feature = (
                enriched_obs.x.shape[1] - self.n_graph2node_components
            )
        else:
            enriched_obs = None
            n_node_features_before_position_feature = 0

        # value
        latent_vf = self.mlp_extractor_value.forward_critic(features)
        values = self.value_net(latent_vf)

        # reset action distrib to forget previous marginals parametrization/masking
        # (important as we skip the end of the loop if no more components has a meaning)
        self._reset_action_dist()

        # action components logits and distribution
        for (
            i_action_component,
            (
                mlp_extractor_action_component,
                action_component_net,
                action_component_node_flag_index,
            ),
        ) in enumerate(
            zip(
                self.mlp_extractors_action,
                self.action_nets,
                self.action_components_node_flag_indices,
            )
        ):
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

            # compute logits only for relevant samples
            if compute_all_samples:
                # all samples are relevant
                if i_action_component < self._n_graphindependent_components:
                    # graph-independent component: mlp from features
                    latent_pi = mlp_extractor_action_component.forward_actor(
                        full_features[:, : n_features_by_component[i_action_component]]
                    )
                    action_component_full_logits = action_component_net(latent_pi)
                else:
                    # graph->node GNN from observation graph enriched with previous graph-independent components
                    # + position of previous graph node components
                    action_component_logits_graph: thg.data.Data = action_component_net(
                        enriched_obs
                    )
                    if action_component_node_flag_index is not None:
                        # keep only nodes with proper flag
                        nodes_to_keep = enriched_obs.x[
                            :, action_component_node_flag_index
                        ].nonzero(as_tuple=True)
                    else:
                        nodes_to_keep = None
                    action_component_full_logits = unbatch_node_logits(
                        action_component_logits_graph, nodes_to_keep=nodes_to_keep
                    )
            else:
                # only some samples are relevant
                if i_action_component < self._n_graphindependent_components:
                    # graph-independent component: mlp from features
                    latent_pi = mlp_extractor_action_component.forward_actor(
                        full_features[
                            indices_samples_with_component,
                            : n_features_by_component[i_action_component],
                        ]
                    )
                    action_component_logits = action_component_net(latent_pi)
                else:
                    # graph->node GNN directly from observation graph enriched with previous components
                    # extract relevant samples from batch
                    relevant_enriched_obs = thg.data.Batch.from_data_list(
                        enriched_obs.index_select(indices_samples_with_component)
                    )
                    action_component_logits_graph: thg.data.Data = action_component_net(
                        relevant_enriched_obs
                    )
                    if action_component_node_flag_index is not None:
                        # keep only nodes with proper flag
                        nodes_to_keep = relevant_enriched_obs.x[
                            :, action_component_node_flag_index
                        ].nonzero(as_tuple=True)
                    else:
                        nodes_to_keep = None
                    action_component_logits = unbatch_node_logits(
                        action_component_logits_graph, nodes_to_keep=nodes_to_keep
                    )
                # fill logits with 0 for irrelevant samples  (will be masked afterwards)
                action_component_full_logits = th.zeros_like(
                    action_component_mask, dtype=action_component_logits.dtype
                )
                action_component_full_logits[
                    indices_samples_with_component, :
                ] = action_component_logits

            # param and mask distributions
            self.action_dist.set_proba_distribution_component(
                i_component=i_action_component,
                action_component_logits=action_component_full_logits,
            )
            self.action_dist.apply_masking_component(
                i_component=i_action_component, component_masks=action_component_mask
            )

            # prepare enriched_obs for next component by adding the encoded position of the current component nodes
            if (
                i_action_component >= self._n_graphindependent_components
            ):  # the component is a graph2node one
                position = (
                    # position of component among graph node components:
                    i_action_component
                    - self._n_graphindependent_components
                )
                actions_component = actions[:, i_action_component]
                node_shift_by_sample = enriched_obs._slice_dict["x"][
                    : enriched_obs.num_graphs
                ]
                component_nodes = (
                    (  # shift node ids to match node ids in the batched graph
                        actions_component + node_shift_by_sample.to(self.device)
                    )[actions_component >= 0].long()
                )  # keep only relevant samples (remove -1 components)
                # NB: avoid torch.autograd.backward issue, we cannot change inplace node features
                # (as already part of previous component gradient computation)
                enriched_obs.x = copy.copy(enriched_obs.x)
                enriched_obs.x[
                    component_nodes, n_node_features_before_position_feature + position
                ] = 1

        log_prob = self.action_dist.log_prob(actions)
        entropy = self.action_dist.entropy()
        return values, log_prob, entropy

    def get_distribution(
        self,
        obs: ObsType,
        action_masks: Optional[th.Tensor] = None,
    ) -> Distribution:
        raise NotImplementedError()

    def _predict(
        self,
        observation: ObsType,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        # features
        features = self.extract_features(observation)

        # action prediction as in forward()
        return self._compute_distribution_and_sample_action(
            features=features,
            observation=observation,
            deterministic=deterministic,
            action_masks=action_masks,
        )

    def _compute_distribution_and_sample_action(
        self,
        features: th.Tensor,
        observation: ObsType,
        deterministic: bool,
        action_masks: Optional[Union[th.Tensor, np.ndarray]],
    ):
        """Update distribution parameters and masks and sample an action

        Hypotheses:
            - single observation, flat features
            - applicable actions given via action_masks

        Args:
            features:
            deterministic:
            action_masks:

        Returns:

        """
        # hyp: features are flat (1 x features_dim)
        assert len(features.shape) == 2 and features.shape[0] == 1

        if self.n_graph2node_components > 0:
            # hyp: node features are flat and no batch dim
            assert observation.batch is None and len(observation.x.shape) == 2

        # hyp: applicable actions given via action_masks
        assert action_masks is not None
        action_masks = th.as_tensor(action_masks, device=self.device)
        applicable_actions = action_masks.reshape(
            (-1, len(self.action_space.nvec))
        )  # remove batch dimension

        # reset action distrib to forget previous marginals parametrization/masking
        # (important as we skip the end of the loop if no more components has a meaning)
        self._reset_action_dist()

        # init enriched observation (graph + previous action components)
        if self.n_graph2node_components > 0:
            enriched_obs = copy.copy(observation)
            if self._n_graphindependent_components == 0:
                # no graph-independent components to first encode => init node feature encoding position of each node
                # in remaining components
                (
                    n_node_features_before_position_feature,
                    enriched_obs,
                ) = self._add_position_feature_to_graph_obs(enriched_obs=enriched_obs)
        else:
            enriched_obs = None

        # action components
        action_components = th.as_tensor([], device=self.device, dtype=int)
        for (
            i_action_component,
            (
                mlp_extractor_action_component,
                action_component_net,
                action_component_node_flag_index,
            ),
        ) in enumerate(
            zip(
                self.mlp_extractors_action,
                self.action_nets,
                self.action_components_node_flag_indices,
            )
        ):

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

            # compute component logits
            if i_action_component < self._n_graphindependent_components:
                # gnn features extraction + MLP
                latent_pi = mlp_extractor_action_component.forward_actor(features)
                action_component_logits = action_component_net(latent_pi)
            else:
                # graph->node GNN directly from observation graph enriched with previous components
                action_component_logits_graph: thg.data.Data = action_component_net(
                    enriched_obs
                )
                if action_component_node_flag_index is not None:
                    # keep only nodes with proper flag
                    nodes_to_keep = enriched_obs.x[
                        :, action_component_node_flag_index
                    ].nonzero(as_tuple=True)
                else:
                    nodes_to_keep = None
                action_component_logits = unbatch_node_logits(
                    action_component_logits_graph, nodes_to_keep=nodes_to_keep
                )

            # Set marginal distribution dim, logits, and mask
            self.action_dist.set_proba_distribution_component(
                i_component=i_action_component,
                action_component_logits=action_component_logits,
            )
            self.action_dist.apply_masking_component(
                i_component=i_action_component, component_masks=action_component_mask
            )
            # Sample the action component
            action_component = self.action_dist.get_actions_component(
                i_component=i_action_component, deterministic=deterministic
            )
            # Update action components
            action_components = th.cat((action_components, action_component), -1)
            # Update features and graph obs: include onehot encoded last action_component to predict the next component
            if i_action_component + 1 < len(
                self.action_dist.distributions
            ):  # except for last iteration
                if (
                    i_action_component < self._n_graphindependent_components
                ):  # graph-independent component
                    # onehot encoding of the action component
                    onehot_encoded_action_component = F.one_hot(
                        action_component.long(),
                        num_classes=int(self.action_space.nvec[i_action_component]),
                    ).float()
                    if self.n_graph2node_components > 0:
                        # add it to all node features
                        # (ideally would be added as a global feature
                        # but not handled by off-the-shelf GNN from pytorch_geometric)
                        enriched_obs.x = th.cat(
                            (
                                enriched_obs.x,
                                onehot_encoded_action_component.expand(
                                    len(enriched_obs.x), -1
                                ),
                            ),
                            -1,
                        )
                    if i_action_component < self._n_graphindependent_components - 1:
                        # not the last graph-independent one => add it to the features
                        features = th.cat(
                            (
                                features,
                                onehot_encoded_action_component,
                            ),
                            -1,
                        )
                    else:
                        # last graph-independent component => init node feature encoding position of each node
                        # in remaining components
                        (
                            n_node_features_before_position_feature,
                            enriched_obs,
                        ) = self._add_position_feature_to_graph_obs(
                            enriched_obs=enriched_obs
                        )
                else:  # graph-node component
                    # add position among graph node components:
                    # ie a 1 for corresponding node (row) at corresponding position (column)
                    position = i_action_component - self._n_graphindependent_components
                    enriched_obs.x[
                        action_component,
                        n_node_features_before_position_feature + position,
                    ] = 1

        # pad actions with -1 in case of early break of the loop (no more components)
        actions = F.pad(
            action_components,
            pad=(0, len(self.action_dist.distributions) - action_components.shape[-1]),
            value=-1,  # -1 = no component
        )[
            None
        ]  # add a batch dimension

        return actions

    def _add_position_feature_to_graph_obs(
        self, enriched_obs: thg.data.Data
    ) -> tuple[int, thg.data.Data]:
        n_node_features_before_position_feature = enriched_obs.x.shape[1]
        enriched_obs.x = th.cat(
            (
                enriched_obs.x,
                th.zeros(
                    (
                        len(enriched_obs.x),
                        self.n_graph2node_components,
                    ),
                    device=self.device,
                    dtype=enriched_obs.x.dtype,
                ),
            ),
            -1,
        )
        return n_node_features_before_position_feature, enriched_obs


def _encode_actions_first_components_for_features(
    actions: th.Tensor, action_space: spaces.MultiDiscrete, n_components: int
) -> th.Tensor:
    # Tensor concatenation of one hot encodings of first categorical sub-spaces
    if n_components > 0:
        return th.cat(
            [
                F.one_hot(
                    actions[:, i_component].long() + 1,  # -1 -> mapped to 0
                    num_classes=int(action_space.nvec[i_component])
                    + 1,  # new category: -1
                )[
                    :, 1:
                ]  # remove new category
                # we loop over action components (except last one)
                for i_component in range(n_components)
            ],
            dim=-1,
        )
    else:
        # 0 components => empty tensor with proper batch size
        return actions[:, :0]


class AutoregressiveGNNActorCriticPolicy(
    BaseGNNActorCriticPolicy, AutoregressiveActorCriticPolicy
):
    """Policy with autoregressive action prediction for multidiscrete restricted actions and graph observations.

    With no more hypotheses on actions structure, we simply use a GNN for feature extraction.
    See `skdecide.hub.solver.utils.gnn.torch_layers.GraphFeaturesExtractor` for more details.


    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        features_extractor_class: type[BaseFeaturesExtractor] = GraphFeaturesExtractor,
        **kwargs: Any
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            **kwargs,
        )


class AutoregressiveGraph2NodeActorCriticPolicy(AutoregressiveGNNActorCriticPolicy):
    """Policy with autoregressive action prediction from graph observations, last action components being graph nodes.

    This is typically used for parametric action where:
    - first component corresponds to the action type
    - other components are parameters that are themselves nodes of the observation graph

    More precisely we have an argument `n_graph2node_components` which specifies the number of components being nodes
    of the graph. The components are assumed to start with the independent ones then the graph nodes.
    By default, we assume that all components are observation graph nodes except for the first one.

    Same hypotheses as for `AutoregressiveActorCriticPolicy` apply:
    - action space multidiscrete, but with possibility of components at -1 to encode variable length actions.
      (e.g. parametric actions whose arity depends on the action type)
    - not all actions are available depending on environment state
    - the current applicable actions are given via the argument `action_masks`, from `MaskableActorCriticPolicy` API,
      as a variable-length numpy.array's (for a single sample) or a list of such numpy.array's when evaluating
      several samples at once.

    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        action_gnn_class: Optional[type[nn.Module]] = None,
        action_gnn_kwargs: Optional[dict[str, Any]] = None,
        n_graph2node_components: Optional[int] = None,
        **kwargs: Any
    ):
        """

        Args:
            observation_space:
            action_space:
            lr_schedule:
            action_gnn_class: GNN class for graph -> node nets.
                See `skdecide.hub.solver.utils.gnn.torch_layers.Graph2NodeLayer` for default values.
            action_gnn_kwargs: kwargs for `action_gnn_class`
            n_graph2node_components: number of action components that are actually nodes of the obs graph.
                It is assumed that the action components start with the graph-independent ones then the graph nodes.
                By default, this evaluates to len(action_space.nvec) - 1
            **kwargs:
        """
        if not isinstance(action_space, spaces.MultiDiscrete):
            raise ValueError("action_space must be a multidiscrete space.")
        if n_graph2node_components is None:
            self.n_graph2node_components = self.default_n_graph2node_components(
                action_space
            )
        else:
            self.n_graph2node_components = n_graph2node_components
        if action_gnn_kwargs is None:
            self.action_gnn_kwargs = {}
        else:
            self.action_gnn_kwargs = action_gnn_kwargs
        self.action_gnn_class = action_gnn_class
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    @staticmethod
    def default_n_graph2node_components(action_space: spaces.MultiDiscrete) -> int:
        """Default number of action components that are graph nodes if not specified."""
        return len(action_space.nvec) - 1


class AutoregressiveHeteroGraph2NodeActorCriticPolicy(
    AutoregressiveGraph2NodeActorCriticPolicy
):
    """Policy with autoregressive action prediction from hetero graph obs, last action components being graph nodes.

    This is typically used for parametric action where:
    - first component corresponds to the action type which are represented by some nodes of the graph
    - other components are parameters that are themselves other nodes of the observation graph

    As for `AutoregressiveGraph2NodeActorCriticPolicy`, we need to specify the numer of components being nodes of the
    observation graph, but also to know which nodes are of which type. For that we assume that some node features encode
    that: the new argument `heterograph2node_flagfeature_by_component` specifies component by component which binary
    node feature decide the nodes that can be used for each component.
    The resulting action component will be the index of the chosen node among the nodes of the proper type.

    Same hypotheses as for `AutoregressiveGraph2NodeActorCriticPolicy` apply:
    - action space multidiscrete, but with possibility of components at -1 to encode variable length actions.
      (e.g. parametric actions whose arity depends on the action type)
    - action components start with the independent ones then the graph nodes
    - not all actions are available depending on environment state
    - the current applicable actions are given via the argument `action_masks`, from `MaskableActorCriticPolicy` API,
      as a variable-length numpy.array's (for a single sample) or a list of such numpy.array's when evaluating
      several samples at once.

    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        action_components_node_flag_indices: list[Optional[int]],
        action_gnn_class: Optional[type[nn.Module]] = None,
        action_gnn_kwargs: Optional[dict[str, Any]] = None,
        n_graph2node_components: Optional[int] = None,
        **kwargs: Any
    ):
        """

        Args:
            observation_space:
            action_space:
            lr_schedule:
            action_components_node_flag_indices: list, component by component, of integer or None.
              - None: all nodes can be used a priori for the corresponding component.
              - integer: this is the index of the binary node feature to use to extract the relevant nodes for this
                component
            action_gnn_class: GNN class for graph -> node nets.
                See `skdecide.hub.solver.utils.gnn.torch_layers.Graph2NodeLayer` for default values.
            action_gnn_kwargs: kwargs for `action_gnn_class`
            n_graph2node_components: number of action components that are actually nodes of the obs graph.
                It is assumed that the action components start with the independent ones then the one that are nodes.
                By default, all components are assumed to be graph dependent ie
                `n_graph2node_components = len(action_space.nvec)`
            **kwargs:
        """
        if not isinstance(action_space, spaces.MultiDiscrete):
            raise ValueError("action_space must be a multidiscrete space.")

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            action_gnn_class=action_gnn_class,
            action_gnn_kwargs=action_gnn_kwargs,
            n_graph2node_components=n_graph2node_components,
            **kwargs,
        )

        # init action_components_node_flag_indices after super().__init__() to avoid to be overriden
        self.action_components_node_flag_indices = action_components_node_flag_indices

    @staticmethod
    def default_n_graph2node_components(action_space: spaces.MultiDiscrete) -> int:
        """Default number of action components that are graph nodes if not specified."""
        return len(action_space.nvec)
