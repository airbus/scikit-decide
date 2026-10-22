#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import _check_if_diag_gaussian
from ray.rllib.core.distribution.distribution import Distribution
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import (
    FreeLogStdMLPHeadConfig,
    MLPHeadConfig,
    ModelConfig,
)

from skdecide.hub.solver.ray_rllib.autoregressive.distribution.torch.autoregressive import (
    TorchAutoregressive,
)
from skdecide.hub.solver.ray_rllib.autoregressive.models.configs import (
    OneHotEncoderConfig,
)
from skdecide.hub.solver.ray_rllib.autoregressive.utils.spaces.space_utils import (
    GraphNodeMultiDiscrete,
    HeteroGraphNodeMultiDiscrete,
)
from skdecide.hub.solver.ray_rllib.common.constants import TORCH_FRAMEWORK
from skdecide.hub.solver.ray_rllib.gnn.models.configs import Graph2NodeConfig
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    add_onehot_encoded_discrete_feature_to_graph_dict_space,
)


class AutoregressiveCatalog(Catalog):
    action_space: gym.spaces.MultiDiscrete
    observation_space: gym.spaces.Dict  # Flatten graph dict space

    def get_action_dist_cls(self, framework: str) -> type[Distribution]:
        return get_action_dist_cls(framework=framework, action_space=self.action_space)

    def get_action_component_spaces(self) -> list[gym.spaces.Space]:
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            return [
                gym.spaces.Discrete(n=n, start=start)
                for n, start in zip(self.action_space.nvec, self.action_space.start)
            ]
        else:
            raise NotImplementedError()

    def get_action_dist_cls_per_component(
        self, framework: str
    ) -> list[type[Distribution]]:
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            if framework == TORCH_FRAMEWORK:
                return [
                    self._get_dist_cls_from_action_space(
                        action_space=action_component_space, framework=framework
                    )
                    for action_component_space in self.get_action_component_spaces()
                ]
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def _get_action_component_encoder_config(
        cls,
        action_component_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        if isinstance(action_component_space, gym.spaces.Discrete):
            return OneHotEncoderConfig(nb_classes=action_component_space.n)
        else:
            return cls._get_encoder_config(
                observation_space=action_component_space,
                model_config_dict=model_config_dict,
                action_space=action_space,
            )

    def build_action_encoder_per_component(self, framework: str) -> list[Model]:
        if self.nb_graph2node_components > 0:
            nb_components_to_encode = self.nb_graph_independent_components
        else:
            nb_components_to_encode = self.nb_action_components - 1
        self.action_encoder_per_component_configs = [
            self._get_action_component_encoder_config(
                action_component_space=action_component_space,
                model_config_dict=self._model_config_dict,
                action_space=self.action_space,
            )
            for action_component_space in self.get_action_component_spaces()[
                :nb_components_to_encode
            ]
        ]
        return [
            config.build(framework=framework)
            for config in self.action_encoder_per_component_configs
        ]

    def build_pi_head_per_component(self, framework: str) -> list[Model]:
        """Builds the policy head for each action component.

        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The policy heads
            .
        """
        # Get action_distribution_cls to find out about the output dimension for pi_head
        action_distribution_cls_per_component = self.get_action_dist_cls_per_component(
            framework=framework
        )
        action_component_spaces = self.get_action_component_spaces()
        enriched_observation_space = (
            self.observation_space
        )  # graph enriched with encoded previous action components
        pi_heads = []
        self.pi_head_configs = []
        # first component using only observation features
        input_dims = self.latent_dims
        # hyp: features are flattened
        assert len(input_dims) == 1, "observation features are supposed to be flattened"
        for i_component, (
            action_distribution_cls_component,
            action_component_space,
        ) in enumerate(
            zip(action_distribution_cls_per_component, action_component_spaces)
        ):
            if i_component < self.nb_graph_independent_components:
                # "regular" action component
                if self._model_config_dict["free_log_std"]:
                    _check_if_diag_gaussian(
                        action_distribution_cls=action_distribution_cls_component,
                        framework=framework,
                    )
                    is_diag_gaussian = True
                else:
                    is_diag_gaussian = _check_if_diag_gaussian(
                        action_distribution_cls=action_distribution_cls_component,
                        framework=framework,
                        no_error=True,
                    )
                pi_head_config_class = (
                    FreeLogStdMLPHeadConfig
                    if self._model_config_dict["free_log_std"]
                    else MLPHeadConfig
                )
                required_output_dim = (
                    action_distribution_cls_component.required_input_dim(
                        space=action_component_space,
                        model_config=self._model_config_dict,
                    )
                )
                pi_head_config = pi_head_config_class(
                    input_dims=input_dims,
                    hidden_layer_dims=self._model_config_dict["head_fcnet_hiddens"],
                    hidden_layer_activation=self._model_config_dict[
                        "head_fcnet_activation"
                    ],
                    hidden_layer_use_layernorm=self._model_config_dict.get(
                        "head_fcnet_use_layernorm", False
                    ),
                    output_layer_dim=required_output_dim,
                    output_layer_activation="linear",
                    clip_log_std=is_diag_gaussian,
                    log_std_clip_param=self._model_config_dict.get(
                        "log_std_clip_param", 20
                    ),
                )
                pi_heads.append(pi_head_config.build(framework=framework))
                self.pi_head_configs.append(pi_head_config)

                # next components heads will use also this encoded component (if any)
                if i_component < self.nb_action_components - 1:
                    encoded_component_dims = self.action_encoder_per_component_configs[
                        i_component
                    ].output_dims
                    assert len(encoded_component_dims) == 1, (
                        "encoded action components are supposed to be flattened"
                    )
                    input_dims = (input_dims[0] + encoded_component_dims[0],)
                    if self.nb_graph2node_components > 0:
                        # for graph node actions, will be added to node features
                        enriched_observation_space = (
                            add_onehot_encoded_discrete_feature_to_graph_dict_space(
                                enriched_observation_space,
                                new_dim=encoded_component_dims[0],
                            )
                        )
            else:
                # graph node component
                # first graph node component?
                if i_component == self.nb_graph_independent_components:
                    # add position of each component (except last) among graph node components to node features
                    enriched_observation_space = (
                        add_onehot_encoded_discrete_feature_to_graph_dict_space(
                            enriched_observation_space,
                            new_dim=self.nb_graph2node_components - 1,
                        )
                    )
                # heteregeneous graph?
                if self.action_components_node_flag_indices is not None:
                    # index of binary node feature used to filter nodes
                    node_flag_index = self.action_components_node_flag_indices[
                        i_component
                    ]
                else:
                    node_flag_index = None

                pi_head_config = Graph2NodeConfig(
                    observation_space=enriched_observation_space,
                    graph2node_kwargs=self._model_config_dict.get(
                        "graph2node_action_net_kwargs", {}
                    ),
                    node_flag_index=node_flag_index,
                )
                pi_heads.append(pi_head_config.build(framework=framework))
                self.pi_head_configs.append(pi_head_config)

        return pi_heads

    @property
    def nb_graph2node_components(self) -> int:
        """Number of action components being obs graph nodes."""
        if isinstance(self.action_space, GraphNodeMultiDiscrete):
            return self.action_space.nb_graph2node_components
        else:
            return 0

    @property
    def nb_graph_independent_components(self) -> int:
        """Number of action components not being obs graph nodes."""
        if isinstance(self.action_space, GraphNodeMultiDiscrete):
            return self.action_space.nb_graph_independent_components
        else:
            return self.nb_action_components

    @property
    def nb_action_components(self) -> int:
        """Number of action components."""
        return len(self.action_space.nvec)

    @property
    def action_components_node_flag_indices(self) -> None | list[int | None]:
        """Indices of binary node features to filter potential nodes for each component."""
        if isinstance(self.action_space, HeteroGraphNodeMultiDiscrete):
            return self.action_space.action_components_node_flag_indices
        else:
            return None


def get_action_dist_cls(framework: str, action_space: gym.Space) -> type[Distribution]:
    if framework == TORCH_FRAMEWORK:
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            return TorchAutoregressive
        else:
            raise ValueError(
                "action space is supposed to be multidiscrete for autoregressive actions."
            )
    else:
        raise NotImplementedError()
