#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from math import prod
from typing import Any

import gymnasium as gym
from ray.rllib.core.models.configs import ModelConfig

from skdecide.hub.solver.ray_rllib.common.constants import TORCH_FRAMEWORK
from skdecide.hub.solver.ray_rllib.gnn.models.torch.encoder import (
    TorchGnnEncoder,
    TorchMultiinputEncoder,
)
from skdecide.hub.solver.ray_rllib.gnn.models.torch.graph2node import TorchGraph2Node


@dataclass(kw_only=True)
class GnnEncoderConfig(ModelConfig):
    """Configuration for GNN encoder."""

    observation_space: gym.spaces.Dict
    features_dim: int = field(init=False)
    features_extractor_kwargs: dict[str, Any] = field(
        default_factory=dict,
    )
    """Kwargs for `skdecide.hub.solver.utils.gnn.torch_layers.GraphFeaturesExtractor`

    At init, put also features_dim inside. It is then extracted in a specific field.

    """

    def __post_init__(self):
        # remove features_dim from kwargs to guess output dims in advance
        self.features_dim = self.features_extractor_kwargs.pop("features_dim", 64)

    def build(self, framework: str = TORCH_FRAMEWORK) -> TorchGnnEncoder:
        if framework == TORCH_FRAMEWORK:
            return TorchGnnEncoder(self)
        else:
            raise NotImplementedError()

    @property
    def output_dims(self) -> tuple[int, ...]:
        return (self.features_dim,)


@dataclass(kw_only=True)
class MultiinputEncoderConfig(ModelConfig):
    """Configuration for multiinput encoder."""

    subspace_encoder_configs: dict[str, ModelConfig]
    """Configs for encoding each observation subspace."""

    def build(self, framework: str = TORCH_FRAMEWORK) -> TorchGnnEncoder:
        if framework == TORCH_FRAMEWORK:
            return TorchMultiinputEncoder(self)
        else:
            raise NotImplementedError()

    @property
    def output_dims(self) -> tuple[int, ...]:
        return (
            sum(
                prod(encoder_config.output_dims)
                for encoder_config in self.subspace_encoder_configs.values()
            ),
        )


@dataclass(kw_only=True)
class Graph2NodeConfig(ModelConfig):
    """Configuration for GNN2Node layer."""

    observation_space: gym.spaces.Dict
    graph2node_kwargs: dict[str, Any] = field(
        default_factory=dict,
    )
    """Kwargs for `skdecide.hub.solver.utils.gnn.torch_layers.Graph2NodeLayer`."""
    node_flag_index: int | None = None
    """Index of node feature to consider to preselect possible nodes, in case of heterogeneous graph."""
    keep_graph_embedding: bool = False
    """Whether to keep graph embeddings in output."""

    def build(self, framework: str) -> TorchGraph2Node:
        return TorchGraph2Node(self)
