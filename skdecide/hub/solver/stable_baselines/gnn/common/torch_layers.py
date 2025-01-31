from typing import Any, Optional, Union

import gymnasium as gym
import torch as th
import torch_geometric as thg
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from torch import nn

from skdecide.hub.solver.utils.gnn import torch_layers


class GraphFeaturesExtractor(BaseFeaturesExtractor):
    """Graph feature extractor for Graph observation spaces.

    Will chain a gnn with a reduction layer to extract a fixed number of features.
    The user can specify both the gnn and reduction layer.

    By default, we use:
    - gnn: a 2-layers GCN
    - reduction layer: global_max_pool + linear layer + relu

    This merely wraps `skdecide.hub.solver.utils.gnn.torch_layers.GraphFeaturesExtractor` to
    makes it a `stable_baselines3.common.torch_layers.BaseFeaturesExtractor`. See the former documentation
    for more precisions about its arguments.

    Args:
        observation_space:
        features_dim: Number of extracted features
            - If reduction_layer_class is given, should match the output of this network.
            - If reduction_layer is None, will be used by the default network as its output dimension.
        gnn_out_dim: dimension of the node embedding in gnn output
            - If gnn is given, should not be None and should match the output of gnn
            - If gnn is not given, will be used to generate it. By default, gnn_out_dim = 2 * features_dim
        gnn_class: GNN network class (for instance chosen from `torch_geometric.nn.models` used to embed the graph observations)
        gnn_kwargs: used by `gnn_class.__init__()`. Without effect if `gnn_class` is None.
        reduction_layer_class: network class to be plugged after the gnn to get a fixed number of features.
        reduction_layer_kwargs: used by `reduction_layer_class.__init__()`. Without effect if `reduction_layer_class` is None.

    """

    def __init__(
        self,
        observation_space: gym.spaces.Graph,
        features_dim: int = 64,
        gnn_out_dim: Optional[int] = None,
        gnn_class: Optional[type[nn.Module]] = None,
        gnn_kwargs: Optional[dict[str, Any]] = None,
        reduction_layer_class: Optional[type[nn.Module]] = None,
        reduction_layer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(observation_space=observation_space, features_dim=features_dim)
        self._extractor = torch_layers.GraphFeaturesExtractor(
            observation_space=observation_space,
            features_dim=features_dim,
            gnn_out_dim=gnn_out_dim,
            gnn_class=gnn_class,
            gnn_kwargs=gnn_kwargs,
            reduction_layer_class=reduction_layer_class,
            reduction_layer_kwargs=reduction_layer_kwargs,
        )

    def forward(self, observations: thg.data.Data) -> th.Tensor:
        return self._extractor.forward(observations=observations)


class CombinedFeaturesExtractor(BaseFeaturesExtractor):
    """Combined features extractor for Dict observation spaces, subspaces potentially including Graph spaces.

    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated.

    Args:
        observation_space:
        cnn_kwargs: to be passed to NatureCNN extractor.
             `cnn_kwargs["normalized_image"] is used to check if the space is an image space
             (see `stable_baselines3.common.torch_layers.NatureCNN`)
        graph_kwargs: to be passed to GraphFeaturesExtractor extractor

    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_kwargs: Optional[dict[str, Any]] = None,
        graph_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if cnn_kwargs is None:
            cnn_kwargs = {}
        if graph_kwargs is None:
            graph_kwargs = {}
        normalized_image = cnn_kwargs.get("normalized_image", False)

        extractors: dict[str, nn.Module] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if isinstance(subspace, gym.spaces.Graph):
                extractors[key] = GraphFeaturesExtractor(subspace, **graph_kwargs)
                total_concat_size += extractors[key].features_dim
            elif is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = NatureCNN(subspace, **cnn_kwargs)
                total_concat_size += extractors[key].features_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        # call __init__ before assigning attributes (but after computing total_concat_size)
        super().__init__(observation_space, features_dim=total_concat_size)
        self.extractors = nn.ModuleDict(extractors)

    def forward(
        self, observations: dict[str, Union[th.Tensor, thg.data.Data]]
    ) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
