from collections import defaultdict
from typing import Optional

import gymnasium as gym
import numpy as np
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn

from skdecide.hub.solver.ray_rllib.gnn.torch_layers import GraphFeaturesExtractor
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    convert_dict_space_to_graph_space,
    is_graph_dict_space,
)


class GnnBasedModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: Optional[int],
        model_config: ModelConfigDict,
        name: str,
        **kw,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # config for custom model
        custom_config = defaultdict(
            lambda: None,  # will return None for missing keys
            model_config.get("custom_model_config", {}),
        )

        # gnn-based feature extractor
        features_extractor_kwargs = custom_config.get("features_extractor", {})
        assert is_graph_dict_space(
            obs_space
        ), f"{self.__class__.__name__} can only be applied to Graph observation spaces."
        graph_observation_space = convert_dict_space_to_graph_space(obs_space)
        self.features_extractor = GraphFeaturesExtractor(
            observation_space=graph_observation_space, **features_extractor_kwargs
        )
        self.features_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.features_extractor.features_dim,)
        )

        if num_outputs is None:
            # only feature extraction (e.g. to be used by GraphComplexInputNetwork)
            self.num_outputs = self.features_extractor.features_dim
            self.pred_action_embed_model = None
        else:
            # fully connected network
            self.pred_action_embed_model = FullyConnectedNetwork(
                obs_space=self.features_space,
                action_space=action_space,
                num_outputs=num_outputs,
                model_config=model_config,
                name=name + "_pred_action_embed",
            )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        features = self.features_extractor(obs)
        if self.pred_action_embed_model is None:
            return features, state
        else:
            return self.pred_action_embed_model(
                input_dict={"obs": features},
                state=state,
                seq_lens=seq_lens,
            )

    def value_function(self):
        return self.pred_action_embed_model.value_function()
