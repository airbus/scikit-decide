from collections import defaultdict
from typing import Optional

import gymnasium as gym
import numpy as np
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn

from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    convert_dict_space_to_graph_space,
    is_graph_dict_space,
)
from skdecide.hub.solver.utils.gnn.torch_layers import (
    Graph2NodeLayer,
    GraphFeaturesExtractor,
)
from skdecide.hub.solver.utils.gnn.torch_utils import unbatch_node_logits


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


class GnnBasedGraph2NodeModel(TorchModelV2, nn.Module):
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

        # Check action space
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                "action_space must be a discrete space, an action being represented by the node index."
            )

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

        # prediction of value (no action)
        if num_outputs is None:
            # only feature extraction (e.g. to be used by GraphComplexInputNetwork)
            self.num_outputs = self.features_extractor.features_dim
            self.pred_action_embed_model = None
        else:
            # fully connected network
            self.pred_action_embed_model = FullyConnectedNetwork(
                obs_space=self.features_space,
                action_space=action_space,
                num_outputs=0,  # no action prediction, only value prediction
                model_config=model_config,
                name=name + "_pred_action_embed",
            )

        # action network: pure gnn
        action_net_kwargs = custom_config.get("action_net", {})
        self.action_net = Graph2NodeLayer(
            observation_space=graph_observation_space, **action_net_kwargs
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        # preparation for value_function() later
        features = self.features_extractor(obs)
        embedded_features, _ = self.pred_action_embed_model(
            input_dict={"obs": features},
            state=state,
            seq_lens=seq_lens,
        )

        # action logits prediction
        logits = unbatch_node_logits(self.action_net(obs))

        return logits, state

    def value_function(self):
        return self.pred_action_embed_model.value_function()
