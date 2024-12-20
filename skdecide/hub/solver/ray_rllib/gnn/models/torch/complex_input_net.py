import gymnasium as gym
from ray.rllib import SampleBatch
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType
from torch import nn

from skdecide.hub.solver.ray_rllib.gnn.models.torch.gnn import GnnBasedModel
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    is_graph_dict_space,
)


class GraphComplexInputNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if not model_config.get("_disable_preprocessor_api"):
            raise ValueError(
                "This model is intent to be used only when preprocessors are disabled."
            )
        if not isinstance(obs_space, gym.spaces.Dict):
            raise ValueError(
                "This model is intent to be used only on dict observation space."
            )

        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.gnn = nn.ModuleDict()
        post_graph_obs_subspaces = dict(obs_space.spaces)
        for k, subspace in obs_space.spaces.items():
            if is_graph_dict_space(subspace):
                submodel_name = f"gnn_{k}"
                gnn = GnnBasedModel(
                    obs_space=subspace,
                    action_space=action_space,
                    num_outputs=None,
                    model_config=model_config,
                    framework="torch",
                    name=submodel_name,
                )
                self.add_module(submodel_name, gnn)
                self.gnn[k] = gnn
                post_graph_obs_subspaces[k] = gnn.features_space

        post_graph_obs_space = gym.spaces.Dict(post_graph_obs_subspaces)
        self.post_graph_model = ComplexInputNetwork(
            obs_space=post_graph_obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name="post_graph_model",
        )

    def forward(self, input_dict: SampleBatch, state, seq_lens):
        post_graph_input_dict = input_dict.copy(shallow=True)
        obs = input_dict["obs"]
        post_graph_obs = dict(obs)
        for k, gnn in self.gnn.items():
            post_graph_obs[k] = gnn(SampleBatch({SampleBatch.OBS: obs[k]}))
        post_graph_input_dict["obs"] = post_graph_obs
        return self.post_graph_model(
            input_dict=post_graph_input_dict, state=state, seq_lens=seq_lens
        )

    def value_function(self) -> TensorType:
        return self.post_graph_model.value_function()
