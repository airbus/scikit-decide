from gymnasium.spaces import flatten_space
from ray.rllib import SampleBatch
from ray.rllib.models.torch.fcnet import (
    FullyConnectedNetwork as TorchFullyConnectedNetwork,
)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray, unbatch
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN

from skdecide.hub.solver.ray_rllib.action_masking.utils.spaces.space_utils import (
    ACTION_MASK,
    TRUE_OBS,
)
from skdecide.hub.solver.ray_rllib.gnn.models.torch.complex_input_net import (
    GraphComplexInputNetwork,
)
from skdecide.hub.solver.ray_rllib.gnn.models.torch.gnn import (
    GnnBasedGraph2NodeModel,
    GnnBasedModel,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    is_graph_dict_multiinput_space,
    is_graph_dict_space,
)

torch, nn = try_import_torch()


class TorchMaskedActionsModel(TorchModelV2, nn.Module):
    """Simplified version of TorchParametricActionsModel.

    Instead of embedding the observations into an intermediate space and then map to the correct
    num_ouputs with weights that depends on the applicability of each action,
    we directly embed the observations in a space of dim the expected num_ouputs.
    As we still apply the log-mask onto the action logits, it seems to perform equally (the differences
    made by the tweak of last layer weights according to the applicablitiy of each action are cancelled
    by the mask at the end).

    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.true_obs_space = model_config["custom_model_config"]["true_obs_space"]
        self.graph2node = model_config["custom_model_config"].get("graph2node", False)

        if is_graph_dict_space(self.true_obs_space):
            if self.graph2node:
                pred_action_embed_model_cls = GnnBasedGraph2NodeModel
            else:
                pred_action_embed_model_cls = GnnBasedModel
            self.obs_with_graph = True
            embed_model_obs_space = self.true_obs_space
        elif is_graph_dict_multiinput_space(self.true_obs_space):
            if self.graph2node:
                raise ValueError(
                    "graph2node mode available only for simple Graph space."
                )
            else:
                pred_action_embed_model_cls = GraphComplexInputNetwork
            self.obs_with_graph = True
            embed_model_obs_space = self.true_obs_space
        else:
            pred_action_embed_model_cls = TorchFullyConnectedNetwork
            self.obs_with_graph = False
            embed_model_obs_space = flatten_space(self.true_obs_space)
        self.pred_action_embed_model = pred_action_embed_model_cls(
            obs_space=embed_model_obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name + "_pred_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions mask tensor from the observation.
        valid_avail_actions_mask = input_dict["obs"][ACTION_MASK]

        if self.obs_with_graph:
            # use directly the obs (already converted at proper format by custom `convert_to_torch_tensor`)
            embed_model_obs = input_dict["obs"][TRUE_OBS]
        else:
            # Unbatch true observations before flattening them
            embed_model_obs = torch.stack(
                [
                    flatten_to_single_ndarray(o)
                    for o in unbatch(input_dict["obs"][TRUE_OBS])
                ]
            )

        # Compute the predicted action embedding
        action_logits, _ = self.pred_action_embed_model(
            SampleBatch({SampleBatch.OBS: embed_model_obs})
        )

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = torch.clamp(
            torch.log(valid_avail_actions_mask), FLOAT_MIN, FLOAT_MAX
        )

        return action_logits + inf_mask, state

    def value_function(self):
        return self.pred_action_embed_model.value_function()
