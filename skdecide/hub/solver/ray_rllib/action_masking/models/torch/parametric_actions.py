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
from skdecide.hub.solver.ray_rllib.gnn.models.torch.gnn import GnnBasedModel
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    is_graph_dict_multiinput_space,
    is_graph_dict_space,
)

torch, nn = try_import_torch()


class TorchParametricActionsModel(TorchModelV2, nn.Module):
    """Parametric action model that handles the dot product and masking and
    that also learns action embeddings. PyTorch version.

    This assumes the outputs are logits for a single Categorical action dist.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        nn.Module.__init__(self)
        super(TorchParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw
        )

        self.action_ids_shifted = torch.arange(1, num_outputs + 1, dtype=torch.int64)
        self.true_obs_space = model_config["custom_model_config"]["true_obs_space"]
        self.graph2node = model_config["custom_model_config"].get("graph2node", False)

        if is_graph_dict_space(self.true_obs_space):
            if self.graph2node:
                raise ValueError(
                    "The 'graph2node' mode is not available for TorchParametricActionsModel. "
                    "Use rather the simplified model TorchMaskedActionsModel."
                )
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
            embed_model_obs_space,
            action_space,
            model_config["custom_model_config"]["action_embed_size"],
            model_config,
            name + "_pred_action_embed",
        )

        self.action_embedding = nn.Embedding(
            num_embeddings=num_outputs + 1,
            embedding_dim=model_config["custom_model_config"]["action_embed_size"],
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
        pred_action_embed, _ = self.pred_action_embed_model(
            SampleBatch({SampleBatch.OBS: embed_model_obs})
        )

        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        intent_vector = torch.unsqueeze(pred_action_embed, 1)

        valid_avail_actions = self.action_ids_shifted * valid_avail_actions_mask

        # Embedding for valid available actions which will be learned.
        # Embedding vector for 0 is an invalid embedding (a "dummy embedding").
        valid_avail_actions_embed = self.action_embedding(valid_avail_actions.int())

        # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
        action_logits = torch.sum(valid_avail_actions_embed * intent_vector, dim=2)

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = torch.clamp(
            torch.log(valid_avail_actions_mask), FLOAT_MIN, FLOAT_MAX
        )

        return action_logits + inf_mask, state

    def value_function(self):
        return self.pred_action_embed_model.value_function()
