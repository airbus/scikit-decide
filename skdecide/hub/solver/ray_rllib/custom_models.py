from gymnasium.spaces import flatten_space
from ray.rllib import SampleBatch
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork as TFFullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.fcnet import (
    FullyConnectedNetwork as TorchFullyConnectedNetwork,
)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray, unbatch
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN

from skdecide.hub.solver.ray_rllib.gnn.models.torch.complex_input_net import (
    GraphComplexInputNetwork,
)
from skdecide.hub.solver.ray_rllib.gnn.models.torch.gnn import GnnBasedModel
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    ACTION_MASK,
    TRUE_OBS,
    is_graph_dict_multiinput_space,
    is_graph_dict_space,
)

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class TFParametricActionsModel(TFModelV2):
    """Parametric action model that handles the dot product and masking and
    that also learns action embeddings. TensorFlow version.

    This assumes the outputs are logits for a single Categorical action dist.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(TFParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw
        )

        self.action_ids_shifted = tf.constant(
            list(range(1, num_outputs + 1)), dtype=tf.float32
        )
        self.true_obs_space = model_config["custom_model_config"]["true_obs_space"]

        self.pred_action_embed_model = TFFullyConnectedNetwork(
            flatten_space(self.true_obs_space),
            action_space,
            model_config["custom_model_config"]["action_embed_size"],
            model_config,
            name + "_pred_action_embed",
        )

        self.action_embedding = tf.keras.layers.Embedding(
            input_dim=num_outputs + 1,
            output_dim=model_config["custom_model_config"]["action_embed_size"],
            name="action_embed_matrix",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions mask tensor from the observation.
        valid_avail_actions_mask = input_dict["obs"][ACTION_MASK]

        # Unbatch true observations before flattening them
        unbatched_true_obs = unbatch(input_dict["obs"][TRUE_OBS])

        # Compute the predicted action embedding
        pred_action_embed, _ = self.pred_action_embed_model(
            {
                "obs": tf.stack(
                    [flatten_to_single_ndarray(o) for o in unbatched_true_obs]
                )
            }
        )

        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        intent_vector = tf.expand_dims(pred_action_embed, 1)

        valid_avail_actions = self.action_ids_shifted * valid_avail_actions_mask

        # Embedding for valid available actions which will be learned.
        # Embedding vector for 0 is an invalid embedding (a "dummy embedding").
        valid_avail_actions_embed = self.action_embedding(valid_avail_actions)

        # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
        action_logits = tf.reduce_sum(valid_avail_actions_embed * intent_vector, axis=2)

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(valid_avail_actions_mask), tf.float32.min)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.pred_action_embed_model.value_function()


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

        if is_graph_dict_space(self.true_obs_space):
            pred_action_embed_model_cls = GnnBasedModel
            self.obs_with_graph = True
            embed_model_obs_space = self.true_obs_space
        elif is_graph_dict_multiinput_space(self.true_obs_space):
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
