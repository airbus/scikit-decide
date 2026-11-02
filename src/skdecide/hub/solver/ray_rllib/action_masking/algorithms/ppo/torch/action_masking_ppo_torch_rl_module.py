#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core import Columns
from ray.rllib.utils.typing import TensorType

from skdecide.hub.solver.ray_rllib.action_masking.distribution.mask import (
    mask_categorical_logits,
)
from skdecide.hub.solver.ray_rllib.action_masking.rl_module.base import (
    ActionMaskingRLModule,
)
from skdecide.hub.solver.ray_rllib.action_masking.utils.spaces.space_utils import (
    ACTION_MASK,
)


class ActionMaskingPPOTorchRLModule(ActionMaskingRLModule, DefaultPPOTorchRLModule):
    def _forward(self, batch: dict[str, TensorType], **kwargs) -> dict[str, TensorType]:
        # Preprocess the original batch to extract the action mask.
        action_mask, batch = self.preprocess_batch(batch)
        # Run the forward pass.
        outs = super()._forward(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, action_mask)

    def _forward_train(
        self, batch: dict[str, TensorType], **kwargs
    ) -> dict[str, TensorType]:
        # Run the forward pass.
        outs = super()._forward_train(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, batch[ACTION_MASK])

    def compute_values(self, batch: dict[str, TensorType], embeddings=None):
        # Check, if the action_mask has been extracted from observations or not.
        if ACTION_MASK not in batch:
            # Preprocess the batch to extract the `observations` to `Columns.OBS`.
            action_mask, batch = self.preprocess_batch(batch)
            # NOTE: Because we manipulate the batch we need to add the `action_mask`
            # to the batch to access them in `_forward_train`.
            batch[ACTION_MASK] = action_mask
        # Call the super's method to compute values for GAE.
        return super().compute_values(batch, embeddings)

    def _mask_action_logits(
        self, batch: dict[str, TensorType], action_mask: TensorType
    ) -> dict[str, TensorType]:
        """Masks the action logits for the output of `forward` methods

        Args:
            batch: A dictionary containing tensors (at least action logits).
            action_mask: A tensor containing the action mask for the current
                observations.

        Returns:
            A modified batch with masked action logits for the action distribution
            inputs.
        """
        batch[Columns.ACTION_DIST_INPUTS] = mask_categorical_logits(
            logits=batch[Columns.ACTION_DIST_INPUTS], mask=action_mask
        )
        return batch
