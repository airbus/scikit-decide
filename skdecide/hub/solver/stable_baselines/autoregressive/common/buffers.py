from typing import Optional

import numpy as np
import torch as th
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer

from skdecide.hub.solver.stable_baselines.common.buffers import (
    MaskableScikitDecideRolloutBufferMixin,
    ScikitDecideRolloutBuffer,
)


class ApplicableActionsRolloutBuffer(
    MaskableScikitDecideRolloutBufferMixin,
    ScikitDecideRolloutBuffer,
    MaskableRolloutBuffer,
):
    """Rollout buffer storing also applicable actions.

    For each step, applicable actions are stored as a numpy array N,M with
    - N: nb of applicable actions
    - M: flattened dim of action space

    As the number of applicable actions vary, we have to use a list of numpy arrays
    instead of a single numpy array to store them in the buffer.

    (And at first it comes as a list of list because of sb3 vectorized environment)

    """

    action_masks: list[np.ndarray]

    def reset(self) -> None:
        super().reset()
        self.action_masks = list()  # actually storing applicable actions

    def _add_action_masks(self, action_masks: Optional[np.ndarray]) -> None:
        if action_masks is None or action_masks.shape[0] > 1:
            raise NotImplementedError()

        self.action_masks.append(action_masks[0])

    def _swap_and_flatten_action_masks(self) -> None:
        # already done when squeezing first dimension in _add_action_masks()
        ...

    def _get_action_masks_samples(self, batch_inds: np.ndarray) -> list[th.Tensor]:
        return [self.to_torch(self.action_masks[idx]) for idx in batch_inds]
