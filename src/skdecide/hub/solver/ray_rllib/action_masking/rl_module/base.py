#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Base RL module for action masking.

Derived from `ray.rllib.examples.rl_modules.classes.action_masking_rlm`.

"""

from typing import Optional, Union

import gymnasium as gym
from ray.rllib.core import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.typing import TensorType

from skdecide.hub.solver.ray_rllib.action_masking.utils.spaces.space_utils import (
    ACTION_MASK,
    TRUE_OBS,
)


class ActionMaskingRLModule(RLModule):
    """An RLModule that implements an action masking for safe RL.

    This RLModule implements action masking to avoid unsafe/unwanted actions
    dependent on the current state (observations). It does so by using an
    environment generated action mask defining which actions are allowed and
    which should be avoided. The action mask is extracted from the
    environment's `gymnasium.spaces.dict.Dict` observation and applied after
    the module's `forward`-pass to the action logits. The resulting action
    logits prevent unsafe/unwanted actions to be sampled from the corresponding
    action distribution.

    Note, this RLModule is implemented for the `PPO` algorithm only. It is not
    guaranteed to work with other algorithms. Furthermore, not that for this
    module to work it requires an environment with a `gymnasium.spaces.dict.Dict`
    observation space containing tow key, `"action_mask"` and `"observations"`.
    """

    def __init__(
        self,
        *,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        inference_only: Optional[bool] = None,
        learner_only: bool = False,
        model_config: Optional[Union[dict, DefaultModelConfig]] = None,
        catalog_class=None,
        **kwargs,
    ):
        # If observation space is not of type `Dict` raise an error.
        if not isinstance(observation_space, gym.spaces.dict.Dict):
            raise ValueError(
                "This RLModule requires the environment to provide a "
                "`gym.spaces.Dict` observation space of the form: \n"
                " {'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                "  'observations': self.observation_space}"
            )

        # While the environment holds an observation space that contains, both,
        # the action mask and the original observation space, the 'RLModule'
        # receives only the `"observation"` element of the space, but not the
        # action mask.
        self.observation_space_with_mask = observation_space
        self.observation_space = observation_space[TRUE_OBS]

        # Keeps track if observation specs have been checked already.
        self._checked_observations = False

        # The DefaultPPORLModule, in its constructor will build networks for the
        # original observation space (i.e. without the action mask).
        super().__init__(
            observation_space=self.observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )

    def setup(self):
        super().setup()
        # We need to reset here the observation space such that the
        # super`s observation space is the
        # original space (i.e. without the action mask) and `self`'s
        # observation space contains the action mask.
        self.observation_space = self.observation_space_with_mask

    def preprocess_batch(
        self, batch: dict[str, TensorType], **kwargs
    ) -> tuple[TensorType, dict[str, TensorType]]:
        """Extracts observations and action mask from the batch

        Args:
            batch: A dictionary containing tensors (at least `Columns.OBS`)

        Returns:
            A tuple with the action mask tensor and the modified batch containing
                the original observations.
        """
        # Check observation specs for action mask and observation keys.
        self._check_batch(batch)

        # Extract the available actions tensor from the observation.
        action_mask = batch[Columns.OBS].pop(ACTION_MASK)

        # Modify the batch for the `DefaultPPORLModule`'s `forward` method, i.e.
        # pass only `"obs"` into the `forward` method.
        batch[Columns.OBS] = batch[Columns.OBS].pop(TRUE_OBS)

        # Return the extracted action mask and the modified batch.
        return action_mask, batch

    def _check_batch(self, batch: dict[str, TensorType]) -> None:
        """Assert that the batch includes action mask and observations.

        Args:
            batch: A dicitonary containing tensors (at least `Columns.OBS`) to be
                checked.

        Raises:
            `ValueError` if the column `Columns.OBS`  does not contain observations
                and action mask.
        """
        if not self._checked_observations:
            if ACTION_MASK not in batch[Columns.OBS]:
                raise ValueError(
                    "No action mask found in observation. This `RLModule` requires "
                    "the environment to provide observations that include an "
                    "action mask (i.e. an observation space of the Dict space "
                    "type that looks as follows: \n"
                    "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                    "'observations': self.observation_space}"
                )
            if TRUE_OBS not in batch[Columns.OBS]:
                raise ValueError(
                    "No observations found in observation. This 'RLModule` requires "
                    "the environment to provide observations that include the original "
                    "observations under a key `'observations'` in a dict (i.e. an "
                    "observation space of the Dict space type that looks as follows: \n"
                    "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                    "'observations': <observation_space>}"
                )
            self._checked_observations = True
