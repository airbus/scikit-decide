#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Optional, Type

import gymnasium as gym
from ray.rllib.core import Columns
from ray.rllib.core.distribution.torch.torch_distribution import TorchDistribution
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.typing import TensorType

from skdecide.hub.solver.ray_rllib.autoregressive.models.catalog import (
    AutoregressiveCatalog,
    get_action_dist_cls,
)
from skdecide.hub.solver.ray_rllib.autoregressive.utils.spaces.space_utils import (
    APPLICABLE_ACTIONS,
    TRUE_OBS,
)
from skdecide.hub.solver.ray_rllib.common.constants import (
    TORCH_FRAMEWORK,
    ForwardPhase,
)


class AutoregressiveTorchRLModule(TorchRLModule):
    catalog: AutoregressiveCatalog

    def __init__(
        self, *args, observation_space: Optional[gym.Space] = None, **kwargs
    ) -> None:
        self.observation_space_with_applicable_actions = observation_space
        # remove applicable actions to init underlying rl module, catalog, obs encoder, ...
        observation_space = observation_space[TRUE_OBS]
        super().__init__(*args, observation_space=observation_space, **kwargs)
        self.action_dist_cls_per_component = (
            self.catalog.get_action_dist_cls_per_component(framework=self.framework)
        )
        self.nb_action_components = len(self.action_dist_cls_per_component)

    def setup(self):
        super().setup()
        # reset observation space to be the actual one for this module
        self.observation_space = self.observation_space_with_applicable_actions

    def get_inference_action_dist_cls(self) -> Type[TorchDistribution]:
        if self.action_dist_cls is not None:
            return self.action_dist_cls
        else:
            return get_action_dist_cls(
                framework=TORCH_FRAMEWORK, action_space=self.action_space
            )

    def get_inference_action_dist_cls_per_component(
        self,
    ) -> list[Type[TorchDistribution]]:
        return self.action_dist_cls_per_component

    def get_exploration_action_dist_cls_per_component(
        self,
    ) -> list[Type[TorchDistribution]]:
        return self.get_inference_action_dist_cls_per_component()

    def get_train_action_dist_cls_per_component(self) -> list[Type[TorchDistribution]]:
        return self.get_inference_action_dist_cls_per_component()

    def get_action_dist_cls_per_component(
        self, phase: ForwardPhase
    ) -> list[Type[TorchDistribution]]:
        match phase:
            case ForwardPhase.INFERENCE:
                return self.get_inference_action_dist_cls_per_component()
            case ForwardPhase.TRAIN:
                return self.get_train_action_dist_cls_per_component()
            case ForwardPhase.EXPLORATION:
                return self.get_exploration_action_dist_cls_per_component()
            case _:
                raise NotImplementedError()

    def get_action_dist_cls(self, phase: ForwardPhase) -> Type[TorchDistribution]:
        match phase:
            case ForwardPhase.INFERENCE:
                return self.get_inference_action_dist_cls()
            case ForwardPhase.TRAIN:
                return self.get_train_action_dist_cls()
            case ForwardPhase.EXPLORATION:
                return self.get_exploration_action_dist_cls()
            case _:
                raise NotImplementedError()

    def preprocess_batch(
        self, batch: dict[str, TensorType], **kwargs
    ) -> tuple[TensorType, dict[str, TensorType]]:
        """Extracts observations and action mask from the batch

        Args:
            batch: A dictionary containing tensors (at least `Columns.OBS`)

        Returns:
            A tuple with the applicable actions and the modified batch containing
                the original observations.
        """
        # Extract the available actions tensor from the observation.
        applicable_actions = batch[Columns.OBS].pop(APPLICABLE_ACTIONS)

        # Modify the batch for the `DefaultPPORLModule`'s `forward` method, i.e.
        # pass only `"obs"` into the `forward` method.
        batch[Columns.OBS] = batch[Columns.OBS].pop(TRUE_OBS)

        # Return the extracted applicable actions and the modified batch.
        return applicable_actions, batch
