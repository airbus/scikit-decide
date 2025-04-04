from functools import partial
from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn
from torch.nn import functional as F

from skdecide.hub.solver.stable_baselines.autoregressive.common.distributions import (
    MultiMaskableCategoricalDistribution,
)
from skdecide.hub.solver.utils.autoregressive.torch_utils import (
    batched_extract_action_component_mask,
    extract_action_component_mask,
)

ACTION_NET_ARCH_KEY = "pi"
VALUE_NET_ARCH_KEY = "vf"


class AutoregressiveActorCriticPolicy(MaskableActorCriticPolicy):
    """Policy with autoregressive action prediction for multidiscrete restricted actions.

    Hypotheses:
    - action space multidiscrete, but with possibility of components at -1 to encode variable length actions.
      (e.g. parametric actions whose arity depends on the action type)
    - not all actions are available depending on environment state
    - the current applicable actions are given via the argument `action_masks`, from `MaskableActorCriticPolicy` API,
      as a variable-length numpy.array's (for a single sample) or a list of such numpy.array's when evaluating
      several samples at once.

    """

    action_space: spaces.MultiDiscrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[dict[str, list[int]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        # Check action space
        if not isinstance(action_space, spaces.MultiDiscrete):
            raise ValueError("action_space must be a multidiscrete space.")
        if not len(np.array(action_space.nvec).shape) == 1:
            raise ValueError(
                "action_space must be a *flat* multidiscrete space. "
                "(i.e action_space.nvec is a 1-d array)."
            )

        if not share_features_extractor:
            raise NotImplementedError()

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self._reset_action_dist()

    def _reset_action_dist(self) -> None:
        """Reset action distribution to ensure all marginal are reset.

        Overwrite also default action distribution (which assume independent marginals)

        """
        self.action_dist = MultiMaskableCategoricalDistribution(
            distributions=[
                MaskableCategoricalDistribution(action_dim=int(action_dim))
                for action_dim in self.action_space.nvec
            ]
        )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # mlp_extractor for value only
        net_arch_value = {VALUE_NET_ARCH_KEY: self.net_arch.get(VALUE_NET_ARCH_KEY, [])}
        self.mlp_extractor_value = MlpExtractor(
            self.features_dim,
            net_arch=net_arch_value,
            activation_fn=self.activation_fn,
            device=self.device,
        )
        self.mlp_extractor = self.mlp_extractor_value  # alias for predict_values()

        # mlp_extractors for each action components
        net_arch_action = {
            ACTION_NET_ARCH_KEY: self.net_arch.get(ACTION_NET_ARCH_KEY, [])
        }
        previous_action_components_features_dims = np.cumsum(
            [0] + list(self.action_space.nvec[:-1])
        )
        self.mlp_extractors_action = [
            MlpExtractor(
                self.features_dim + previous_action_components_features_dim,
                net_arch=net_arch_action,
                activation_fn=self.activation_fn,
                device=self.device,
            )
            for previous_action_components_features_dim in previous_action_components_features_dims
        ]

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        self.value_net = nn.Linear(self.mlp_extractor_value.latent_dim_vf, 1)

        self.action_nets = [
            nn.Linear(mlp_extractor_action.latent_dim_pi, action_component_dim)
            for mlp_extractor_action, action_component_dim in zip(
                self.mlp_extractors_action, self.action_space.nvec
            )
        ]

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor_value: np.sqrt(2),
                **{
                    mlp_extractor: np.sqrt(2)
                    for mlp_extractor in self.mlp_extractors_action
                },
                self.value_net: 1,
                **{action_net: 0.01 for action_net in self.action_nets},
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[Union[th.Tensor, np.ndarray]] = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # features
        features = self.extract_features(obs)

        # value
        latent_vf = self.mlp_extractor_value.forward_critic(features)
        values = self.value_net(latent_vf)

        # actions + distribution update
        actions = self._compute_distribution_and_sample_action_from_features(
            features=features, deterministic=deterministic, action_masks=action_masks
        )

        # logp computation
        log_prob = self.action_dist.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        action_masks: Optional[list[th.Tensor]] = None,
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        # hyp: actions are flat multidiscrete
        assert len(actions.shape) == 2

        # features
        features = self.extract_features(obs)
        # hyp: features are flat
        assert len(features.shape) == 2

        # applicable actions
        # hyp: applicable actions given via action_masks, sample by sample
        applicable_actions = action_masks
        assert (
            applicable_actions is not None
            and len(applicable_actions) == features.shape[0]
            and all(
                len(applicable_actions_by_sample) > 0
                for applicable_actions_by_sample in applicable_actions
            )
        )

        # features + onehot encoded actions (concatenate once for efficiency)
        encoded_actions_wo_last_component = _encode_actions_for_features(
            actions=actions, action_space=self.action_space
        )
        full_features = th.cat((features, encoded_actions_wo_last_component), -1)
        n_features_by_component = features.shape[-1] + np.cumsum(
            [0] + list(self.action_space.nvec[:-1])
        )

        # value
        latent_vf = self.mlp_extractor_value.forward_critic(features)
        values = self.value_net(latent_vf)

        # reset action distrib to forget previous marginals parametrization/masking
        # (important as we skip the end of the loop if no more components has a meaning)
        self._reset_action_dist()

        # action components logits and distribution
        for (
            i_action_component,
            (
                mlp_extractor_action_component,
                action_component_net,
                n_features_component,
            ),
        ) in enumerate(
            zip(
                self.mlp_extractors_action,
                self.action_nets,
                n_features_by_component,
            )
        ):
            # all (batchwise) components for this idx are <0 => no more true components
            if (actions[:, i_action_component] < 0).all():
                # no more computation needed
                break

            # compute logits only for samples with >=0 component
            mask_samples_with_component = actions[:, i_action_component] >= 0
            if mask_samples_with_component.all():
                # all (batchwise) components needed
                compute_all_samples = True
                indices_samples_with_component = None
                mask_samples_with_component = None
            else:
                # some components do need to be computed
                compute_all_samples = False
                indices_samples_with_component = (
                    mask_samples_with_component.nonzero().flatten()
                )

            # mask for current action component considered
            action_component_mask = batched_extract_action_component_mask(
                action_components=actions,
                applicable_actions=applicable_actions,
                i_action_component=i_action_component,
                action_component_dim=self.action_space.nvec[i_action_component],
                nonzeromask_samples_mask=mask_samples_with_component,
            )

            # compute logits only for relevant samples
            if compute_all_samples:
                latent_pi = mlp_extractor_action_component.forward_actor(
                    full_features[:, :n_features_component]
                )
            else:
                latent_pi = mlp_extractor_action_component.forward_actor(
                    full_features[indices_samples_with_component, :n_features_component]
                )
            action_component_logits = action_component_net(latent_pi)
            # fill logits with 0 for irrelevant samples
            if compute_all_samples:
                action_component_full_logits = action_component_logits
            else:
                action_component_full_logits = th.zeros_like(
                    action_component_mask, dtype=action_component_logits.dtype
                )
                action_component_full_logits[
                    indices_samples_with_component, :
                ] = action_component_logits
            # param and mask distributions
            self.action_dist.set_proba_distribution_component(
                i_component=i_action_component,
                action_component_logits=action_component_full_logits,
            )
            self.action_dist.apply_masking_component(
                i_component=i_action_component, component_masks=action_component_mask
            )

        log_prob = self.action_dist.log_prob(actions)
        entropy = self.action_dist.entropy()
        return values, log_prob, entropy

    def get_distribution(
        self,
        obs: PyTorchObs,
        action_masks: Optional[th.Tensor] = None,
    ) -> Distribution:
        raise NotImplementedError()

    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        # features
        features = self.extract_features(observation)

        # action prediction as in forward()
        return self._compute_distribution_and_sample_action_from_features(
            features=features, deterministic=deterministic, action_masks=action_masks
        )

    def _compute_distribution_and_sample_action_from_features(
        self,
        features: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[Union[th.Tensor, np.ndarray]] = None,
    ):
        """Update distribution parameters and masks and sample an action

        Hypotheses:
            - single observation, flat features
            - applicable actions given via action_masks

        Args:
            features:
            deterministic:
            action_masks:

        Returns:

        """
        # hyp: features are flat and only a single sample (1 x features_dim)
        assert len(features.shape) == 2 and features.shape[0] == 1

        # hyp: applicable actions given via action_masks
        assert action_masks is not None
        action_masks = th.as_tensor(action_masks, device=self.device)
        applicable_actions = action_masks.reshape(
            (-1, len(self.action_space.nvec))
        )  # remove batch dimension

        # reset action distrib to forget previous marginals parametrization/masking
        # (important as we skip the end of the loop if no more components has a meaning)
        self._reset_action_dist()

        # action components
        action_components = th.as_tensor([], dtype=int)
        for (
            i_action_component,
            (
                mlp_extractor_action_component,
                action_component_net,
            ),
        ) in enumerate(
            zip(
                self.mlp_extractors_action,
                self.action_nets,
            )
        ):

            # mask for current action component considered
            action_component_mask = extract_action_component_mask(
                action_components=action_components,
                applicable_actions=applicable_actions,
                i_action_component=i_action_component,
                action_component_dim=self.action_space.nvec[i_action_component],
            )

            # more action_component allowed?
            if action_component_mask.sum() == 0:
                # no more components available
                break

            latent_pi = mlp_extractor_action_component.forward_actor(features)
            action_component_logits = action_component_net(latent_pi)
            self.action_dist.set_proba_distribution_component(
                i_component=i_action_component,
                action_component_logits=action_component_logits,
            )
            self.action_dist.apply_masking_component(
                i_component=i_action_component, component_masks=action_component_mask
            )
            action_component = self.action_dist.get_actions_component(
                i_component=i_action_component, deterministic=deterministic
            )
            # Update action components
            action_components = th.cat((action_components, action_component), -1)

            # Update features: include onehot encoded last action_component in features to predict the next component
            if i_action_component + 1 < len(
                self.action_dist.distributions
            ):  # except for last iteration
                features = th.cat(
                    (
                        features,
                        preprocess_obs(
                            action_component,
                            observation_space=spaces.Discrete(
                                self.action_space.nvec[i_action_component]
                            ),
                        ),
                    ),
                    -1,
                )

        actions = F.pad(
            action_components,
            pad=(0, len(self.action_dist.distributions) - action_components.shape[-1]),
            value=-1,  # -1 = no component
        )[
            None
        ]  # add a batch dimension

        return actions


def _encode_actions_for_features(
    actions: th.Tensor, action_space: spaces.MultiDiscrete
):
    # Tensor concatenation of one hot encodings of each Categorical sub-space (dropping last one)
    return th.cat(
        [
            F.one_hot(
                actions[:, i_component].long() + 1,  # -1 -> mapped to 0
                num_classes=int(action_space.nvec[i_component]) + 1,  # new category: -1
            )[
                :, 1:
            ]  # remove new category
            # we loop over action components (except last one)
            for i_component in range(actions.shape[1] - 1)
        ],
        dim=-1,
    )
