from typing import Optional, Tuple, Union

import torch as th
from sb3_contrib.common.maskable.distributions import (
    MaskableCategorical,
    MaskableCategoricalDistribution,
    MaskableDistribution,
    MaybeMasks,
)
from stable_baselines3.common.distributions import Distribution, SelfDistribution
from torch import nn


class MultiMaskableCategoricalDistribution(Distribution):
    """Distribution for variable-length multidiscrete actions with partial masking on each component.

    This is meant for autoregressive prediction.

    The distribution is considered as the joint distribution of discrete distributions (MaskableCategoricalDistribution)
    with the possibility to mask each marginal.
    This distribution is meant to be used for autoregressive action:
    - Each component is sampled sequentially
    - The partial mask for the next component is conditioned by the previous components
    - It is possible to have missing components when this has no meaning for the action.
      this corresponds in the simulation to
      - either not initialized marginal (if all samples discard the component)
      - 0 masks for the given sample (the partial mask row corresponding to the sample has only 0's)

    When computing entropy of the distribution or log-probability of an action, we add only contribution
    of marginal distributions for which we have an actual component (dropping the one with a 0-mask).

    As this distribution is used to sample component by component, the sample(), and mode() methods are left
    unimplemented.

    """

    def __init__(self, distributions: list[MaskableCategoricalDistribution]):
        super().__init__()
        self.distributions = distributions
        self._ind_valid_samples_by_distributions: list[
            Optional[tuple[th.Tensor, th.Tensor]]
        ] = [None] * len(distributions)
        self._all_valid_samples_by_distributions: list[bool] = [False] * len(
            distributions
        )
        self._any_valid_samples_by_distributions: list[bool] = [False] * len(
            distributions
        )

    def get_actions_component(
        self, i_component: int, deterministic: bool = False
    ) -> th.Tensor:
        return self.distributions[i_component].get_actions(deterministic=deterministic)

    def apply_masking_component(
        self, i_component: int, component_masks: MaybeMasks
    ) -> None:
        self.distributions[i_component].apply_masking(masks=component_masks)
        # valid samples: at least one 1 in the corresponding mask
        valid_samples = component_masks.sum(-1) > 0
        self._any_valid_samples_by_distributions[i_component] = valid_samples.all()
        self._all_valid_samples_by_distributions[i_component] = valid_samples.any()
        # store valid sample indices if not all valid
        if (
            self._any_valid_samples_by_distributions[i_component]
            and not self._all_valid_samples_by_distributions[i_component]
        ):
            self._ind_valid_samples_by_distributions[
                i_component
            ] = valid_samples.nonzero(as_tuple=True)

    def set_proba_distribution_component(
        self, i_component: int, action_component_logits: th.Tensor
    ) -> None:
        """Fix parameters of the marginal distribution.

        We allow to modify dynamically the marginal dimension by inferring it from
        last dimension of `action_component_logits`.
        This is useful when the dimension of the marginal can change during rollout
        (e.g. when this predict node id's of a graph whose structure vary)

        """
        action_component_dim = action_component_logits.shape[-1]
        self.distributions[i_component].action_dim = action_component_dim
        self.distributions[i_component].proba_distribution(
            action_logits=action_component_logits
        )
        self._any_valid_samples_by_distributions[i_component] = True
        self._all_valid_samples_by_distributions[i_component] = True
        self._ind_valid_samples_by_distributions[i_component] = None

    def get_proba_distribution_component_for_valid_samples(
        self, i_component: int
    ) -> Optional[MaskableCategorical]:
        if not (self._any_valid_samples_by_distributions[i_component]):
            return None
        elif self._all_valid_samples_by_distributions[i_component]:
            return self.distributions[i_component]
        else:
            distribution = self.distributions[i_component]
            ind_valid_samples = self._ind_valid_samples_by_distributions[i_component]
            return MaskableCategorical(
                logits=distribution.distribution.logits[ind_valid_samples],
                masks=distribution.distribution.masks[ind_valid_samples],
            )

    def get_proba_distribution_component_batch_shape(
        self, i_component: int
    ) -> Optional[tuple[int, ...]]:
        distribution = self.distributions[i_component]
        if self.distribution.distribution is None:
            return None
        else:
            return distribution.distribution.logits.shape[:-1]

    def log_prob(self, x: th.Tensor) -> th.Tensor:
        marginal_logps = []
        # loop over marginals but no contribution if not initialized or 0-masked
        for i_component, distribution in enumerate(self.distributions):
            marginal_dist = self.get_proba_distribution_component_for_valid_samples(
                i_component
            )
            if marginal_dist is not None:
                if self._all_valid_samples_by_distributions[i_component]:
                    marginal_logp = marginal_dist.log_prob(x[:, i_component])
                else:
                    # add only contribution for valid samples
                    marginal_logp = th.zeros(
                        self.get_proba_distribution_component_batch_shape(i_component),
                        dtype=x.dtype,
                    )
                    ind_valid_samples = self._ind_valid_samples_by_distributions[
                        i_component
                    ]
                    marginal_logp[ind_valid_samples] = marginal_dist.log_prob(
                        x[ind_valid_samples, i_component]
                    )
                marginal_logps.append(marginal_logp)

        return sum(marginal_logps)

    def entropy(self) -> Optional[th.Tensor]:
        marginal_entropies = []
        # loop over marginals but no contribution if not initialized or 0-masked
        for i_component, distribution in enumerate(self.distributions):
            marginal_dist = self.get_proba_distribution_component_for_valid_samples(
                i_component
            )
            if marginal_dist is not None:
                if self._all_valid_samples_by_distributions[i_component]:
                    marginal_entropy = marginal_dist.entropy()
                else:
                    # add only contribution for valid samples
                    marginal_entropy = th.zeros(
                        self.get_proba_distribution_component_batch_shape(i_component),
                        dtype=marginal_dist.logits.dtype,
                    )
                    ind_valid_samples = self._ind_valid_samples_by_distributions[
                        i_component
                    ]
                    marginal_entropy[ind_valid_samples] = marginal_dist.entropy()
                marginal_entropies.append(marginal_entropy)

        return sum(marginal_entropies)

    def sample(self) -> th.Tensor:
        raise NotImplementedError()

    def mode(self) -> th.Tensor:
        raise NotImplementedError()

    def actions_from_params(self, *args, **kwargs) -> th.Tensor:
        raise NotImplementedError()

    def log_prob_from_params(self, *args, **kwargs) -> Tuple[th.Tensor, th.Tensor]:
        raise NotImplementedError()

    def proba_distribution_net(
        self, *args, **kwargs
    ) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        raise NotImplementedError()

    def proba_distribution(self: SelfDistribution, *args, **kwargs) -> SelfDistribution:
        raise NotImplementedError()
