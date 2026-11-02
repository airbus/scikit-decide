#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Tuple, Union

import gymnasium as gym
import torch as th
from ray.rllib.core.distribution.distribution import Distribution
from ray.rllib.core.distribution.torch.torch_distribution import TorchCategorical
from ray.rllib.utils.typing import TensorType

from skdecide.hub.solver.ray_rllib.autoregressive.distribution.autoregressive import (
    AutoregressiveDistribution,
)


class TorchAutoregressive(AutoregressiveDistribution):
    """Distribution for variable-length parameteric actions.

    This is meant for autoregressive prediction.

    The distribution is considered as the joint distribution of "marginal" distributions conditioned to previous components
    with the possibility to mask each marginal.
    This distribution is meant to be used for autoregressive action:
    - Each component is sampled sequentially
    - The partial mask for the next component is conditioned by the previous components
    - It is possible to have missing components when this has no meaning for the action.
      this corresponds in the simulation to 0 masks for the given sample
      (the partial mask row corresponding to the sample has only 0's)

    When computing entropy of the distribution or log-probability of an action, we add only contribution
    of marginal distributions for which we have an actual component (dropping the one with a 0-mask).

    As this distribution is used to sample component by component, the sample() and rsample() methods are left
    unimplemented.

    """

    marginal_distribution_cls = TorchCategorical
    distributions: tuple[TorchCategorical, ...]

    def get_proba_distribution_component_batch_shape(
        self, i_component: int
    ) -> tuple[int, ...]:
        distribution: TorchCategorical = self.distributions[i_component]
        if distribution.logits is not None:
            return distribution.logits.shape[:-1]
        else:
            return distribution.probs.shape[:-1]

    def get_proba_distribution_component_for_valid_samples(
        self, i_component: int
    ) -> TorchCategorical:
        return self.get_other_proba_distribution_component_for_valid_samples(
            i_component=i_component, other=self
        )

    def get_other_proba_distribution_component_for_valid_samples(
        self, i_component: int, other: TorchAutoregressive
    ) -> TorchCategorical:
        if self.all_valid_samples_by_distributions[i_component]:
            return other.distributions[i_component]
        else:
            distribution: TorchCategorical = other.distributions[i_component]
            ind_valid_samples = self.ind_valid_samples_by_distributions[i_component]
            return TorchCategorical(logits=distribution.logits[ind_valid_samples])

    def logp(self, value: TensorType, **kwargs) -> TensorType:
        marginal_logps = []
        # loop over marginals but no contribution if not initialized or 0-masked
        for i_component, distribution in enumerate(self.distributions):
            marginal_dist = self.get_proba_distribution_component_for_valid_samples(
                i_component
            )
            if self.all_valid_samples_by_distributions[i_component]:
                marginal_logp = marginal_dist.logp(value[:, i_component])
            else:
                # add only contribution for valid samples
                ind_valid_samples = self.ind_valid_samples_by_distributions[i_component]
                marginal_logp_valid_samples = marginal_dist.logp(
                    value[ind_valid_samples, i_component]
                )
                marginal_logp = th.zeros(
                    distribution.logits.shape[:-1],
                    device=marginal_logp_valid_samples.device,
                    dtype=marginal_logp_valid_samples.dtype,
                )
                marginal_logp.scatter_(
                    dim=0,
                    index=ind_valid_samples,
                    src=marginal_logp_valid_samples,
                )
            marginal_logps.append(marginal_logp)

        return sum(marginal_logps)

    def kl(self, other: "Distribution", **kwargs) -> TensorType:
        marginal_kls = []
        # loop over marginals but no contribution if not initialized or 0-masked
        for i_component, distribution in enumerate(self.distributions):
            marginal_dist = self.get_proba_distribution_component_for_valid_samples(
                i_component
            )
            marginal_dist_other = (
                self.get_other_proba_distribution_component_for_valid_samples(
                    i_component, other=other
                )
            )
            if self.all_valid_samples_by_distributions[i_component]:
                marginal_kl = marginal_dist.kl(marginal_dist_other)
            else:
                # add only contribution for valid samples
                ind_valid_samples = self.ind_valid_samples_by_distributions[i_component]
                marginal_kl_valid_samples = marginal_dist.kl(marginal_dist_other)
                marginal_kl = th.zeros(
                    distribution.logits.shape[:-1],
                    device=marginal_kl_valid_samples.device,
                    dtype=marginal_kl_valid_samples.dtype,
                )
                marginal_kl.scatter_(
                    dim=0,
                    index=ind_valid_samples,
                    src=marginal_kl_valid_samples,
                )
            marginal_kls.append(marginal_kl)

        return sum(marginal_kls)

    def entropy(self, **kwargs) -> TensorType:
        marginal_entropies = []
        # loop over marginals but no contribution if not initialized or 0-masked
        for i_component, distribution in enumerate(self.distributions):
            marginal_dist = self.get_proba_distribution_component_for_valid_samples(
                i_component
            )
            if self.all_valid_samples_by_distributions[i_component]:
                marginal_entropy = marginal_dist.entropy()
            else:
                # add only contribution for valid samples
                ind_valid_samples = self.ind_valid_samples_by_distributions[i_component]
                marginal_entropy_valid_samples = marginal_dist.entropy()
                marginal_entropy = th.zeros(
                    self.get_proba_distribution_component_batch_shape(i_component),
                    device=marginal_entropy_valid_samples.device,
                    dtype=marginal_entropy_valid_samples.dtype,
                )
                marginal_entropy.scatter_(
                    dim=0,
                    index=ind_valid_samples,
                    src=marginal_entropy_valid_samples,
                )
            marginal_entropies.append(marginal_entropy)

        return sum(marginal_entropies)

    def sample(
        self,
        *,
        sample_shape: Tuple[int, ...] = None,
        return_logp: bool = False,
        **kwargs,
    ) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        raise NotImplementedError()

    def rsample(
        self,
        *,
        sample_shape: Tuple[int, ...] = None,
        return_logp: bool = False,
        **kwargs,
    ) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        raise NotImplementedError()

    @staticmethod
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        raise NotImplementedError()
