#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch as th
from ray.rllib.core.distribution.distribution import Distribution
from ray.rllib.utils.typing import TensorType


class AutoregressiveDistribution(Distribution):
    """Distribution for variable-length parameteric actions.

    This is meant for autoregressive prediction.

    The distribution is considered as the joint distribution of "marginal" distributions conditioned to previous components
    with the possibility to mask each marginal.
    This distribution is meant to be used for autoregressive action:
    - Each component is sampled sequentially
    - The partial mask for the next component is conditioned by the previous components
    - It is possible to have missing components when this has no meaning for the action.
      this corresponds in the simulation to
      - either not initialized marginal (if all samples discard the component)
      - 0 masks for the given sample (the partial mask row corresponding to the sample has only 0's), for maskable versions

    When computing entropy of the distribution or log-probability of an action, we add only contribution
    of marginal distributions for which we have an actual component (dropping the one with a 0-mask).

    As this distribution is used to sample component by component, the sample(), and mode() methods are left
    unimplemented.

    """

    marginal_distribution_cls: type[Distribution]
    logits_already_masked: bool = True

    def __init__(
        self,
        distributions: tuple[Distribution, ...],
        masks: tuple[th.Tensor, ...] | None = None,
        logits_already_masked=True,
    ):
        """

        # Parameters
        distributions: marginal distributions
        masks: mask for marginal
        logits_already_masked: whether the marginal logits have already been masked or not

        """
        super().__init__()
        self.distributions = distributions
        if not logits_already_masked:
            # todo: update distributions logits with given masks
            raise NotImplementedError()

        if masks is None:
            self.ind_valid_samples_by_distributions: tuple[th.Tensor | None, ...] = (
                None,
            ) * len(distributions)
            self.all_valid_samples_by_distributions: tuple[bool, ...] = (True,) * len(
                distributions
            )

        else:
            all_valid_samples_by_distributions = []
            ind_valid_samples_by_distributions = []
            for i_component, mask_component in enumerate(masks):
                # valid samples: at least one 1 in the corresponding mask
                valid_samples = mask_component.sum(-1) > 0
                any_valid_samples = valid_samples.any()
                if not any_valid_samples:
                    raise ValueError(f"No valid samples for component {i_component}")
                all_valid_samples = valid_samples.all()
                all_valid_samples_by_distributions.append(all_valid_samples)
                # store valid sample indices if not all valid
                if not all_valid_samples:
                    ind_valid_samples_by_distributions.append(
                        valid_samples.nonzero(as_tuple=True)[0]
                    )
                else:
                    ind_valid_samples_by_distributions.append(None)
            self.all_valid_samples_by_distributions = tuple(
                all_valid_samples_by_distributions
            )
            self.ind_valid_samples_by_distributions = tuple(
                ind_valid_samples_by_distributions
            )

    @classmethod
    def from_logits(
        cls,
        logits: tuple[tuple[TensorType, ...], tuple[TensorType, ...] | None],
        **kwargs,
    ) -> AutoregressiveDistribution:
        logits_per_component, masks_per_component = logits
        distributions = tuple(
            cls.marginal_distribution_cls.from_logits(logits=logits_component)
            for logits_component, mask_component in zip(
                logits_per_component, masks_per_component
            )
            if mask_component.any()
        )
        return cls(
            distributions=distributions,
            masks=masks_per_component,
            logits_already_masked=cls.logits_already_masked,
        )
