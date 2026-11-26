#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import torch as th
from ray.rllib.core.distribution.distribution import Distribution
from ray.rllib.core.distribution.torch.torch_distribution import TorchCategorical
from ray.rllib.utils.typing import TensorType
from torch.nn.functional import pad


class TorchVariableSizeCategorical(TorchCategorical):
    """Subclass of TorchCategorical able to compare 2 distributions with different number of categories.

    Hypothesis: in that case we consider that we can add missing categories associated to null probability.

    """

    def kl(self, other: Distribution) -> TensorType:
        if not isinstance(other, TorchVariableSizeCategorical):
            return super().kl(other)
        self_logits = self.logits if self.logits is not None else self._dist.logits
        other_logits = other.logits if other.logits is not None else other._dist.logits
        self_nb_categories = self_logits.shape[-1]
        other_nb_categories = other_logits.shape[-1]
        if self_nb_categories == other_nb_categories:
            return super().kl(other)
        else:
            if self_nb_categories < other_nb_categories:
                self_logits = pad(
                    self_logits,
                    pad=(0, other_nb_categories - self_nb_categories),
                    value=th.finfo(self.logits.dtype).min,
                )
                p = type(self).from_logits(self_logits)
                q = other
            else:
                p = self
                other_logits = pad(
                    other_logits,
                    pad=(0, self_nb_categories - other_nb_categories),
                    value=th.finfo(self.logits.dtype).min,
                )
                q = type(other).from_logits(other_logits)
            return p.kl(q)
