#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import torch as th
from torch.nn.functional import pad


def mask_categorical_logits(logits: th.Tensor, mask: th.Tensor) -> th.Tensor:
    """Mask logits for a categorical distribution.

    # Parameters
    logits: potentially batched logits. shape=  batch_shape + (nb_categories,)
    mask: 0-1 valued tensor with same number of dimension as logits with same batch shape and last dimension <= logits last dimension
        Missing values are considered being 0's.

    # Returns
    masked logits.

    """
    # Handle different size of mask and logits
    nb_logits = logits.shape[-1]
    mask_size = mask.shape[-1]
    if mask_size < nb_logits:
        mask = pad(mask, pad=(0, nb_logits - mask_size))
    elif mask_size > nb_logits:
        raise NotImplementedError()

    # log(mask) + clip at float value corresponding to -infty
    minus_infty_approx = th.finfo(logits.dtype).min
    inf_mask = th.clamp(th.log(mask), min=minus_infty_approx)

    # mask logits + clip at float value corresponding to -infty
    return th.clamp(logits + inf_mask, min=minus_infty_approx)
