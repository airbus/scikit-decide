#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Union

import numpy as np


def get_item(s: dict[str, Any], keys: tuple[str, ...]) -> Any:
    if len(keys) == 0:
        return s
    else:
        return get_item(s[keys[0]], keys[1:])


def set_item(s: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    if len(keys) == 0:
        raise ValueError("keys must of len >=1")
    elif len(keys) == 1:
        s[keys[0]] = value
    else:
        set_item(s[keys[0]], keys=keys[1:], value=value)


def pad_axis(
    x: np.ndarray, max_dim: int, value: Union[int, float] = 0, axis: int = 0
) -> np.ndarray:
    actual_dim = x.shape[axis]
    pad_width = np.zeros((len(x.shape), 2), dtype=int)
    pad_width[axis, 1] = max_dim - actual_dim
    return np.pad(
        x,
        pad_width=pad_width,
        constant_values=value,
    )
