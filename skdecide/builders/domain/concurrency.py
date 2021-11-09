# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List, Union

__all__ = ["Parallel", "Sequential"]


class Parallel:
    """A domain must inherit this class if multiple events/actions can happen in parallel."""

    T_concurrency = List  # note: Set cannot handle non-hashable events (and Iterable would not provide enough guidance)


class Sequential(Parallel):
    """A domain must inherit this class if its events/actions are sequential (non-parallel)."""

    T_concurrency = Union
