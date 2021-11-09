# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, List

__all__ = ["WithPreallocations", "WithoutPreallocations"]


class WithPreallocations:
    """A domain must inherit this class if there are some pre-allocations to consider."""

    def _get_preallocations(
        self,
    ) -> Dict[int, List[str]]:  # TODO: To be handled by domain (applicable actions)
        """
        Return a dictionary where the key is the id of a task (int)
        and the value indicates the pre-allocated resources for this task (as a list of str)"""
        raise NotImplementedError

    def get_preallocations(
        self,
    ) -> Dict[int, List[str]]:  # TODO: To be handled by domain (applicable actions)
        """
        Return a dictionary where the key is the id of a task (int)
        and the value indicates the pre-allocated resources for this task (as a list of str)"""
        return self._get_preallocations()


class WithoutPreallocations(WithPreallocations):
    """A domain must inherit this class if there are no pre-allocations to consider."""

    def _get_preallocations(self) -> Dict[int, List[str]]:
        return {}
