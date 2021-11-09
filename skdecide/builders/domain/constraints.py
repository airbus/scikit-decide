# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from typing import List

from skdecide.core import Constraint, D, autocastable

__all__ = ["Constrained"]


class Constrained:
    """A domain must inherit this class if it has constraints."""

    @autocastable
    def get_constraints(
        self,
    ) -> List[
        Constraint[
            D.T_memory[D.T_state], D.T_agent[D.T_concurrency[D.T_event]], D.T_state
        ]
    ]:
        """Get the (cached) domain constraints.

        By default, #Constrained.get_constraints() internally calls #Constrained._get_constraints_() the first time and
        automatically caches its value to make future calls more efficient (since the list of constraints is assumed to
        be constant).

        # Returns
        The list of constraints.
        """
        return self._get_constraints()

    @functools.lru_cache()
    def _get_constraints(
        self,
    ) -> List[
        Constraint[
            D.T_memory[D.T_state], D.T_agent[D.T_concurrency[D.T_event]], D.T_state
        ]
    ]:
        """Get the (cached) domain constraints.

        By default, #Constrained._get_constraints() internally calls #Constrained._get_constraints_() the first time and
        automatically caches its value to make future calls more efficient (since the list of constraints is assumed to
        be constant).

        # Returns
        The list of constraints.
        """
        return self._get_constraints_()

    def _get_constraints_(
        self,
    ) -> List[
        Constraint[
            D.T_memory[D.T_state], D.T_agent[D.T_concurrency[D.T_event]], D.T_state
        ]
    ]:
        """Get the domain constraints.

        This is a helper function called by default from #Constrained.get_constraints(), the difference being that the
        result is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        The list of constraints.
        """
        raise NotImplementedError
