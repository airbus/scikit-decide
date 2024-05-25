# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from skdecide.builders.domain.initialization import Initializable
from skdecide.core import D, autocast_all, autocastable

if TYPE_CHECKING:  # avoids circular imports
    from skdecide.domains import Domain

__all__ = ["FromInitialState", "FromAnyState"]


class FromInitialState:
    """ "A solver must inherit this class if it can solve only from the initial state"""

    def solve(
        self,
    ) -> None:
        """Run the solving process.

        After solving by calling self._solve(), autocast itself so that rollout methods apply
        to the domain original characteristics.

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        self._solve()
        self.autocast()

    def _solve(
        self,
    ) -> None:
        """Run the solving process.

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        raise NotImplementedError


class FromAnyState(FromInitialState):
    """A solver must inherit this class if it can solve from any given state."""

    @autocastable
    def solve(
        self,
        from_memory: Optional[D.T_memory[D.T_state]] = None,
    ) -> None:
        """Run the solving process.

        After solving by calling self._solve(), autocast itself so that rollout methods apply
        to the domain original characteristics.

        # Parameters
        from_memory: The source memory (state or history) from which we begin the solving process.
            If None, initial state is used if the domain is initializable, else a ValueError is raised.

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        self._solve(from_memory=from_memory)
        self.autocast()

    def _solve(
        self,
        from_memory: Optional[D.T_memory[D.T_state]] = None,
    ) -> None:
        """Run the solving process.

        # Parameters
        from_memory: The source memory (state or history) from which we begin the solving process.
            If None, initial state is used if the domain is initializable, else a ValueError is raised.

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        if from_memory is None:
            domain = self._domain_factory()
            if not isinstance(domain, Initializable):
                raise ValueError(
                    "from_memory cannot be None if the domain is not initializable."
                )
            domain.reset()
            from_memory = domain._memory  # updated by domain.reset()

        self._solve_from(from_memory)

    @autocastable
    def solve_from(self, memory: D.T_memory[D.T_state]) -> None:
        """Run the solving process from a given state.

        After solving by calling self._solve_from(), autocast itself so that rollout methods apply
        to the domain original characteristics.

        # Parameters
        memory: The source memory (state or history) of the transition.

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        self._solve_from(memory)
        self.autocast()

    def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
        """Run the solving process from a given state.

        # Parameters
        memory: The source memory (state or history) of the transition.

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        raise NotImplementedError
