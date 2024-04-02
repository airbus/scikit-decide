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
        domain_factory: Callable[[], Domain],
    ) -> None:
        """Run the solving process.

        By default, #FromInitialState.solve() provides some boilerplate code and internally calls #FromInitialState._solve(). The
        boilerplate code transforms the domain factory to auto-cast the new domains to the level expected by the solver.

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (can be just a domain class).

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """

        def cast_domain_factory():
            domain = domain_factory()
            autocast_all(domain, domain, self.T_domain)
            return domain

        return self._solve(cast_domain_factory)

    def _solve(
        self,
        domain_factory: Callable[[], Domain],
    ) -> None:
        """Run the solving process.

        This is a helper function called by default from #FromInitialState.solve(), the difference being that the domain factory
        here returns domains auto-cast to the level expected by the solver.

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (auto-cast to expected level).

        !!! tip
        domain_factory: A callable with no argument returning the domain to solve (auto-cast to expected level).
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        raise NotImplementedError


class FromAnyState(FromInitialState):
    """A solver must inherit this class if it can solve from any given state."""

    def solve(
        self,
        domain_factory: Callable[[], Domain],
        from_memory: Optional[D.T_memory[D.T_state]] = None,
    ) -> None:
        """Run the solving process.

        By default, #FromInitialState.solve() provides some boilerplate code and internally calls #FromInitialState._solve(). The
        boilerplate code transforms the domain factory to auto-cast the new domains to the level expected by the solver.

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (can be just a domain class).
        from_memory: The source memory (state or history) from which we begin the solving process.
            If None, initial state is used if the domain is initializable, else a ValueError is raised.

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """

        def cast_domain_factory():
            domain = domain_factory()
            autocast_all(domain, domain, self.T_domain)
            return domain

        return self._solve(cast_domain_factory, from_memory=from_memory)

    def _solve(
        self,
        domain_factory: Callable[[], Domain],
        from_memory: Optional[D.T_memory[D.T_state]] = None,
    ) -> None:
        """Run the solving process.

        This is a helper function called by default from #FromInitState.solve(), the difference being that the domain factory
        here returns domains auto-cast to the level expected by the solver.

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (auto-cast to expected level).
        from_memory: The source memory (state or history) from which we begin the solving process.
            If None, initial state is used if the domain is initializable, else a ValueError is raised.

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        self._init_solve(domain_factory=domain_factory)
        if from_memory is None:
            domain = domain_factory()
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

        !!! tip
            Create the domain first by calling the @FromAnyState.init_solve() method

        # Parameters
        memory: The source memory (state or history) of the transition.

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        return self._solve_from(memory)

    def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
        """Run the solving process from a given state.

        !!! tip
            Create the domain first by calling the @FromAnyState._init_solve() method

        # Parameters
        memory: The source memory (state or history) of the transition.

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        raise NotImplementedError

    def init_solve(self, domain_factory: Callable[[], Domain]) -> None:
        """Initialize solver before calling `solve_from()`

        In particular, initialize the underlying domain.

        By default, #FromAnyState.init_solve() provides some boilerplate code and internally calls #FromAnyState._init_solve(). The
        boilerplate code transforms the domain factory to auto-cast the new domains to the level expected by the solver.

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (can be just a domain class).

        """

        def cast_domain_factory():
            domain = domain_factory()
            autocast_all(domain, domain, self.T_domain)
            return domain

        return self._init_solve(cast_domain_factory)

    def _init_solve(self, domain_factory: Callable[[], Domain]) -> None:
        """Initialize solver before calling `solve_from()`

        In particular, initialize the underlying domain.

        This is a helper function called by default from #FromAnyState.init_solve(), the difference being that the domain factory
        here returns domains auto-cast to the level expected by the solver.

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (can be just a domain class).

        """
        raise NotImplementedError
