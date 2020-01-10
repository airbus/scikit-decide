# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module contains base classes for quickly building solvers."""
from __future__ import annotations

from typing import List, Callable

from skdecide.core import D, autocast_all, autocastable
from skdecide.domains import Domain
from skdecide.builders.solver.policy import DeterministicPolicies

__all__ = ['Solver', 'DeterministicPolicySolver']


# MAIN BASE CLASS

class Solver:
    """This is the highest level solver class (inheriting top-level class for each mandatory solver characteristic).

    This helper class can be used as the main base class for solvers.

    Typical use:
    ```python
    class MySolver(Solver, ...)
    ```

    with "..." replaced when needed by a number of classes from following domain characteristics (the ones in
    parentheses are optional):

    - **(assessability)**: Utilities -> QValues
    - **(policy)**: Policies -> UncertainPolicies -> DeterministicPolicies
    - **(restorability)**: Restorable
    """
    T_domain = Domain

    @classmethod
    def get_domain_requirements(cls) -> List[type]:
        """Get domain requirements for this solver class to be applicable.

        Domain requirements are classes from the #skdecide.builders.domain package that the domain needs to inherit from.

        # Returns
        A list of classes to inherit from.
        """
        return cls._get_domain_requirements()

    @classmethod
    def _get_domain_requirements(cls) -> List[type]:
        """Get domain requirements for this solver class to be applicable.

        Domain requirements are classes from the #skdecide.builders.domain package that the domain needs to inherit from.

        # Returns
        A list of classes to inherit from.
        """

        def is_domain_builder(cls):  # detected by having only single-'base class' ancestors until root
            remove_ancestors = []
            while True:
                bases = cls.__bases__
                if len(bases) == 0:
                    return True, remove_ancestors
                elif len(bases) == 1:
                    cls = bases[0]
                    remove_ancestors.append(cls)
                else:
                    return False, []

        i = 0
        sorted_ancestors = list(cls.T_domain.__mro__[:-1])
        while i < len(sorted_ancestors):
            ancestor = sorted_ancestors[i]
            is_builder, remove_ancestors = is_domain_builder(ancestor)
            if is_builder:
                sorted_ancestors = [a for a in sorted_ancestors if a not in remove_ancestors]
                i += 1
            else:
                sorted_ancestors.remove(ancestor)
        return sorted_ancestors

    @classmethod
    def check_domain(cls, domain: Domain) -> bool:
        """Check whether a domain is compliant with this solver type.

        By default, #Solver.check_domain() provides some boilerplate code and internally
        calls #Solver._check_domain_additional() (which returns True by default but can be overridden  to define
        specific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all
        domain requirements are met.

        # Parameters
        domain: The domain to check.

        # Returns
        True if the domain is compliant with the solver type (False otherwise).
        """
        return cls._check_domain(domain)

    @classmethod
    def _check_domain(cls, domain: Domain) -> bool:
        """Check whether a domain is compliant with this solver type.

        By default, #Solver._check_domain() provides some boilerplate code and internally
        calls #Solver._check_domain_additional() (which returns True by default but can be overridden to define specific
        checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all domain
        requirements are met.

        # Parameters
        domain: The domain to check.

        # Returns
        True if the domain is compliant with the solver type (False otherwise).
        """
        check_requirements = all(isinstance(domain, req) for req in cls._get_domain_requirements())
        return check_requirements and cls._check_domain_additional(domain)

    @classmethod
    def _check_domain_additional(cls, domain: D) -> bool:
        """Check whether the given domain is compliant with the specific requirements of this solver type (i.e. the
        ones in addition to "domain requirements").

        This is a helper function called by default from #Solver._check_domain(). It focuses on specific checks, as
        opposed to taking also into account the domain requirements for the latter.

        # Parameters
        domain: The domain to check.

        # Returns
        True if the domain is compliant with the specific requirements of this solver type (False otherwise).
        """
        return True

    def reset(self) -> None:
        """Reset whatever is needed on this solver before running a new episode.

        This function does nothing by default but can be overridden if needed (e.g. to reset the hidden state of a LSTM
        policy network, which carries information about past observations seen in the previous episode).
        """
        return self._reset()

    def _reset(self) -> None:
        """Reset whatever is needed on this solver before running a new episode.

        This function does nothing by default but can be overridden if needed (e.g. to reset the hidden state of a LSTM
        policy network, which carries information about past observations seen in the previous episode).
        """
        pass

    def solve(self, domain_factory: Callable[[], Domain]) -> None:
        """Run the solving process.

        By default, #Solver.solve() provides some boilerplate code and internally calls #Solver._solve(). The
        boilerplate code transforms the domain factory to auto-cast the new domains to the level expected by the solver.

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (can be just a domain class).

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        return self._solve(domain_factory)

    def _solve(self, domain_factory: Callable[[], Domain]) -> None:
        """Run the solving process.

        By default, #Solver._solve() provides some boilerplate code and internally calls #Solver._solve_domain(). The
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

        return self._solve_domain(cast_domain_factory)

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        """Run the solving process.

        This is a helper function called by default from #Solver._solve(), the difference being that the domain factory
        here returns domains auto-cast to the level expected by the solver.

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (auto-cast to expected level).

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        raise NotImplementedError

    @autocastable
    def solve_from(self, memory: D.T_memory[D.T_state]) -> None:
        """Run the solving process from a given state.

        !!! tip
            Create the domain first by calling the @Solver.reset() method

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
            Create the domain first by calling the @Solver.reset() method

        # Parameters
        memory: The source memory (state or history) of the transition.

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        pass


# ALTERNATE BASE CLASSES (for typical combinations)

class DeterministicPolicySolver(Solver, DeterministicPolicies):
    """This is a typical deterministic policy solver class.

    This helper class can be used as an alternate base class for domains, inheriting the following:

    - Solver
    - DeterministicPolicies

    Typical use:
    ```python
    class MySolver(DeterministicPolicySolver)
    ```

    !!! tip
        It is also possible to refine any alternate base class, like for instance:
        ```python
        class MySolver(DeterministicPolicySolver, QValues)
        ```
    """
    pass
