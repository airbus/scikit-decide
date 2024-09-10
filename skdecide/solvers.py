# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module contains base classes for quickly building solvers."""
from __future__ import annotations

from typing import Callable, List, Optional, Type

from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import (
    Hyperparametrizable,
)

from skdecide import autocast_all
from skdecide.builders.solver.fromanystatesolvability import FromInitialState
from skdecide.builders.solver.policy import DeterministicPolicies
from skdecide.domains import Domain

__all__ = ["Solver", "DeterministicPolicySolver"]


# MAIN BASE CLASS


class Solver(Hyperparametrizable, FromInitialState):
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

    _already_autocast = False

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
    ):
        """

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (can be a mere domain class).
            The resulting domain will be auto-cast to the level expected by the solver.

        """

        def cast_domain_factory():
            domain = domain_factory()
            autocast_all(domain, domain, self.T_domain)
            return domain

        self._domain_factory = cast_domain_factory
        self._original_domain_factory = domain_factory

    @classmethod
    def get_domain_requirements(cls) -> List[type]:
        """Get domain requirements for this solver class to be applicable.

        Domain requirements are classes from the #skdecide.builders.domain package that the domain needs to inherit from.

        # Returns
        A list of classes to inherit from.
        """

        def is_domain_builder(
            cls,
        ):  # detected by having only single-'base class' ancestors until root
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
                sorted_ancestors = [
                    a for a in sorted_ancestors if a not in remove_ancestors
                ]
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
        check_requirements = all(
            isinstance(domain, req) for req in cls.get_domain_requirements()
        )
        return check_requirements and cls._check_domain_additional(domain)

    @classmethod
    def _check_domain_additional(cls, domain: Domain) -> bool:
        """Check whether the given domain is compliant with the specific requirements of this solver type (i.e. the
        ones in addition to "domain requirements").

        This is a helper function called by default from #Solver.check_domain(). It focuses on specific checks, as
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

    def _initialize(self):
        """Runs long-lasting initialization code here."""
        pass

    def _cleanup(self):
        """Runs cleanup code here, or code to be executed at the exit of a
        'with' context statement.
        """
        pass

    def __enter__(self):
        """Allow for calling the solver within a 'with' context statement.
        Note that some solvers require such context statements to properly
        clean their status before exiting the Python interpreter, thus it
        is a good habit to always call solvers within a 'with' statement.
        """
        return self

    def __exit__(self, type, value, tb):
        """Allow for calling the solver within a 'with' context statement.
        Note that some solvers require such context statements to properly
        clean their status before exiting the Python interpreter, thus it
        is a good habit to always call solvers within a 'with' statement.
        """
        self._cleanup()

    def autocast(self, domain_cls: Optional[Type[Domain]] = None) -> None:
        """Autocast itself to the level corresponding to the given domain class.

        # Parameters
        domain_cls: the domain class to which level the solver needs to autocast itself.
            By default, use the original domain factory passed to its constructor.

        """
        if not self._already_autocast:
            if domain_cls is None:
                domain_cls = type(self._original_domain_factory())
            autocast_all(self, self.T_domain, domain_cls)
            self._already_autocast = True


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
