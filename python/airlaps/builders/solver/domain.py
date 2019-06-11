from typing import Optional, Callable, Iterable, List

from airlaps.domains import Domain

__all__ = ['DomainSolver']


class DomainSolver:
    """A solver must inherit this class if it can address domains to provide a solution.

    This class can be considered as the base solver class since every solver should inherit from it.
    """

    def reset(self, domain_factory: Callable[[], Domain]) -> None:
        """Reset the solver to its initial state and set the domain factory.

        By default, #DomainSolver.reset() provides some boilerplate code and internally
        calls #DomainSolver._reset(). The boilerplate code automatically stores the given domain factory into
        the #_domain_factory attribute.

        !!! note
            The domain factory is a callable (typically the domain class, whose constructor would be called), that the
            solver should call every time it needs to (re)instantiate the domain. Some algorithms might only require one
            domain instantiation while others need to interact with many of them in parallel (e.g. some Deep
            Reinforcement Learning ones). This is why a domain factory is required in general and not just a domain
            instance.

        !!! warning
            Resetting the solver means setting all its internal parameters (if any) to their initial value, as if the
            solver were newly instantiated. The reason for this function to exist (rather than just using #__init__()),
            is that it makes its arguments a bit more explicit to the end-user in the collaborative multi-inheritance
            pattern used by AIRLAPS. The drawback is that #DomainSolver.reset() **must be called after instantiating a
            solver.**

        # Parameters
        domain_factory: The domain factory to be called whenever the domain must be (re)instantiated.

        # Example
        ```python
        # Initialize a solver, assuming that MySolver/MyDomain are the AIRLAPS solver/domain classes to use
        solver = MySolver()  # this constructor could have some parameters depending on the solver
        solver.reset(MyDomain)
        ```
        """
        self._domain_factory = domain_factory
        self._reset()

    def _reset(self) -> None:
        """Reset the solver to its initial state.

        This is a helper function called by default from #DomainSolver.reset(). It focuses on the reset itself, whereas
        the latter also stores the domain factory.

        !!! warning
            Resetting the solver means setting all its internal parameters (if any) to their initial value, as if the
            solver were newly instantiated. The reason for this function to exist (rather than just using #__init__()),
            is that it makes its parameters a bit more explicit to the end-user in the collaborative multi-inheritance
            pattern used by AIRLAPS. The drawback is that #DomainSolver.reset() **must be called after instantiating a
            solver.**
        """
        raise NotImplementedError

    def _new_domain(self) -> Domain:
        """Instantiate a new domain and return it.

        This is a helper function that can be called in the solver code whenever needed. New domains are instantiated
        based on the stored domain factory.

        # Returns
        The newly instantiated domain.
        """
        return self._domain_factory()

    def get_domain_requirements(self) -> Iterable[type]:
        """Get domain requirements for this solver to be applicable.

        Domain requirements are classes from the #airlaps.builders.domain package that the domain needs to inherit from.

        # Returns
        An iterable (e.g. a list) of classes to inherit from.

        # Example
        ```python
        from airlaps.builders.domain.dynamics import EnvironmentDomain
        from airlaps.builders.domain.events import UnrestrictedActionDomain
        from airlaps.builders.domain.initialization import InitializableDomain

        # Typical function implementation for a Deep Reinforcement Learning solver (like one from OpenAI Baselines)
        def get_domain_requirements(self):
            return [EnvironmentDomain, UnrestrictedActionDomain, InitializableDomain]
        ```
        """
        raise NotImplementedError

    def check_domain_requirements(self, domain: Optional[Domain] = None) -> List[bool]:
        """Check which domain requirements of the solver are met by the domain.

        If the domain is omitted or None, the check will be done on a new one instantiated
        by #DomainSolver._new_domain().

        # Parameters
        domain: The domain to check (if None, a new one is instantiated by #DomainSolver._new_domain()).

        # Returns
        A list of booleans, where the value at index N is True if the Nth domain requirement is met (False otherwise).
        """
        if domain is None:
            domain = self._new_domain()
        return [isinstance(domain, req) for req in self.get_domain_requirements()]

    def check_domain(self, domain: Optional[Domain] = None) -> bool:
        """Check whether a domain is compliant with the solver.

        If the domain is omitted or None, the check will be done on a new one instantiated
        by #DomainSolver._new_domain().

        By default, #DomainSolver.check_domain() provides some boilerplate code and internally
        calls #DomainSolver._check_domain() (which is used to define specific checks in addition to the "domain
        requirements"). The boilerplate code automatically check whether all domain requirements are met.

        # Parameters
        domain: The domain to check (if None, a new one is instantiated by #DomainSolver._new_domain()).

        # Returns
        True if the domain is compliant with the solver (False otherwise).
        """
        if domain is None:
            domain = self._new_domain()
        return all(self.check_domain_requirements()) and self._check_domain(domain)

    def _check_domain(self, domain: Domain) -> bool:
        """Check whether the given domain is compliant with the solver's specific constraints (i.e. the ones in addition
        to "domain requirements").

        This is a helper function called by default from #DomainSolver.check_domain(). It focuses on specific checks, as
        opposed to taking also into account the domain requirements for the latter.

        # Parameters
        domain: The domain to check.

        # Returns
        True if the domain is compliant with the solver's specific constraints (False otherwise).
        """
        raise NotImplementedError
