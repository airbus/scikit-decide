"""This module contains template classes for quickly building solvers."""

from airlaps.builders.solver.domain import DomainSolver
from airlaps.builders.solver.temporality import SolutionSolver

__all__ = ['Solver']


# MAIN TEMPLATE CLASS

class Solver(DomainSolver, SolutionSolver):
    """This is the highest level solver class (inheriting top-level class for each mandatory solver characteristic).

    This helper class can be used as the main template class for solvers.

    Typical use:
    ```python
    class MySolver(Solver, ...)
    ```

    with "..." replaced when needed by a number of classes from following domain characteristics (the ones in
    parentheses are optional):

    - **(assessability)**: UtilitySolver -> QSolver
    - **domain**: DomainSolver
    - **(policy)**: PolicySolver -> UncertainPolicySolver -> DeterministicPolicySolver
    - **(restorability)**: RestorableSolver
    - **temporality**: SolutionSolver -> AnytimeSolver -> RealtimeSolver
    """
    pass
