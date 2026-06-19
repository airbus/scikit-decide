# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from skdecide import Domain, Solver
from skdecide.builders.domain import (
    Actions,
    DeterministicTransitions,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    Sequential,
    SingleAgent,
)
from skdecide.builders.solver import DeterministicPolicies, Utilities

try:
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_FFSolver_ as CppFFSolver
    from skdecide.hub.domain.pddl.domain import PDDLAction

    class D(
        Domain,
        SingleAgent,
        Sequential,
        DeterministicTransitions,
        Actions,
        Goals,
        Markovian,
        FullyObservable,
        PositiveCosts,
    ):
        pass

    class FF(Solver, DeterministicPolicies, Utilities):
        """FF (Fast-Forward) planning solver for PDDL domains.

        Uses Enforced Hill Climbing (EHC) with the h_FF heuristic and
        helpful actions as described in:

            Hoffmann, J. (2001). FF: The Fast-Forward Planning System.
            AI Magazine, 22(3), 57-62.

        Requires a PDDLDomain or PPDDLDomain.

        When parallel=True, BFS node expansion runs action successors
        in parallel using C++17 parallel algorithms / OpenMP.
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            parallel: bool = False,
            dead_end_cost: float = 1e9,
            callback: Callable[["FF"], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct an FF solver instance.

            # Parameters
            domain_factory: Lambda to create a PDDLDomain or PPDDLDomain
                instance.
            parallel: Parallelize BFS node expansion using C++ parallel
                algorithms. Defaults to False.
            dead_end_cost: Cost assigned to states where FF finds no plan.
                Defaults to 1e9.
            callback: Called after each EHC iteration; return True to stop.
                Defaults to never stop.
            verbose: Log progress messages. Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            self._parallel = parallel
            self._dead_end_cost = dead_end_cost
            self._callback = callback
            self._verbose = verbose

        def _solve(self) -> None:
            domain = self._domain_factory()
            if not hasattr(domain, "_task"):
                raise TypeError("FF solver requires a PDDLDomain or PPDDLDomain")
            self._task = domain._task
            self._cpp_solver = CppFFSolver(
                self,
                self._task,
                self._parallel,
                self._dead_end_cost,
                self._callback,
                self._verbose,
            )
            self._cpp_solver.solve(self._task.initial_state())

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            return self._cpp_solver.is_solution_defined_for(observation.to_cpp())

        def _get_next_action(
            self,
            observation: D.T_agent[D.T_observation],
            domain: Optional[Domain] = None,
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            action = self._cpp_solver.get_next_action(observation.to_cpp())
            if action is None:
                return None
            return PDDLAction(action, self._task)

        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            return None

        def get_nb_explored_states(self) -> int:
            return self._cpp_solver.get_nb_explored_states()

        def get_explored_states(self) -> list:
            return self._cpp_solver.get_explored_states()

        def get_solving_time(self) -> int:
            return self._cpp_solver.get_solving_time()

        def get_plan(self) -> list:
            return self._cpp_solver.get_plan()

except ImportError:
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
