# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from skdecide import Domain, Solver
from skdecide.builders.domain import (
    Actions,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    Sequential,
    Simulation,
    SingleAgent,
)
from skdecide.builders.solver import DeterministicPolicies, Utilities

try:
    from skdecide.hub.__skdecide_hub_cpp import (
        _FFReplanSolver_ as CppFFReplanSolver,
    )
    from skdecide.hub.domain.pddl.domain import PDDLAction

    class D(
        Domain,
        SingleAgent,
        Sequential,
        Simulation,
        Actions,
        Goals,
        Markovian,
        FullyObservable,
        PositiveCosts,
    ):
        pass

    class FFReplan(Solver, DeterministicPolicies, Utilities):
        """FF-Replan from Yoon, Fern & Givan (ICAPS 2007): reactive replanning
        for probabilistic PDDL (PPDDL) domains.

        Repeatedly determinizes the stochastic domain at the PDDL effect-tree
        level, plans with FF (EHC + FF heuristic), executes the plan, and
        replans when the actual outcome deviates from the expected
        deterministic outcome.

        Three determinization strategies:
        - "most_probable_outcome": use highest-probability outcome
        - "all_outcomes": expand each stochastic action into N deterministic actions
        - "random_outcome": sample one outcome randomly (re-sampled each replan)

        Requires a PPDDLDomain.

        # Reference
        Yoon, S. W., Fern, A., & Givan, R. (2007). FF-Replan: A Baseline for
        Probabilistic Planning. In *Proceedings of the 17th International
        Conference on Automated Planning and Scheduling (ICAPS)*, pp. 352-359.
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            determinization: str = "most_probable_outcome",
            max_replans: int = 1000,
            max_steps: int = 10000,
            dead_end_cost: float = 1e9,
            parallel: bool = False,
            callback: Callable[["FFReplan"], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct an FFReplan solver instance.

            # Parameters
            domain_factory: Lambda to create a PPDDLDomain instance.
            determinization: Determinization strategy. One of
                "most_probable_outcome", "all_outcomes", or
                "random_outcome". Defaults to "most_probable_outcome".
            max_replans: Maximum number of replanning episodes.
                Defaults to 1000.
            max_steps: Maximum total simulation steps. Defaults to 10000.
            dead_end_cost: Cost penalty for dead-end states where FF
                fails to find a plan. Defaults to 1e9.
            parallel: Parallelize domain evaluation. Defaults to False.
            callback: Called after each replan; return True to stop.
                Defaults to never stop.
            verbose: Log progress messages. Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            self._determinization = determinization
            self._max_replans = max_replans
            self._max_steps = max_steps
            self._dead_end_cost = dead_end_cost
            self._parallel = parallel
            self._callback = callback
            self._verbose = verbose

        def _solve(self) -> None:
            domain = self._domain_factory()
            if not hasattr(domain, "_task"):
                raise TypeError(
                    "FFReplan solver requires a PPDDLDomain (with _task attribute)"
                )
            self._task = domain._task
            self._cpp_solver = CppFFReplanSolver(
                self,
                self._task,
                self._determinization,
                self._parallel,
                self._dead_end_cost,
                self._max_replans,
                self._max_steps,
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
            return self._cpp_solver.get_total_cost()

        def get_plan(self):
            """Get the last computed deterministic plan.

            Returns a list of (PDDLState, PDDLAction) tuples representing the
            planned trajectory from the last replan state to the goal.
            """
            raw = self._cpp_solver.get_plan()
            return [(s, PDDLAction(a, self._task)) for s, a in raw]

        def get_nb_replans(self) -> int:
            return self._cpp_solver.get_nb_replans()

        def get_nb_steps(self) -> int:
            return self._cpp_solver.get_nb_steps()

        def get_solving_time(self) -> int:
            return self._cpp_solver.get_solving_time()

        def get_total_cost(self) -> float:
            return self._cpp_solver.get_total_cost()

except ImportError:
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
