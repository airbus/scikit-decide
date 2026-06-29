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
from skdecide.builders.solver import DeterministicPolicies, FromAnyState, Utilities

try:
    from skdecide.hub.__skdecide_hub_cpp import (
        _PPDDLReplanSolver_ as CppPPDDLReplanSolver,
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

    class PPDDLReplan(Solver, DeterministicPolicies, Utilities, FromAnyState):
        """PPDDL-Replan: reactive replanning for probabilistic PDDL (PPDDL)
        domains with a pluggable inner deterministic solver.

        Repeatedly determinizes the stochastic domain at the PDDL effect-tree
        level, plans with the selected inner solver, executes the plan, and
        replans when the actual outcome deviates from the expected
        deterministic outcome.

        Counterpart of FFReplan, but supports pluggable inner solvers via name.

        Three determinization strategies:
        - "most_probable_outcome": use highest-probability outcome
        - "all_outcomes": expand each stochastic action into N deterministic
          actions
        - "random_outcome": sample one outcome randomly (re-sampled each
          replan)

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
            inner_solver_factory: Optional[Callable[[], tuple[str, dict]]] = None,
            determinization: str = "most_probable_outcome",
            max_replans: int = 1000,
            max_steps: int = 10000,
            dead_end_cost: float = 1e9,
            parallel: bool = False,
            callback: Callable[["PPDDLReplan"], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a PPDDLReplan solver instance.

            # Parameters
            domain_factory: Lambda to create a PPDDL domain instance.
            inner_solver_factory: Factory returning (solver_name, params_dict).
                Defaults to ("FF", {}). Use get_available_pddl_inner_solvers()
                to list registered solvers.
            determinization: Determinization strategy. One of
                "most_probable_outcome", "all_outcomes", or
                "random_outcome". Defaults to "most_probable_outcome".
            max_replans: Maximum number of replanning episodes.
                Defaults to 1000.
            max_steps: Maximum total simulation steps. Defaults to 10000.
            dead_end_cost: Cost penalty for dead-end states where the
                inner solver fails. Defaults to 1e9.
            parallel: Parallelize domain evaluation. Defaults to False.
            callback: Called after each replan; return True to stop.
                Defaults to never stop.
            verbose: Log progress messages. Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            inner_solver_name, inner_solver_params = (
                inner_solver_factory()
                if inner_solver_factory is not None
                else ("FF", {})
            )
            self._inner_solver_name = inner_solver_name
            self._inner_solver_params = inner_solver_params
            self._determinization = determinization
            self._max_replans = max_replans
            self._max_steps = max_steps
            self._dead_end_cost = dead_end_cost
            self._parallel = parallel
            self._callback = callback
            self._verbose = verbose

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
            """Run the PPDDL-Replan algorithm from a given PDDL state.

            # Parameters
            memory: PDDL state from which to start replanning.
            """
            if not hasattr(self, "_cpp_solver"):
                domain = self._domain_factory()
                if not hasattr(domain, "_task"):
                    raise TypeError(
                        "PPDDLReplan solver requires a PPDDLDomain (with _task attribute)"
                    )
                self._task = domain._task
                self._cpp_solver = CppPPDDLReplanSolver(
                    self,
                    self._task,
                    self._inner_solver_name,
                    self._determinization,
                    self._parallel,
                    self._dead_end_cost,
                    self._max_replans,
                    self._max_steps,
                    self._callback,
                    self._verbose,
                    self._inner_solver_params,
                )
            self._cpp_solver.solve(memory.to_cpp())

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
            """Get the total number of replanning episodes."""
            return self._cpp_solver.get_nb_replans()

        def get_nb_steps(self) -> int:
            """Get the total number of simulation steps taken."""
            return self._cpp_solver.get_nb_steps()

        def get_solving_time(self) -> int:
            """Get the total solving time in milliseconds."""
            return self._cpp_solver.get_solving_time()

        def get_total_cost(self) -> float:
            """Get the total accumulated cost."""
            return self._cpp_solver.get_total_cost()

except ImportError:
    print(
        "Scikit-decide C++ hub library not found. Please check it is "
        'installed in "skdecide/hub".'
    )
    raise
