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
    SingleAgent,
    UncertainTransitions,
)
from skdecide.builders.solver import (
    DeterministicPolicies,
    FromAnyState,
    ParallelSolver,
    Utilities,
)
from skdecide.core import Value

try:
    from skdecide.hub.__skdecide_hub_cpp import (
        _SSPReplanSolver_ as sspreplan_solver,
    )

    class D(
        Domain,
        SingleAgent,
        Sequential,
        UncertainTransitions,
        Actions,
        Goals,
        Markovian,
        FullyObservable,
        PositiveCosts,
    ):
        pass

    class SSPReplan(
        ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState
    ):
        """SSP-Replan solver: determinizes a stochastic domain at the transition
        level and replans when the actual outcome deviates from the expected one.

        Supports three determinization strategies:
        - ``"most_probable_outcome"``: picks the most likely successor
        - ``"all_outcomes"``: creates a deterministic action per outcome
        - ``"random_outcome"``: picks a random successor

        Inner deterministic solvers:
        - ``"Astar"``: optimal A* search (default)
        - ``"EHC"``: Enforced Hill Climbing (faster, incomplete)
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], D],
            heuristic: Callable[[D, D.T_state], D.T_agent[Value[D.T_value]]] = lambda d,
            s: Value(cost=0),
            determinization: str = "most_probable_outcome",
            inner_solver_factory: Optional[Callable[[], tuple[str, dict]]] = None,
            max_replans: int = 1000,
            max_steps: int = 10000,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[SSPReplan, Optional[int]], bool] = lambda slv,
            i=None: False,
            verbose: bool = False,
        ) -> None:
            """Construct an SSPReplan solver instance.

            # Parameters
            domain_factory: Lambda to create a domain instance.
            heuristic: Function h(domain, state) -> Value estimating cost-to-go.
                Defaults to Value(cost=0).
            determinization: Determinization strategy. One of
                ``"most_probable_outcome"``, ``"all_outcomes"``,
                ``"random_outcome"``. Defaults to ``"most_probable_outcome"``.
            inner_solver_factory: Callable returning a (name, params) tuple
                specifying the inner solver and its parameters. Available
                inner solvers: "Astar", "EHC".
                Defaults to ``lambda: ("Astar", {})``.
            max_replans: Maximum number of replanning episodes.
                Defaults to 1000.
            max_steps: Maximum total simulation steps. Defaults to 10000.
            parallel: Parallelize domain calls. Defaults to False.
            shared_memory_proxy: Optional shared memory proxy.
            callback: Called after each replan; return True to stop.
                Defaults to never stop.
            verbose: Log progress messages. Defaults to False.
            """
            if inner_solver_factory is None:
                inner_solver_factory = lambda: ("Astar", {})
            inner_solver, inner_solver_params = inner_solver_factory()

            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._lambdas = [heuristic]
            self._ipc_notify = True

            self._solver = sspreplan_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: d.is_goal(s),
                heuristic=lambda d, s: heuristic(d, s),
                determinization=determinization,
                inner_solver=inner_solver,
                max_replans=max_replans,
                max_steps=max_steps,
                inner_solver_params=inner_solver_params,
                parallel=parallel,
                callback=callback,
                verbose=verbose,
            )

        def close(self):
            """Joins parallel domain processes."""
            if self._parallel:
                self._solver.close()
            ParallelSolver.close(self)

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
            """Run the SSP-Replan algorithm from a given state.

            # Parameters
            memory: State from which to start replanning.
            """
            self._solver.solve(memory)

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            """Check whether the policy covers a given state.

            # Parameters
            observation: State to check.

            # Returns
            bool: True if an action is defined for this state.
            """
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self,
            observation: D.T_agent[D.T_observation],
            domain: Optional[Domain] = None,
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            """Get the best action for a given state.

            !!! warning
                Returns a random action if no action is defined in the given
                state.

            # Parameters
            observation: State for which the best action is requested.

            # Returns
            Best action from the computed policy.
            """
            action = self._solver.get_next_action(observation)
            if action is None:
                print(
                    "\x1b[3;33;40m"
                    + "No best action found in observation "
                    + str(observation)
                    + ", applying random action"
                    + "\x1b[0m"
                )
                return self.call_domain_method("get_action_space").sample()
            return action

        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            """Get the cost recorded for a given state.

            # Parameters
            observation: State for which the cost is requested.

            # Returns
            Recorded transition cost, or None if undefined.
            """
            return self._solver.get_utility(observation)

        def get_plan(self):
            """Get the last computed deterministic plan.

            Returns a list of (state, action) tuples representing the planned
            trajectory from the last replan state to the goal.  Empty if no
            plan has been computed yet.
            """
            return self._solver.get_plan()

        def get_nb_replans(self) -> int:
            """Get the number of replanning episodes performed."""
            return self._solver.get_nb_replans()

        def get_nb_steps(self) -> int:
            """Get the total number of simulation steps taken."""
            return self._solver.get_nb_steps()

        def get_solving_time(self) -> int:
            """Get the total solving time in milliseconds."""
            return self._solver.get_solving_time()

        def get_total_cost(self) -> float:
            """Get the accumulated cost along the executed trajectory."""
            return self._solver.get_total_cost()

except ImportError:
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
