# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Optional

from skdecide import Domain, Solver
from skdecide.builders.domain import (
    Actions,
    EnumerableTransitions,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    Sequential,
    SingleAgent,
)
from skdecide.builders.solver import (
    DeterministicPolicies,
    FromAnyState,
    ParallelSolver,
    Utilities,
)

try:
    from skdecide.hub.__skdecide_hub_cpp import (
        _GPCISolver_ as gpci_solver,
    )

    class D(
        Domain,
        SingleAgent,
        Sequential,
        EnumerableTransitions,
        Actions,
        Goals,
        Markovian,
        FullyObservable,
        PositiveCosts,
    ):
        pass

    class GPCIPhase(IntEnum):
        """Phase of the GPCI algorithm."""

        ENUMERATION = 0
        PROBABILITY = 1
        COST = 2

    class GPCI(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """Goal-Probability and Cost Iteration (GPCI) solver.

        Dual-criterion solver for SSPs with dead-ends
        (Teichteil-Königsbuch, AAAI 2012). Phase 1 maximizes
        goal-reaching probability P*(s) via value iteration. Phase 2
        minimizes expected cost C*(s) conditioned on reaching the goal,
        restricted to probability-preserving actions.
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            goal_checker: Callable[
                [Domain, D.T_state], D.T_agent[D.T_predicate]
            ] = None,
            epsilon: float = 0.001,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[GPCI], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a GPCI solver instance.

            # Parameters
            domain_factory: The lambda function to create a domain instance.
            goal_checker: Function (domain, state) -> bool testing whether a
                state is a goal. Defaults to domain.is_goal().
            epsilon: Maximum residual for convergence in both phases.
                Defaults to 0.001.
            parallel: Parallelize updates on different processes.
                Defaults to False.
            shared_memory_proxy: The optional shared memory proxy.
                Defaults to None.
            callback: Lambda function called at the end of each sweep,
                taking the solver as argument, returning true to stop.
                Defaults to never stop.
            verbose: Whether verbose messages should be logged.
                Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._ipc_notify = True

            if goal_checker is None:
                goal_checker = lambda d, s: d.is_goal(s)

            self._solver = gpci_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=(
                    (lambda d, s: goal_checker(d, s))
                    if not parallel
                    else (lambda d, s: d.call(None, 0, s))
                ),
                epsilon=epsilon,
                parallel=parallel,
                callback=callback,
                verbose=verbose,
            )

        def close(self):
            """Joins the parallel domains' processes."""
            if self._parallel:
                self._solver.close()
            ParallelSolver.close(self)

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
            """Run GPCI from a given state.

            # Parameters
            memory: State from which to enumerate reachable states and run GPCI
            """
            self._solver.solve(memory)

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self,
            observation: D.T_agent[D.T_observation],
            domain: Optional[Domain] = None,
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            if not self._is_solution_defined_for(observation):
                self._solve_from(observation)
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
            else:
                return action

        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            return self._solver.get_utility(observation)

        def get_goal_probability(
            self, observation: D.T_agent[D.T_observation]
        ) -> float:
            """Get the optimal goal-reaching probability P*(s)."""
            return self._solver.get_goal_probability(observation)

        def get_goal_cost(self, observation: D.T_agent[D.T_observation]) -> float:
            """Get the optimal goal-conditioned expected cost C*(s)."""
            return self._solver.get_goal_cost(observation)

        def get_current_phase(self) -> "GPCIPhase":
            """Get the current phase of the GPCI algorithm."""
            return GPCIPhase(self._solver.get_current_phase())

        def get_nb_of_explored_states(self) -> int:
            """Get the number of states discovered by BFS."""
            return self._solver.get_nb_explored_states()

        def get_nb_prob_iterations(self) -> int:
            """Get the number of probability iteration sweeps (phase 1)."""
            return self._solver.get_nb_prob_iterations()

        def get_nb_cost_iterations(self) -> int:
            """Get the number of cost iteration sweeps (phase 2)."""
            return self._solver.get_nb_cost_iterations()

        def get_explored_states(self) -> set[D.T_agent[D.T_observation]]:
            """Get all reachable states discovered by BFS."""
            return self._solver.get_explored_states()

        def get_solving_time(self) -> int:
            """Get the solving time in milliseconds."""
            return self._solver.get_solving_time()

        def get_policy(
            self,
        ) -> dict[
            D.T_agent[D.T_observation],
            tuple[D.T_agent[D.T_concurrency[D.T_event]], float],
        ]:
            """Get the full solution policy."""
            return self._solver.get_policy()

except ImportError:
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
