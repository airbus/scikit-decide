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
from skdecide.builders.solver import (
    DeterministicPolicies,
    FromAnyState,
    ParallelSolver,
    Utilities,
)
from skdecide.core import Value

try:
    from skdecide.hub.__skdecide_hub_cpp import _EHCSolver_ as ehc_solver

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

    class EHC(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """Enforced Hill Climbing (EHC) solver.

        From the current state, performs a breadth-first search for a state
        with strictly lower heuristic value. Commits the path, then repeats
        from the improved state until a goal is reached.

        When a preferred_actions functor is provided, preferred actions are
        expanded first in each BFS layer before non-preferred ones.
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            heuristic: Callable[
                [Domain, D.T_state], D.T_agent[Value[D.T_value]]
            ] = lambda d, s: Value(cost=0),
            preferred_actions: Optional[
                Callable[
                    [Domain, D.T_state],
                    list[D.T_agent[D.T_concurrency[D.T_event]]],
                ]
            ] = None,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[EHC], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct an EHC solver instance.

            # Parameters
            domain_factory: Lambda returning a domain instance.
            heuristic: Function h(domain, state) -> Value returning the
                heuristic cost estimate. EHC searches for successors with
                strictly lower heuristic value. Defaults to Value(cost=0).
            preferred_actions: Optional function (domain, state) -> list of
                actions to expand first in each BFS layer. When provided,
                preferred actions are tried before non-preferred ones.
                Defaults to None (no preference).
            parallel: Parallelize the search. Defaults to False.
            shared_memory_proxy: The optional shared memory proxy.
                Defaults to None.
            callback: Called after each EHC improvement step, taking the
                solver as argument. Returns True to stop. Defaults to
                never stop.
            verbose: Enable verbose logging. Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._lambdas = [heuristic]
            if preferred_actions is not None:
                self._lambdas.append(preferred_actions)
            self._ipc_notify = True

            self._solver = ehc_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: d.is_goal(s),
                heuristic=(
                    (lambda d, s: heuristic(d, s))
                    if not parallel
                    else (lambda d, s: d.call(None, 0, s))
                ),
                preferred_actions=(
                    (lambda d, s: preferred_actions(d, s))
                    if preferred_actions is not None and not parallel
                    else (
                        (lambda d, s: d.call(None, 1, s))
                        if preferred_actions is not None
                        else None
                    )
                ),
                parallel=parallel,
                callback=callback,
                verbose=verbose,
            )

        def close(self):
            if self._parallel:
                self._solver.close()
            ParallelSolver.close(self)

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
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

        def get_nb_explored_states(self) -> int:
            return self._solver.get_nb_explored_states()

        def get_explored_states(self) -> set:
            return self._solver.get_explored_states()

        def get_solving_time(self) -> int:
            return self._solver.get_solving_time()

        def get_plan(
            self, observation: D.T_agent[D.T_observation]
        ) -> list[
            tuple[
                D.T_agent[D.T_observation],
                D.T_agent[D.T_concurrency[D.T_event]],
                D.T_value,
            ]
        ]:
            return self._solver.get_plan(observation)

        def get_policy(
            self,
        ) -> dict[
            D.T_agent[D.T_observation],
            tuple[D.T_agent[D.T_concurrency[D.T_event]], D.T_value],
        ]:
            return self._solver.get_policy()

except ImportError:
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
