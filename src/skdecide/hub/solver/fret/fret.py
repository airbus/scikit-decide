# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
)

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
from skdecide.core import Value

try:
    from skdecide.hub.__skdecide_hub_cpp import _FRETSolver_ as fret_solver

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

    class FRET(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """FRET (Find, Revise, Eliminate Traps) from Kolobov et al.
        (ICAPS 2011).

        FRET is a meta-solver for Generalized Stochastic Shortest Path
        MDPs. It iterates: (1) Find-and-Revise — run an inner solver to
        convergence, (2) Eliminate-Traps — detect traps via Tarjan SCC on
        the greedy graph and adjust values. Converges to V* even in the
        presence of dead ends and 0-cost cycles.
        """

        T_domain = D

        hyperparameters = [
            FloatHyperparameter(name="epsilon"),
            FloatHyperparameter(name="discount"),
            FloatHyperparameter(name="dead_end_cost"),
        ]

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            heuristic: Callable[
                [Domain, D.T_state], D.T_agent[Value[D.T_value]]
            ] = lambda d, s: Value(cost=0),
            inner_solver_factory: Optional[Callable[[], tuple[str, dict]]] = None,
            discount: float = 1.0,
            epsilon: float = 0.001,
            dead_end_cost: float = 10000.0,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[FRET], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a FRET solver instance.

            # Parameters
            domain_factory: Lambda returning a domain instance.
            heuristic: Function h(domain, state) -> Value returning the
                heuristic cost estimate. Defaults to Value(cost=0).
            inner_solver_factory: Callable returning a (name, params) tuple
                specifying the inner solver and its parameters. Available
                inner solvers: "LRTDP", "LDFS", "VI". ILAOstar is not
                supported because it lacks terminal_value, which FRET
                needs to propagate dead-end costs.
                Defaults to ``lambda: ("LRTDP", {})``.
            discount: Value function's discount factor. Defaults to 1.0.
            epsilon: Greedy action tolerance and convergence threshold.
                Defaults to 0.001.
            dead_end_cost: Cost assigned to permanent trap (dead-end)
                states. Defaults to 10000.0.
            parallel: Parallelize the inner solver. Defaults to False.
            shared_memory_proxy: Optional shared memory proxy.
            callback: Called after each FRET iteration. Returns True
                to stop.
            verbose: Enable verbose logging. Defaults to False.
            """
            if inner_solver_factory is None:
                inner_solver_factory = lambda: ("LRTDP", {})
            inner_solver, inner_solver_params = inner_solver_factory()

            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._lambdas = [heuristic]
            self._ipc_notify = True

            self._solver = fret_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: d.is_goal(s),
                heuristic=(
                    (lambda d, s: heuristic(d, s))
                    if not parallel
                    else (lambda d, s: d.call(None, 0, s))
                ),
                discount=discount,
                epsilon=epsilon,
                dead_end_cost=dead_end_cost,
                inner_solver=inner_solver,
                inner_solver_params=inner_solver_params,
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
            action = self._solver.get_next_action(observation)
            if action is None:
                print(
                    "\x1b[3;33;40m"
                    + "No best action found in FRET for observation "
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

        def get_nb_fret_iterations(self) -> int:
            return self._solver.get_nb_fret_iterations()

        def get_nb_traps_eliminated(self) -> int:
            return self._solver.get_nb_traps_eliminated()

        def get_solving_time(self) -> int:
            return self._solver.get_solving_time()

        def get_explored_states(self) -> set:
            return self._solver.get_explored_states()

        def get_dead_end_states(self) -> set:
            return self._solver.get_dead_end_states()

        def get_trapped_sccs(self) -> list[set]:
            return self._solver.get_trapped_sccs()

        def get_policy(
            self,
        ) -> dict[
            D.T_state,
            tuple[D.T_agent[D.T_concurrency[D.T_event]], D.T_value],
        ]:
            return self._solver.get_policy()

except ImportError:
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
