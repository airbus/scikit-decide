# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
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
    from skdecide.hub.__skdecide_hub_cpp import _SSiPPSolver_ as ssipp_solver

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

    class SSiPP(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """SSiPP (Short-Sighted Probabilistic Planner) from Trevizan & Veloso
        (ICAPS 2012).

        SSiPP repeatedly builds short-sighted sub-SSPs by BFS to a fixed
        depth t from the current state, solves each optimally with a
        configurable inner solver (LRTDP, ILAOstar, or LDFS), and
        accumulates a global value function across iterations. Boundary
        states at distance t receive V(s) as goal cost, guiding the search
        toward the original goals.

        SSiPP is asymptotically optimal: V converges to V* over relevant
        states as the number of iterations grows.
        """

        T_domain = D

        hyperparameters = [
            IntegerHyperparameter(name="depth"),
            FloatHyperparameter(name="epsilon"),
            FloatHyperparameter(name="discount"),
        ]

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            heuristic: Callable[
                [Domain, D.T_state], D.T_agent[Value[D.T_value]]
            ] = lambda d, s: Value(cost=0),
            depth: int = 3,
            inner_solver: str = "LRTDP",
            inner_solver_params: Optional[dict] = None,
            discount: float = 1.0,
            epsilon: float = 0.001,
            max_iterations: int = 10000,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[SSiPP], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct an SSiPP solver instance.

            # Parameters
            domain_factory: Lambda returning a domain instance.
            heuristic: Function h(domain, state) -> Value returning the
                heuristic cost estimate. Defaults to Value(cost=0).
            depth: Short-sighted depth t. Larger values explore more states
                per sub-SSP but take longer. Defaults to 3.
            inner_solver: Inner optimal solver to use for sub-SSPs. One of
                "LRTDP", "ILAOstar", or "LDFS". Defaults to "LRTDP".
            inner_solver_params: Optional dict of extra parameters forwarded
                to the inner solver's constructor. Keys and value types
                depend on the chosen inner_solver. Refer to the inner
                solver's own documentation for available parameters.
            discount: Value function's discount factor. Defaults to 1.0.
            epsilon: Bellman residual threshold for convergence.
                Defaults to 0.001.
            max_iterations: Maximum number of sub-SSP solve iterations
                per call to solve(). Defaults to 10000.
            parallel: Parallelize the inner solver. Defaults to False.
            shared_memory_proxy: Optional shared memory proxy.
            callback: Called after each sub-SSP solve. Returns True to stop.
            verbose: Enable verbose logging. Defaults to False.
            """
            _supported = ("LRTDP", "ILAOstar", "LDFS")
            if inner_solver not in _supported:
                raise ValueError(
                    f"SSiPP inner_solver must be one of {_supported}, "
                    f"got '{inner_solver}'."
                )

            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._lambdas = [heuristic]
            self._ipc_notify = True

            self._solver = ssipp_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: d.is_goal(s),
                heuristic=(
                    (lambda d, s: heuristic(d, s))
                    if not parallel
                    else (lambda d, s: d.call(None, 0, s))
                ),
                depth=depth,
                discount=discount,
                epsilon=epsilon,
                max_iterations=max_iterations,
                inner_solver_params=inner_solver_params or {},
                inner_solver=inner_solver,
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
                    + "No best action found in SSiPP for observation "
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

        def get_nb_sub_ssps(self) -> int:
            return self._solver.get_nb_sub_ssps()

        def get_solving_time(self) -> int:
            return self._solver.get_solving_time()

        def get_explored_states(self) -> set:
            return self._solver.get_explored_states()

        def get_current_subssp_states(self) -> set:
            return self._solver.get_current_subssp_states()

        def get_boundary_states(self) -> set:
            return self._solver.get_boundary_states()

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
