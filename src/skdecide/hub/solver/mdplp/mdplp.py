# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from skdecide import Domain, Solver
from skdecide.builders.domain import (
    Actions,
    EnumerableTransitions,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    Rewards,
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
    from skdecide.hub.__skdecide_hub_cpp import _MDPLPSolver_ as mdplp_solver
    from skdecide.hub.__skdecide_hub_cpp import _SSPLPSolver_ as ssplp_solver

    class D(
        Domain,
        SingleAgent,
        Sequential,
        EnumerableTransitions,
        Actions,
        Markovian,
        FullyObservable,
        Rewards,
    ):
        pass

    class MDPLP(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """LP solver for general discounted MDPs (primal and dual formulations).

        Enumerates the full reachable state space via BFS, then solves a
        linear program using HiGHS. Supports both primal (value variables)
        and dual (occupation measure) formulations.

        Works with general MDPs using Rewards (no goals required).
        Terminal states are handled via the terminal_value functor.
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            heuristic: Callable[
                [Domain, D.T_state], D.T_agent[Value[D.T_value]]
            ] = lambda d, s: Value(cost=0),
            variant: str = "dual",
            discount: float = 0.99,
            epsilon: float = 0.001,
            lp_infinity: float = 1e20,
            parallel: bool = False,
            callback: Callable[[MDPLP], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct an MDPLP solver instance.

            # Parameters
            domain_factory: Lambda returning a domain instance.
            heuristic: Function h(domain, state) -> Value. Used as initial
                value estimate. Defaults to Value(cost=0).
            variant: LP formulation. "primal" (value variables, one per
                state) or "dual" (occupation measures, one per state-action
                pair). Defaults to "dual".
            discount: Discount factor. Must be < 1 for general MDPs to
                ensure LP feasibility.
            epsilon: Convergence threshold. Defaults to 0.001.
            lp_infinity: Upper bound for LP variable bounds and constraint
                lower bounds used with HiGHS. Defaults to 1e20.
            parallel: If True, explore action transitions in parallel
                during state enumeration.
            callback: Called after solving. Returns True to stop.
            verbose: Enable verbose logging.
            """
            _supported_variants = ("primal", "dual")
            if variant not in _supported_variants:
                raise ValueError(
                    f"MDPLP variant must be one of {_supported_variants}, "
                    f"got '{variant}'."
                )

            if discount >= 1.0:
                raise ValueError(
                    "MDPLP requires discount < 1.0 for LP feasibility. "
                    "For undiscounted SSPs with goals, use a dedicated "
                    "SSP solver (LRTDP, LDFS, etc.)."
                )

            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(self, parallel=parallel)
            self._lambdas = [heuristic]
            self._ipc_notify = True

            self._solver = mdplp_solver(
                solver=self,
                domain=self.get_domain(),
                heuristic=lambda d, s: heuristic(d, s),
                terminal_value=lambda s: Value(cost=0),
                variant=variant,
                discount=discount,
                epsilon=epsilon,
                lp_infinity=lp_infinity,
                parallel=parallel,
                callback=callback,
                verbose=verbose,
            )

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
                    + "No best action found in MDPLP for observation "
                    + str(observation)
                    + ", applying random action"
                    + "\x1b[0m"
                )
                return self.call_domain_method("get_action_space").sample()
            return action

        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            return self._solver.get_utility(observation)

        def get_nb_states(self) -> int:
            return self._solver.get_nb_states()

        def get_nb_lp_variables(self) -> int:
            return self._solver.get_nb_lp_variables()

        def get_nb_lp_constraints(self) -> int:
            return self._solver.get_nb_lp_constraints()

        def get_solving_time(self) -> int:
            return self._solver.get_solving_time()

        def get_explored_states(self) -> set[D.T_agent[D.T_observation]]:
            return self._solver.get_explored_states()

    class D_SSP(
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

    class SSPLP(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """LP solver for undiscounted SSPs (primal and dual formulations).

        Enumerates the full reachable state space via BFS, then solves a
        linear program using HiGHS with discount=1. Goal states have V(g)=0.
        Supports both primal (value variables) and dual (occupation measure)
        formulations.

        Requires Goals and PositiveCosts domain mixins.
        """

        T_domain = D_SSP

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            goal_checker: Callable[
                [Domain, D_SSP.T_state], D_SSP.T_agent[D_SSP.T_predicate]
            ] = lambda d, s: d.is_goal(s),
            heuristic: Callable[
                [Domain, D_SSP.T_state], D_SSP.T_agent[Value[D_SSP.T_value]]
            ] = lambda d, s: Value(cost=0),
            variant: str = "dual",
            epsilon: float = 0.001,
            lp_infinity: float = 1e20,
            parallel: bool = False,
            callback: Callable[[SSPLP], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct an SSPLP solver instance.

            # Parameters
            domain_factory: Lambda returning a domain instance.
            goal_checker: Function gc(domain, state) -> bool. Identifies
                goal states where V(g)=0. Defaults to domain.is_goal().
            heuristic: Function h(domain, state) -> Value. Used as initial
                value estimate. Defaults to Value(cost=0).
            variant: LP formulation. "primal" or "dual". Defaults to "dual".
            epsilon: Convergence threshold. Defaults to 0.001.
            lp_infinity: Upper bound for LP variable bounds and constraint
                lower bounds used with HiGHS. Defaults to 1e20.
            parallel: If True, explore action transitions in parallel
                during state enumeration.
            callback: Called after solving. Returns True to stop.
            verbose: Enable verbose logging.
            """
            _supported_variants = ("primal", "dual")
            if variant not in _supported_variants:
                raise ValueError(
                    f"SSPLP variant must be one of {_supported_variants}, "
                    f"got '{variant}'."
                )

            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(self, parallel=parallel)
            self._lambdas = [goal_checker, heuristic]
            self._ipc_notify = True

            self._solver = ssplp_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: goal_checker(d, s),
                heuristic=lambda d, s: heuristic(d, s),
                variant=variant,
                epsilon=epsilon,
                lp_infinity=lp_infinity,
                parallel=parallel,
                callback=callback,
                verbose=verbose,
            )

        def _solve_from(self, memory: D_SSP.T_memory[D_SSP.T_state]) -> None:
            self._solver.solve(memory)

        def _is_solution_defined_for(
            self, observation: D_SSP.T_agent[D_SSP.T_observation]
        ) -> bool:
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self,
            observation: D_SSP.T_agent[D_SSP.T_observation],
            domain: Optional[Domain] = None,
        ) -> D_SSP.T_agent[D_SSP.T_concurrency[D_SSP.T_event]]:
            action = self._solver.get_next_action(observation)
            if action is None:
                print(
                    "\x1b[3;33;40m"
                    + "No best action found in SSPLP for observation "
                    + str(observation)
                    + ", applying random action"
                    + "\x1b[0m"
                )
                return self.call_domain_method("get_action_space").sample()
            return action

        def _get_utility(
            self, observation: D_SSP.T_agent[D_SSP.T_observation]
        ) -> D_SSP.T_value:
            return self._solver.get_utility(observation)

        def get_nb_states(self) -> int:
            return self._solver.get_nb_states()

        def get_nb_lp_variables(self) -> int:
            return self._solver.get_nb_lp_variables()

        def get_nb_lp_constraints(self) -> int:
            return self._solver.get_nb_lp_constraints()

        def get_solving_time(self) -> int:
            return self._solver.get_solving_time()

        def get_explored_states(self) -> set[D_SSP.T_agent[D_SSP.T_observation]]:
            return self._solver.get_explored_states()

except ImportError:
    print(
        "Scikit-decide C++ hub library not found. "
        "MDPLP/SSPLP solvers require the C++ hub with HiGHS support."
    )
    raise
