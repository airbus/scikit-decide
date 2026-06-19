# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from skdecide import DiscreteDistribution, Domain, Solver
from skdecide.builders.domain import (
    Actions,
    Constrained,
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
    UncertainPolicies,
    Utilities,
)
from skdecide.core import BoundConstraint, Value

try:
    from skdecide.hub.__skdecide_hub_cpp import _CIDualSolver_ as cidual_solver
    from skdecide.hub.__skdecide_hub_cpp import _IDualSolver_ as idual_solver

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

    class IDual(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """Heuristic search in dual space for SSPs (i-dual algorithm).

        Incrementally builds the search space from the initial state,
        solving growing dual LPs with heuristic terminal costs on fringe
        states. Focuses on promising regions reachable from s0, unlike
        full-enumeration LP solvers.

        Based on: Trevizan et al., "Heuristic Search in Dual Space for
        Constrained Stochastic Shortest Path Problems", ICAPS 2016.
        """

        T_domain = D_SSP

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            goal_checker: Callable[
                [Domain, D_SSP.T_state], D_SSP.T_agent[D_SSP.T_predicate]
            ] = lambda d, s: d.is_goal(s),
            heuristic: Callable[
                [Domain, D_SSP.T_state],
                D_SSP.T_agent[Value[D_SSP.T_value]],
            ] = lambda d, s: Value(cost=0),
            terminal_value: Callable[
                [D_SSP.T_state], D_SSP.T_agent[Value[D_SSP.T_value]]
            ] = lambda s: Value(cost=1000.0),
            lp_infinity: float = 1e20,
            lp_tolerance: float = 1e-15,
            default_dead_end_cost: float = 1000.0,
            lp_callback_interval: int = 0,
            parallel: bool = False,
            callback: Callable[[IDual], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct an IDual solver instance.

            # Parameters
            domain_factory: Lambda returning a domain instance.
            goal_checker: Function gc(domain, state) -> bool. Identifies
                goal states where V(g)=0. Defaults to domain.is_goal().
            heuristic: Function h(domain, state) -> Value. Admissible
                heuristic for primary cost. Defaults to Value(cost=0).
            terminal_value: Function tv(state) -> Value. Penalty for
                dead-end (non-goal terminal) states. Defaults to 1000.
            lp_infinity: Upper bound for LP variable bounds and constraint
                lower bounds used with HiGHS. Defaults to 1e20.
            lp_tolerance: Sparsity threshold for LP coefficients. Values
                below this are treated as zero. Defaults to 1e-15.
            default_dead_end_cost: Default per-constraint dead-end cost
                when no explicit dead_end_costs vector is provided (only
                used in the constrained variant). Defaults to 1000.0.
            lp_callback_interval: Fire callback every N simplex iterations
                during each LP solve, reporting intermediate V(s) and
                policy. 0 disables LP-level callbacks. Defaults to 0.
            parallel: If True, explore action transitions in parallel.
            callback: Called after each LP iteration. Returns True to stop.
            verbose: Enable verbose logging.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(self, parallel=parallel)
            self._lambdas = [goal_checker, heuristic]
            self._ipc_notify = True

            self._solver = idual_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: goal_checker(d, s),
                heuristic=lambda d, s: heuristic(d, s),
                terminal_value=lambda s: terminal_value(s),
                lp_infinity=lp_infinity,
                lp_tolerance=lp_tolerance,
                default_dead_end_cost=default_dead_end_cost,
                lp_callback_interval=lp_callback_interval,
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
                    + "No best action found in IDual for observation "
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

        def get_nb_explored_states(self) -> int:
            return self._solver.get_nb_explored_states()

        def get_nb_lp_iterations(self) -> int:
            return self._solver.get_nb_lp_iterations()

        def get_solving_time(self) -> int:
            return self._solver.get_solving_time()

        def get_explored_states(
            self,
        ) -> set[D_SSP.T_agent[D_SSP.T_observation]]:
            return self._solver.get_explored_states()

        def get_callback_event(self) -> str:
            return self._solver.get_callback_event()

    # =================================================================

    class D_CSSP(
        Domain,
        SingleAgent,
        Sequential,
        EnumerableTransitions,
        Actions,
        Goals,
        Markovian,
        FullyObservable,
        PositiveCosts,
        Constrained,
    ):
        pass

    class CIDual(ParallelSolver, Solver, UncertainPolicies, Utilities, FromAnyState):
        """Heuristic search in dual space for constrained SSPs (i-dual).

        Like IDual but for constrained SSPs: the domain provides
        secondary cost constraints via the Constrained mixin with
        BoundConstraint objects. The optimal policy may be stochastic.

        Based on: Trevizan et al., ICAPS 2016 / IJCAI 2017.
        """

        T_domain = D_CSSP

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            goal_checker: Callable[
                [Domain, D_CSSP.T_state],
                D_CSSP.T_agent[D_CSSP.T_predicate],
            ] = lambda d, s: d.is_goal(s),
            heuristic: Callable[
                [Domain, D_CSSP.T_state],
                D_CSSP.T_agent[Value[D_CSSP.T_value]],
            ] = lambda d, s: Value(cost=0),
            terminal_value: Callable[
                [D_CSSP.T_state], D_CSSP.T_agent[Value[D_CSSP.T_value]]
            ] = lambda s: Value(cost=1000.0),
            secondary_heuristics: Optional[
                list[
                    Callable[
                        [Domain, D_CSSP.T_state],
                        D_CSSP.T_agent[Value[D_CSSP.T_value]],
                    ]
                ]
            ] = None,
            dead_end_costs: Optional[list[float]] = None,
            lp_infinity: float = 1e20,
            lp_tolerance: float = 1e-15,
            default_dead_end_cost: float = 1000.0,
            lp_callback_interval: int = 0,
            parallel: bool = False,
            callback: Callable[[CIDual], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a CIDual solver instance.

            # Parameters
            domain_factory: Lambda returning a domain instance.
            goal_checker: Function gc(domain, state) -> bool.
            heuristic: Function h(domain, state) -> Value. Primary cost
                heuristic.
            terminal_value: Function tv(state) -> Value. Dead-end penalty.
            secondary_heuristics: List of h_j(domain, state) -> Value,
                one per constraint. Admissible heuristics for secondary
                costs. Defaults to h=0 (always admissible).
            dead_end_costs: List of d_j per constraint for dead-end
                penalty in secondary cost constraints. Defaults to
                default_dead_end_cost.
            lp_infinity: Upper bound for LP variable bounds and constraint
                lower bounds used with HiGHS. Defaults to 1e20.
            lp_tolerance: Sparsity threshold for LP coefficients. Values
                below this are treated as zero. Defaults to 1e-15.
            default_dead_end_cost: Default per-constraint dead-end cost
                when no explicit dead_end_costs is provided.
                Defaults to 1000.0.
            lp_callback_interval: Fire callback every N simplex iterations
                during each LP solve, reporting intermediate V(s) and
                policy. 0 disables LP-level callbacks. Defaults to 0.
            parallel: If True, explore action transitions in parallel.
            callback: Called after each LP iteration. Returns True to stop.
            verbose: Enable verbose logging.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(self, parallel=parallel)

            domain = self.get_domain()
            constraints = domain.get_constraints()
            for c in constraints:
                if not isinstance(c, BoundConstraint):
                    raise ValueError(
                        "CIDual requires all constraints to be "
                        "BoundConstraint instances, got "
                        f"{type(c).__name__}."
                    )

            n_constraints = len(constraints)

            sec_heur_func = None
            if secondary_heuristics is not None:
                if len(secondary_heuristics) != n_constraints:
                    raise ValueError(
                        f"secondary_heuristics must have {n_constraints} "
                        f"entries (one per constraint), got "
                        f"{len(secondary_heuristics)}."
                    )

                def sec_heur_func(d, s, j):
                    return secondary_heuristics[j](d, s)

            de_costs = (
                dead_end_costs
                if dead_end_costs is not None
                else [1000.0] * n_constraints
            )

            self._lambdas = [goal_checker, heuristic]
            if secondary_heuristics:
                self._lambdas.extend(secondary_heuristics)
            self._ipc_notify = True

            self._solver = cidual_solver(
                solver=self,
                domain=domain,
                goal_checker=lambda d, s: goal_checker(d, s),
                heuristic=lambda d, s: heuristic(d, s),
                terminal_value=lambda s: terminal_value(s),
                secondary_heuristic=sec_heur_func,
                dead_end_costs=de_costs,
                lp_infinity=lp_infinity,
                lp_tolerance=lp_tolerance,
                default_dead_end_cost=default_dead_end_cost,
                lp_callback_interval=lp_callback_interval,
                parallel=parallel,
                callback=callback,
                verbose=verbose,
            )

        def _get_next_action_distribution(
            self,
            observation: D_CSSP.T_agent[D_CSSP.T_observation],
            domain: Optional[Domain] = None,
        ):
            dist = self._solver.get_action_distribution(observation)
            if not dist:
                raise RuntimeError(
                    "No action distribution found in CIDual for "
                    f"observation {observation}"
                )
            return DiscreteDistribution(dist)

        def _solve_from(self, memory: D_CSSP.T_memory[D_CSSP.T_state]) -> None:
            self._solver.solve(memory)

        def _is_solution_defined_for(
            self, observation: D_CSSP.T_agent[D_CSSP.T_observation]
        ) -> bool:
            return self._solver.is_solution_defined_for(observation)

        def _get_utility(
            self, observation: D_CSSP.T_agent[D_CSSP.T_observation]
        ) -> D_CSSP.T_value:
            return self._solver.get_utility(observation)

        def get_nb_explored_states(self) -> int:
            return self._solver.get_nb_explored_states()

        def get_nb_lp_iterations(self) -> int:
            return self._solver.get_nb_lp_iterations()

        def get_solving_time(self) -> int:
            return self._solver.get_solving_time()

        def get_explored_states(
            self,
        ) -> set[D_CSSP.T_agent[D_CSSP.T_observation]]:
            return self._solver.get_explored_states()

        def get_callback_event(self) -> str:
            return self._solver.get_callback_event()

except ImportError:
    print(
        "Scikit-decide C++ hub library not found. "
        "IDual/CIDual solvers require the C++ hub with HiGHS support."
    )
    raise
