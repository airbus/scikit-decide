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
    DeterministicTransitions,
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
    from skdecide.hub.__skdecide_hub_cpp import _IDAstarSolver_ as idastar_solver
    from skdecide.hub.__skdecide_hub_cpp import _LDFSSolver_ as ldfs_solver

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

    class LDFS(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """Labeled Depth-First Search (LDFS) solver for MDPs.

        From: Bonet & Geffner, "Learning Depth-First Search: A Unified
        Approach to Heuristic Search in Deterministic and Non-Deterministic
        Settings, and Its Application to MDPs", ICAPS 2008.

        LDFS is a systematic alternative to LRTDP for solving SSPs and
        discounted MDPs. Instead of random trials, it performs depth-first
        search from the initial state, following the greedy policy and
        expanding states on-the-fly. After the DFS returns from each state,
        the check_solved procedure labels converged states as solved.

        Uses cost minimization: V(s) = min_a [C(s,a) + gamma * sum_s'
        P(s'|s,a) * V(s')]. Terminal states are absorbing with value set
        by the terminal_value functor (defaults to Value(cost=0)).

        Unlike VI/PI which enumerate the full reachable state space, LDFS
        only explores states reachable under the evolving greedy policy,
        which can be much smaller on large domains.
        """

        T_domain = D

        hyperparameters = [
            FloatHyperparameter(name="discount"),
            FloatHyperparameter(name="epsilon"),
        ]

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            heuristic: Callable[
                [Domain, D.T_state], D.T_agent[Value[D.T_value]]
            ] = lambda d, s: Value(cost=0),
            terminal_value: Callable[[D.T_state], Value[D.T_value]] = lambda s: Value(
                cost=0
            ),
            discount: float = 1.0,
            epsilon: float = 0.001,
            max_depth: int = 0,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[LDFS], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a LDFS solver instance

            # Parameters
            domain_factory: The lambda function to create a domain instance.
            heuristic: Function h(domain, state) -> Value used to initialize
                V(s) = h(s).cost for newly discovered states. An admissible
                heuristic (h(s) <= V*(s)) accelerates convergence.
                Defaults to Value(cost=0).
            terminal_value: Function f(state) -> Value assigning a fixed value to
                terminal (absorbing) states. Use Value(cost=0) for goal-like
                terminals and Value(cost=large_penalty) for dead-end-like
                terminals. Defaults to Value(cost=0).
            discount: Value function's discount factor. Defaults to 0.999.
            epsilon: Maximum Bellman error allowed to label a state as solved.
                Defaults to 0.001.
            max_depth: Maximum DFS depth per driver iteration. 0 means unlimited.
                When reached, the DFS backtracks as if the state were unsolved.
                The driver loop retries, so correctness is preserved — only
                per-iteration work is bounded. Defaults to 0 (unlimited).
            parallel: Parallelize action-transition generation. Defaults to False.
            shared_memory_proxy: The optional shared memory proxy. Defaults to None.
            callback: Lambda function called at the end of each LDFS pass,
                taking the solver as argument, returning true to stop.
                Defaults to never stop.
            verbose: Whether verbose messages should be logged. Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._lambdas = [heuristic]
            self._ipc_notify = True

            self._solver = ldfs_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: d.is_goal(s),
                heuristic=(
                    (lambda d, s: heuristic(d, s))
                    if not parallel
                    else (lambda d, s: d.call(None, 0, s))
                ),
                terminal_value=terminal_value,
                discount=discount,
                epsilon=epsilon,
                max_depth=max_depth,
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

        def get_nb_of_explored_states(self) -> int:
            """Get the number of states explored so far"""
            return self._solver.get_nb_explored_states()

        def get_nb_tip_states(self) -> int:
            """Get the number of tip states expanded during search"""
            return self._solver.get_nb_tip_states()

        def get_explored_states(self) -> set[D.T_agent[D.T_observation]]:
            """Get all states explored so far"""
            return self._solver.get_explored_states()

        def get_solved_states(self) -> set[D.T_agent[D.T_observation]]:
            """Get states labeled as solved (converged)"""
            return self._solver.get_solved_states()

        def get_strongly_connected_components(
            self,
        ) -> list[set[D.T_agent[D.T_observation]]]:
            """Get the strongly connected components discovered so far by Tarjan's
            algorithm during the DFS. Each SCC is a set of states that form a
            cycle under the greedy policy. Useful for monitoring convergence
            within cyclic components."""
            return self._solver.get_strongly_connected_components()

        def get_solving_time(self) -> int:
            """Get the solving time in milliseconds"""
            return self._solver.get_solving_time()

        def get_policy(
            self,
        ) -> dict[
            D.T_agent[D.T_observation],
            tuple[D.T_agent[D.T_concurrency[D.T_event]], D.T_value],
        ]:
            """Get the (partial) solution policy"""
            return self._solver.get_policy()

        def get_last_trajectory(self) -> list[D.T_agent[D.T_observation]]:
            """Get the ordered list of states visited during the last LDFS iteration.

            Returns the trajectory (path) explored during the most recent depth-first
            descent from the root state. The trajectory begins with the root state and
            ends at the deepest state reached before backtracking.

            This is useful for:
            - Debugging algorithm behavior (which path was explored?)
            - Custom heuristic updates based on trajectory
            - Visualizing/logging the search process
            - Analyzing convergence patterns in the callback

            Returns an empty list if solve() has not been called yet.
            """
            return self._solver.get_last_trajectory()

    class D_IDAstar(
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

    class IDAstar(LDFS):
        """IDA* solver for deterministic planning problems.

        From: Bonet & Geffner, "Learning Depth-First Search: A Unified
        Approach to Heuristic Search in Deterministic and Non-Deterministic
        Settings, and Its Application to MDPs", ICAPS 2008, Proposition 6.

        IDA* is a specialization of LDFS for deterministic domains. When all
        transitions are deterministic and the heuristic is admissible (monotone),
        LDFS reduces exactly to IDA* with transposition tables. The algorithm,
        SCCs, and Tarjan bookkeeping simplify away since deterministic policies
        cannot have cycles, leaving pure iterative-deepening depth-first search.

        Uses cost minimization with goals, same as LDFS.
        """

        T_domain = D_IDAstar

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            heuristic: Callable[
                [Domain, D_IDAstar.T_state],
                D_IDAstar.T_agent[Value[D_IDAstar.T_value]],
            ] = lambda d, s: Value(cost=0),
            max_depth: int = 0,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[LDFS], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct an IDA* solver instance.

            # Parameters
            domain_factory: The lambda function to create a domain instance.
            heuristic: Function h(domain, state) -> Value used to initialize
                V(s) = h(s).cost for newly discovered states. An admissible
                heuristic (h(s) <= V*(s)) is required for optimality.
                Defaults to Value(cost=0).
            max_depth: Maximum DFS depth per iteration. 0 means unlimited.
                Defaults to 0.
            parallel: Parallelize action-transition generation. Defaults to False.
            shared_memory_proxy: The optional shared memory proxy. Defaults to None.
            callback: Lambda function called at the end of each IDA* pass,
                taking the solver as argument, returning true to stop.
                Defaults to never stop.
            verbose: Whether verbose messages should be logged. Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._lambdas = [heuristic]
            self._ipc_notify = True

            self._solver = idastar_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: d.is_goal(s),
                heuristic=(
                    (lambda d, s: heuristic(d, s))
                    if not parallel
                    else (lambda d, s: d.call(None, 0, s))
                ),
                max_depth=max_depth,
                parallel=parallel,
                callback=callback,
                verbose=verbose,
            )

        def get_plan(
            self,
            from_memory: Optional[D_IDAstar.T_memory[D_IDAstar.T_state]] = None,
        ) -> list[D_IDAstar.T_agent[D_IDAstar.T_concurrency[D_IDAstar.T_event]]]:
            """Get the solution plan (sequence of actions to goal).

            Since the domain is deterministic, the greedy policy defines a
            unique action sequence. Calls the C++ IDAstarSolver::get_plan()
            which follows the greedy policy through the search graph.

            # Parameters
            from_memory: State from which to extract the plan. If None, uses
                the domain's initial state.

            Returns an empty list if no solution is defined.
            """
            if from_memory is None:
                domain = self._domain_factory()
                from_memory = domain.reset()
            return self._solver.get_plan(from_memory)

except ImportError:
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
