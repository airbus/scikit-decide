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
    Markovian,
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
    from skdecide.hub.__skdecide_hub_cpp import (
        _VISolver_ as vi_solver,
    )

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

    class VI(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """Value Iteration solver for Markov Decision Processes.

        Enumerates all reachable states via BFS from the initial state, then
        performs synchronous Bellman backups V(s) = max_a [R(s,a) + gamma *
        sum_s' P(s'|s,a) * V(s')] until the maximum residual drops below
        epsilon. Terminal states (where is_terminal() returns true) are
        absorbing; their value is set by the terminal_value functor (defaults
        to Value(reward=0) for goal-like terminals; use a large negative
        reward for dead-end-like terminals).

        This implementation supports an optional heuristic for initializing
        the value function (non-standard extension). Classical VI initializes
        V(s) = 0 for all states. When a heuristic h(s) is provided, states
        are initialized to V(s) = h(s).reward during enumeration. If h is
        admissible (h(s) >= V*(s) for reward-maximizing problems), this
        warm-start preserves correctness and typically reduces the number of
        sweeps to convergence. The default heuristic returns Value(reward=0),
        recovering standard Value Iteration.
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
            ] = lambda d, s: Value(reward=0),
            terminal_value: Callable[[D.T_state], Value[D.T_value]] = lambda s: Value(
                reward=0
            ),
            discount: float = 0.999,
            epsilon: float = 0.001,
            max_sweeps: int = 0,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[VI], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a Value Iteration solver instance

            # Parameters
            domain_factory: The lambda function to create a domain instance.
            heuristic: Optional function h(domain, state) -> Value used to initialize
                V(s) = h(s).reward before Bellman sweeps begin (non-standard warm-start).
                When admissible, preserves correctness and accelerates convergence.
                Defaults to Value(reward=0) for all states (standard Value Iteration).
            terminal_value: Function f(state) -> Value assigning a fixed value to
                terminal (absorbing) states. Use Value(reward=0) for goal-like
                terminals and Value(reward=-penalty) for dead-end-like terminals.
                Defaults to Value(reward=0).
            max_sweeps: Maximum number of Bellman sweeps. Only active when
                discount=1.0 to prevent divergence on non-contracting problems.
                Ignored when discount < 1.0 (convergence is guaranteed).
                0 means unlimited. Defaults to 0.
            discount: Value function's discount factor. Defaults to 1.0.
            epsilon: Maximum Bellman error allowed to decide convergence.
                Defaults to 0.001.
            parallel: Parallelize Bellman updates on different processes. Defaults to False.
            shared_memory_proxy: The optional shared memory proxy. Defaults to None.
            callback: Lambda function called at the end of each Bellman sweep,
                taking the solver as argument, returning true to stop. Defaults to never stop.
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

            self._solver = vi_solver(
                solver=self,
                domain=self.get_domain(),
                heuristic=(
                    (lambda d, s: heuristic(d, s))
                    if not parallel
                    else (lambda d, s: d.call(None, 0, s))
                ),
                terminal_value=terminal_value,
                discount=discount,
                epsilon=epsilon,
                max_sweeps=max_sweeps,
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
            """Run Value Iteration from a given state

            # Parameters
            memory: State from which to enumerate reachable states and run VI
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

        def get_nb_of_explored_states(self) -> int:
            """Get the number of states discovered by BFS"""
            return self._solver.get_nb_explored_states()

        def get_nb_iterations(self) -> int:
            """Get the number of Bellman sweeps performed"""
            return self._solver.get_nb_iterations()

        def get_explored_states(self) -> set[D.T_agent[D.T_observation]]:
            """Get all reachable states discovered by BFS"""
            return self._solver.get_explored_states()

        def get_converged_states(self) -> set[D.T_agent[D.T_observation]]:
            """Get states where Bellman residual < epsilon"""
            return self._solver.get_converged_states()

        def get_states_updated_in_last_sweep(self) -> set[D.T_agent[D.T_observation]]:
            """Get states whose value changed in the last Bellman sweep"""
            return self._solver.get_states_updated_in_last_sweep()

        def get_solving_time(self) -> int:
            """Get the solving time in milliseconds"""
            return self._solver.get_solving_time()

        def get_policy(
            self,
        ) -> dict[
            D.T_agent[D.T_observation],
            tuple[D.T_agent[D.T_concurrency[D.T_event]], D.T_value],
        ]:
            """Get the full solution policy"""
            return self._solver.get_policy()

except ImportError:
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
