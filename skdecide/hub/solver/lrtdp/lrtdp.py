# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Optional, Tuple

from skdecide import Domain, Solver, hub
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

record_sys_path = sys.path
skdecide_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if skdecide_cpp_extension_lib_path not in sys.path:
    sys.path.append(skdecide_cpp_extension_lib_path)

try:

    from __skdecide_hub_cpp import _LRTDPSolver_ as lrtdp_solver

    # TODO: remove Markovian req?
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

    class LRTDP(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """This is the skdecide implementation of "Labeled RTDP: Improving the
        Convergence of Real-Time Dynamic Programming" by Blai Bonet and Hector
        Geffner (ICAPS 2013)
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain] = None,
            heuristic: Optional[
                Callable[[Domain, D.T_state], D.T_agent[Value[D.T_value]]]
            ] = None,
            use_labels: bool = True,
            time_budget: int = 3600000,
            rollout_budget: int = 100000,
            max_depth: int = 1000,
            epsilon_moving_average_window: int = 100,
            epsilon: float = 0.001,
            discount: float = 1.0,
            online_node_garbage: bool = False,
            continuous_planning: bool = True,
            parallel: bool = False,
            shared_memory_proxy=None,
            debug_logs: bool = False,
            callback: Callable[[Domain, LRTDP], bool] = None,
        ) -> None:
            """Construct a LRTDP solver instance

            Args:
                domain_factory (Callable[[], Domain], optional): The domain instance. Defaults to None.
                heuristic (Optional[ Callable[[Domain, D.T_state], D.T_agent[Value[D.T_value]]] ], optional):
                    Lambda function taking as arguments the domain and a state, and returning the heuristic
                    estimate from the state to the goal. Defaults to None.
                use_labels (bool, optional): Boolean indicating whether labels must be used (True) or not
                    (False, in which case the algorithm is equivalent to the standard RTDP). Defaults to True.
                time_budget (int, optional): Maximum solving time in milliseconds. Defaults to 3600000.
                rollout_budget (int, optional): Maximum number of rollouts (deactivated when
                    use_labels is True). Defaults to 100000.
                max_depth (int, optional): Maximum depth of each LRTDP trial (rollout). Defaults to 1000.
                epsilon_moving_average_window (int, optional): Number of latest computed residual values
                    to memorize in order to compute the average Bellman error (residual) at the root state
                    of the search (deactivated when use_labels is True). Defaults to 100.
                epsilon (float, optional): Maximum Bellman error (residual) allowed to decide that a state
                    is solved, or to decide when no labels are used that the value function of the root state
                    of the search has converged (in the latter case: the root state's Bellman error is averaged
                    over the epsilon_moving_average_window, deactivated when use_labels is True). Defaults to 0.001.
                discount (float, optional): Value function's discount factor. Defaults to 1.0.
                online_node_garbage (bool, optional): Boolean indicating whether the search graph which is
                    no more reachable from the root solving state should be deleted (True) or not (False). Defaults to False.
                continuous_planning (bool, optional): Boolean whether the solver should optimize again the policy
                    from the current solving state (True) or not (False) even if the policy is already defined
                    in this state. Defaults to True.
                parallel (bool, optional): Parallelize LRTDP trials on different processes using duplicated domains (True)
                    or not (False). Defaults to False.
                shared_memory_proxy (_type_, optional): The optional shared memory proxy. Defaults to None.
                debug_logs (bool, optional): Boolean indicating whether debugging messages should be logged (True)
                    or not (False). Defaults to False.
                callback (Callable[[Domain, LRTDP], optional): Function called at the end of each LRTDP trial (rollout),
                    taking as arguments the solver and the domain, and returning True if the solver must be stopped.
                    Defaults to None.
            """
            ParallelSolver.__init__(
                self,
                domain_factory=domain_factory,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._solver = None
            if heuristic is None:
                self._heuristic = lambda d, s: Value(cost=0)
            else:
                self._heuristic = heuristic
            self._lambdas = [self._heuristic, self._callback]
            self._use_labels = use_labels
            self._time_budget = time_budget
            self._rollout_budget = rollout_budget
            self._max_depth = max_depth
            self._epsilon_moving_average_window = epsilon_moving_average_window
            self._epsilon = epsilon
            self._discount = discount
            self._online_node_garbage = online_node_garbage
            self._continuous_planning = continuous_planning
            self._debug_logs = debug_logs
            if callback is None:
                self._callback = lambda d, s: False
            else:
                self._callback = callback
            self._ipc_notify = True

        def close(self):
            """Joins the parallel domains' processes.

            !!! warning
                Not calling this method (or not using the 'with' context statement)
                results in the solver forever waiting for the domain processes to exit.

            """
            if self._parallel:
                self._solver.close()
            ParallelSolver.close(self)

        def _init_solve(self, domain_factory: Callable[[], Domain]) -> None:
            self._domain_factory = domain_factory
            self._solver = lrtdp_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s, i=None: (
                    d.is_goal(s) if not self._parallel else d.is_goal(s, i)
                ),
                heuristic=lambda d, s, i=None: (
                    self._heuristic(d, s) if not self._parallel else d.call(i, 0, s)
                ),
                use_labels=self._use_labels,
                time_budget=self._time_budget,
                rollout_budget=self._rollout_budget,
                max_depth=self._max_depth,
                epsilon_moving_average_window=self._epsilon_moving_average_window,
                epsilon=self._epsilon,
                discount=self._discount,
                online_node_garbage=self._online_node_garbage,
                parallel=self._parallel,
                debug_logs=self._debug_logs,
                callback=lambda d, s, i=None: (
                    self._callback(d, s) if not self._parallel else d.call(i, 1, s)
                ),
            )
            self._solver.clear()

        def _reset(self) -> None:
            """Clears the search graph."""
            self._solver.clear()

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
            """Run the LRTDP algorithm from a given root solving state

            Args:
                memory (D.T_memory[D.T_state]): State from which to run the LRTDP algorithm
                    (root of the search graph)
            """
            self._solver.solve(memory)

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            """Indicates whether the solution policy is defined for a given state

            Args:
                observation (D.T_agent[D.T_observation]): State for which an entry is searched
                    in the policy graph

            Returns:
                bool: True if the state has been explored and an action is defined in this state,
                    False otherwise
            """
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self, observation: D.T_agent[D.T_observation]
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            """Get the best computed action in terms of best Q-value in a given state

            !!! warning
                Returns a random action if no action is defined in the given state,
                which is why it is advised to call :py:meth:`LRTDP.is_solution_defined_for` before

            Args:
                observation (D.T_agent[D.T_observation]): State for which the best action is requested

            Returns:
                D.T_agent[D.T_concurrency[D.T_event]]: Best computed action
            """
            if self._continuous_planning or not self._is_solution_defined_for(
                observation
            ):
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
            """Get the best Q-value in a given state

            !!! warning
                Returns None if no action is defined in the given state, which is why
                it is advised to call :py:meth:`LRTDP.is_solution_defined_for` before

            Args:
                observation (D.T_agent[D.T_observation]): State from which the best Q-value is requested

            Returns:
                D.T_value: Maximum Q-value of the given state over the applicable actions in this state
            """
            return self._solver.get_utility(observation)

        def get_nb_of_explored_states(self) -> int:
            """Get the number of states present in the search graph (which can be
                lower than the number of actually explored states if node garbage has been
                set to true in the LRTDP instance's constructor)

            Returns:
                int: Number of states present in the search graph
            """
            return self._solver.get_nb_of_explored_states()

        def get_nb_rollouts(self) -> int:
            """Get the number of rollouts since the beginning of the search from
                the root solving state

            Returns:
                int: Number of rollouts (LRTDP trials)
            """
            return self._solver.get_nb_rollouts()

        def get_residual_moving_average(self) -> float:
            """Get the average Bellman error (residual) at the root state of the search,
                or an infinite value if the number of computed residuals is lower than
                the epsilon moving average window set in the LRTDP instance's constructor

            Returns:
                float: Bellman error at the root state of the search averaged over
                    the epsilon moving average window
            """
            return self._solver.get_residual_moving_average()

        def get_solving_time(self) -> int:
            """Get the solving time in milliseconds since the beginning of the
                search from the root solving state

            Returns:
                int: Solving time in milliseconds
            """
            return self._solver.get_solving_time()

        def get_policy(
            self,
        ) -> Dict[
            D.T_agent[D.T_observation],
            Tuple[D.T_agent[D.T_concurrency[D.T_event]], float],
        ]:
            """Get the (partial) solution policy defined for the states for which
                the Q-value has been updated at least once (which is optimal if the
                algorithm has converged and labels are used); warning: only defined over
                the states reachable from the last root solving state when node garbage is
                set in the LRTDP instance's constructor

            Returns:
                Dict[ D.T_agent[D.T_observation], Tuple[D.T_agent[D.T_concurrency[D.T_event]], float], ]:
                    Mapping from states to pairs of action and best Q-value
            """
            return self._solver.get_policy()

except ImportError:
    sys.path = record_sys_path
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
