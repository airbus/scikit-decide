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
    MultiAgent,
    PositiveCosts,
    Sequential,
    Simulation,
)
from skdecide.builders.solver import DeterministicPolicies, ParallelSolver, Utilities
from skdecide.core import Value

record_sys_path = sys.path
skdecide_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if skdecide_cpp_extension_lib_path not in sys.path:
    sys.path.append(skdecide_cpp_extension_lib_path)

try:

    from __skdecide_hub_cpp import _MARTDPSolver_ as martdp_solver

    # TODO: remove Markovian req?
    class D(
        Domain,
        MultiAgent,
        Sequential,
        Simulation,
        Actions,
        Goals,
        Markovian,
        FullyObservable,
        PositiveCosts,
    ):
        pass

    class MARTDP(ParallelSolver, Solver, DeterministicPolicies, Utilities):
        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain] = None,
            heuristic: Optional[
                Callable[
                    [Domain, D.T_state],
                    Tuple[
                        D.T_agent[Value[D.T_value]],
                        D.T_agent[D.T_concurrency[D.T_event]],
                    ],
                ]
            ] = None,
            time_budget: int = 3600000,
            rollout_budget: int = 100000,
            max_depth: int = 1000,
            max_feasibility_trials: int = 0,  # will then choose nb_agents if 0
            graph_expansion_rate: float = 0.1,
            epsilon_moving_average_window: int = 100,
            epsilon: float = 0.0,  # not a stopping criterion by default
            discount: float = 1.0,
            action_choice_noise: float = 0.1,
            dead_end_cost: float = 10000,
            online_node_garbage: bool = False,
            continuous_planning: bool = True,
            parallel: bool = False,
            shared_memory_proxy=None,
            debug_logs: bool = False,
            watchdog: Callable[[int, int, float, float], bool] = None,
        ) -> None:
            ParallelSolver.__init__(
                self,
                domain_factory=domain_factory,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._solver = None
            if heuristic is None:
                self._heuristic = lambda d, s: (
                    {a: Value(cost=0) for a in s},
                    {a: None for a in s},
                )
            else:
                self._heuristic = heuristic
            self._lambdas = [self._heuristic]
            self._time_budget = time_budget
            self._rollout_budget = rollout_budget
            self._max_depth = max_depth
            self._max_feasibility_trials = max_feasibility_trials
            self._graph_expansion_rate = graph_expansion_rate
            self._epsilon_moving_average_window = epsilon_moving_average_window
            self._epsilon = epsilon
            self._discount = discount
            self._action_choice_noise = action_choice_noise
            self._dead_end_cost = dead_end_cost
            self._online_node_garbage = online_node_garbage
            self._continuous_planning = continuous_planning
            self._debug_logs = debug_logs
            if watchdog is None:
                self._watchdog = (
                    lambda elapsed_time, number_rollouts, best_value, epsilon_moving_average: True
                )
            else:
                self._watchdog = watchdog
            self._ipc_notify = True

        def close(self):
            """Joins the parallel domains' processes.
            Not calling this method (or not using the 'with' context statement)
            results in the solver forever waiting for the domain processes to exit.
            """
            if self._parallel:
                self._solver.close()
            ParallelSolver.close(self)

        def _init_solve(self, domain_factory: Callable[[], Domain]) -> None:
            self._domain_factory = domain_factory
            self._solver = martdp_solver(
                domain=self.get_domain(),
                goal_checker=lambda d, s: d.is_goal(s),
                heuristic=lambda d, s: self._heuristic(d, s)
                if not self._parallel
                else d.call(None, 0, s),
                time_budget=self._time_budget,
                rollout_budget=self._rollout_budget,
                max_depth=self._max_depth,
                max_feasibility_trials=self._max_feasibility_trials,
                graph_expansion_rate=self._graph_expansion_rate,
                epsilon_moving_average_window=self._epsilon_moving_average_window,
                epsilon=self._epsilon,
                discount=self._discount,
                action_choice_noise=self._action_choice_noise,
                dead_end_cost=self._dead_end_cost,
                online_node_garbage=self._online_node_garbage,
                parallel=self._parallel,
                debug_logs=self._debug_logs,
                watchdog=self._watchdog,
            )
            self._solver.clear()

        def _solve(self, domain_factory: Callable[[], D]) -> None:
            self._init_solve(domain_factory)

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
            self._solver.solve(memory)

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self, observation: D.T_agent[D.T_observation]
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
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
            return self._solver.get_utility(observation)

        def get_nb_of_explored_states(self) -> int:
            return self._solver.get_nb_of_explored_states()

        def get_nb_rollouts(self) -> int:
            return self._solver.get_nb_rollouts()

        def get_policy(
            self,
        ) -> Dict[
            D.T_agent[D.T_observation],
            Tuple[D.T_agent[D.T_concurrency[D.T_event]], float],
        ]:
            return self._solver.get_policy()

except ImportError:
    sys.path = record_sys_path
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
