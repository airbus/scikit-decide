# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import multiprocessing
import os
import sys
from math import sqrt
from typing import Callable, Any

from airlaps import Domain, Solver
from airlaps import hub
from airlaps.builders.domain import SingleAgent, Sequential, Environment, Actions, \
    DeterministicInitialized, Markovian, FullyObservable, Rewards
from airlaps.builders.solver import DeterministicPolicies, Utilities

record_sys_path = sys.path
airlaps_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if airlaps_cpp_extension_lib_path not in sys.path:
    sys.path.append(airlaps_cpp_extension_lib_path)

try:

    from __airlaps_hub_cpp import _MCTSSolver_ as mcts_solver

    class D(Domain, SingleAgent, Sequential, Environment, Actions, DeterministicInitialized, Markovian,
            FullyObservable, Rewards):  # TODO: check why DeterministicInitialized & PositiveCosts/Rewards?
        pass


    class MCTS(Solver, DeterministicPolicies, Utilities):
        T_domain = D

        def __init__(self,
                     time_budget: int = 3600000,
                     rollout_budget: int = 100000,
                     max_depth: int = 1000,
                     discount: float = 1.0,
                     uct_mode: bool = True,
                     ucb_constant: float = 1.0 / sqrt(2.0),
                     parallel: bool = True,
                     debug_logs: bool = False) -> None:
            self._solver = None
            self._domain = None
            self._time_budget = time_budget
            self._rollout_budget = rollout_budget
            self._max_depth = max_depth
            self._discount = discount
            self._uct_mode = uct_mode
            self._ucb_constant = ucb_constant
            self._parallel = parallel
            self._debug_logs = debug_logs

        def _init_solve(self, domain_factory: Callable[[], D]) -> None:
            self._domain = domain_factory()
            self._solver = mcts_solver(domain=self._domain,
                                       time_budget=self._time_budget,
                                       rollout_budget=self._rollout_budget,
                                       max_depth=self._max_depth,
                                       discount=self._discount,
                                       uct_mode=self._uct_mode,
                                       ucb_constant=self._ucb_constant,
                                       parallel=self._parallel,
                                       debug_logs=self._debug_logs)
            self._solver.clear()

        def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
            self._init_solve(domain_factory)

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
            self._solver.solve(memory)
        
        def _is_solution_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
            return self._solver.is_solution_defined_for(observation)
        
        def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
            if not self._is_solution_defined_for(observation):
                self._solve_from(observation)
            action = self._solver.get_next_action(observation)
            if action is None:
                print('\x1b[3;33;40m' + 'No best action found in observation ' +
                      str(observation) + ', applying random action' + '\x1b[0m')
                return self._domain.get_action_space().sample()
            else:
                return action
        
        def _reset(self) -> None:
            self._solver.clear()
        
        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            return self._solver.get_utility(observation)
        
        def get_nb_of_explored_states(self) -> int:
            return self._solver.get_nb_of_explored_states()
        
        def get_nb_rollouts(self) -> int:
            return self._solver.get_nb_rollouts()
    

    class UCT(MCTS):
        def __init__(self,
                     time_budget: int = 3600000,
                     rollout_budget: int = 100000,
                     max_depth: int = 1000,
                     discount: float = 1.0,
                     ucb_constant: float = 1.0 / sqrt(2.0),
                     parallel: bool = True,
                     debug_logs: bool = False) -> None:
            super().__init__(time_budget=time_budget,
                             rollout_budget=rollout_budget,
                             max_depth=max_depth,
                             discount=discount,
                             uct_mode=True,
                             ucb_constant=ucb_constant,
                             parallel=parallel,
                             debug_logs=debug_logs)
    
except ImportError:
    sys.path = record_sys_path
    print('AIRLAPS C++ hub library not found. Please check it is installed in "airlaps/hub".')
    raise
