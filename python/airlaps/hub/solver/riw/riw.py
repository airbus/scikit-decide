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
from typing import Callable, Any

from airlaps import Domain, Solver
from airlaps import hub
from airlaps.domains import ParallelDomain
from airlaps.builders.domain import SingleAgent, Sequential, Environment, Actions, \
    DeterministicInitialized, Markovian, FullyObservable, Rewards
from airlaps.builders.solver import DeterministicPolicies, Utilities

record_sys_path = sys.path
airlaps_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if airlaps_cpp_extension_lib_path not in sys.path:
    sys.path.append(airlaps_cpp_extension_lib_path)

try:

    from __airlaps_hub_cpp import _RIWSolver_ as riw_solver

    class D(Domain, SingleAgent, Sequential, Environment, Actions, DeterministicInitialized, Markovian,
            FullyObservable, Rewards):  # TODO: check why DeterministicInitialized & PositiveCosts/Rewards?
        pass


    class RIW(Solver, DeterministicPolicies, Utilities):
        T_domain = D

        def __init__(self,
                     state_features: Callable[[Domain, D.T_state], Any],
                     use_state_feature_hash: bool = False,
                     use_simulation_domain = False,
                     time_budget: int = 3600000,
                     rollout_budget: int = 100000,
                     max_depth: int = 1000,
                     exploration: float = 0.25,
                     discount: float = 1.0,
                     online_node_garbage: bool = False,
                     parallel: bool = True,
                     debug_logs: bool = False) -> None:
            self._solver = None
            self._domain = None
            self._state_features = state_features
            self._use_state_feature_hash = use_state_feature_hash
            self._use_simulation_domain = use_simulation_domain
            self._time_budget = time_budget
            self._rollout_budget = rollout_budget
            self._max_depth = max_depth
            self._exploration = exploration
            self._discount = discount
            self._online_node_garbage = online_node_garbage
            self._parallel = parallel
            self._debug_logs = debug_logs

        def _init_solve(self, domain_factory: Callable[[], D]) -> None:
            self._domain = ParallelDomain(domain_factory) if self._parallel else domain_factory()
            self._solver = riw_solver(domain=self._domain,
                                      state_features=self._state_features,
                                      use_state_feature_hash=self._use_state_feature_hash,
                                      use_simulation_domain=self._use_simulation_domain,
                                      time_budget=self._time_budget,
                                      rollout_budget=self._rollout_budget,
                                      max_depth=self._max_depth,
                                      exploration=self._exploration,
                                      discount=self._discount,
                                      online_node_garbage=self._online_node_garbage,
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
        
        def get_nb_of_pruned_states(self) -> int:
            return self._solver.get_nb_of_pruned_states()
        
        def get_nb_rollouts(self) -> int:
            return self._solver.get_nb_rollouts()
        
        def get_policy(self) -> Dict[D.T_agent[D.T_observation], Tuple[D.T_agent[D.T_concurrency[D.T_event]], float]]:
            return self._solver.get_policy()
        
        def get_action_prefix(self) -> List[D.T_agent[D.T_observation]]:
            return self._solver.get_action_prefix()
    
except ImportError:
    sys.path = record_sys_path
    print('AIRLAPS C++ hub library not found. Please check it is installed in "airlaps/hub".')
    raise
