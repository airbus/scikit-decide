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
from airlaps.builders.domain import SingleAgent, Sequential, DeterministicTransitions, Actions, \
    DeterministicInitialized, Markovian, FullyObservable, Rewards
from airlaps.builders.solver import DeterministicPolicies, Utilities
from airlaps.hub.space.gym import ListSpace

record_sys_path = sys.path
airlaps_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if airlaps_cpp_extension_lib_path not in sys.path:
    sys.path.append(airlaps_cpp_extension_lib_path)

try:

    from __airlaps_hub_cpp import _IWSolver_ as iw_solver

    class IWActionProxy:
        def __init__(self, a):
            self.action = a
            self.ns_result = None
        
        def __str__(self):
            return self.action.__str__()

    def IWDomain_pickable_get_next_state(domain, state, action):
        return domain.get_next_state(state, action)
    

    class IWProxyDomain:
        def __init__(self, domain):
            self._domain = domain
        
        def get_initial_state(self):
            return self._domain.get_initial_state()

        def get_applicable_actions(self, state):
            return ListSpace([IWActionProxy(a) for a in self._domain.get_applicable_actions(state).get_elements()])
        
        def get_transition_value(self, memory, action, next_state):
            return self._domain.get_transition_value(memory, action, next_state)
        
        def is_goal(self, state):
            return self._domain.is_goal(state)
        
        def is_terminal(self, state):
            return self._domain.is_terminal(state)
    
    class IWParallelDomain(IWProxyDomain):
        def __init__(self, domain):
            super().__init__(domain)
            self.iw_pool = multiprocessing.Pool()

        def compute_next_state(self, state, action):  # self is a domain
            action.ns_result = self.iw_pool.apply_async(
                                    IWDomain_pickable_get_next_state,
                                    (self._domain, state, action.action))
        
        def get_next_state(self, state, action):  # self is a domain
            return action.ns_result.get()
    
    class IWSequentialDomain(IWProxyDomain):
        def compute_next_state(self, state, action):  # self is a domain
            action.ns_result = self._domain.get_next_state(state, action.action)

        def get_next_state(self, state, action):  # self is a domain
            return action.ns_result


    class D(Domain, SingleAgent, Sequential, DeterministicTransitions, Actions, DeterministicInitialized, Markovian,
            FullyObservable, Rewards):  # TODO: check why DeterministicInitialized & PositiveCosts/Rewards?
        pass


    class IW(Solver, DeterministicPolicies, Utilities):
        T_domain = D

        def __init__(self,
                     state_features: Callable[[D.T_state, Domain], Any],
                     use_state_feature_hash: bool = False,
                     node_ordering: Callable[[float, int, int, float, int, int], bool] = None,
                     parallel: bool = True,
                     debug_logs: bool = False) -> None:
            self._solver = None
            self._domain = None
            self._state_features = state_features
            self._use_state_feature_hash = use_state_feature_hash
            self._node_ordering = node_ordering
            self._parallel = parallel
            self._debug_logs = debug_logs

        def _init_solve(self, domain_factory: Callable[[], D]) -> None:
            if self._parallel:
                self._domain = IWParallelDomain(domain_factory())
            else:
                self._domain = IWSequentialDomain(domain_factory())
            self._solver = iw_solver(domain=self._domain,
                                     state_features=lambda o: self._state_features(o, self._domain._domain),
                                     use_state_feature_hash=self._use_state_feature_hash,
                                     node_ordering=self._node_ordering,
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
            action_proxy = self._solver.get_next_action(observation)
            return action_proxy.action if action_proxy is not None else None
        
        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            return self._solver.get_utility(observation)
        
        def _reset(self) -> None:
            self._solver.clear()
        
        def get_nb_of_explored_states(self) -> int:
            return self._solver.get_nb_of_explored_states()
        
        def get_nb_of_pruned_states(self) -> int:
            return self._solver.get_nb_of_pruned_states()
    
except ImportError:
    sys.path = record_sys_path
    print('AIRLAPS C++ hub library not found. Please check it is installed in "airlaps/hub".')
    raise
