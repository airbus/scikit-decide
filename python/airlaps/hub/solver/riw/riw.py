from __future__ import annotations

import multiprocessing
import os
import sys
from typing import Callable, Any

from airlaps import Domain, Solver
from airlaps import hub
from airlaps.builders.domain import SingleAgent, Sequential, Environment, Actions, \
    DeterministicInitialized, Markovian, FullyObservable, Rewards
from airlaps.builders.solver import DeterministicPolicies, Utilities
from airlaps.hub.space.gym import ListSpace

record_sys_path = sys.path
airlaps_cpp_extension_lib_path = os.path.join(hub.__path__[0], 'lib')
if airlaps_cpp_extension_lib_path not in sys.path:
    sys.path.append(airlaps_cpp_extension_lib_path)

try:

    from __airlaps_hub_cpp import _RIWSolver_ as riw_solver

    # riw_pool = None  # must be separated from the domain since it cannot be pickled

    # class RIWActionProxy:
    #     def __init__(self, a):
    #         self.action = a
    #         self.ns_result = None
        
    #     def __str__(self):
    #         return self.action.__str__()

    # def RIWDomain_parallel_get_applicable_actions(self, state):  # self is a domain
    #     return ListSpace([RIWActionProxy(a) for a in self.get_applicable_actions(state).get_elements()])


    # def RIWDomain_sequential_get_applicable_actions(self, state):  # self is a domain
    #     return ListSpace([RIWActionProxy(a) for a in self.get_applicable_actions(state).get_elements()])


    # def RIWDomain_pickable_get_next_state(domain, state, action):
    #     return domain.get_next_state(state, action)


    # def RIWDomain_parallel_compute_next_state(self, state, action):  # self is a domain
    #     global riw_pool
    #     action.ns_result = riw_pool.apply_async(RIWDomain_pickable_get_next_state, (self, state, action.action))


    # def RIWDomain_sequential_compute_next_state(self, state, action):  # self is a domain
    #     action.ns_result = self.get_next_state(state, action.action)


    # def RIWDomain_parallel_get_next_state(self, state, action):  # self is a domain
    #     return action.ns_result.get()


    # def RIWDomain_sequential_get_next_state(self, state, action):  # self is a domain
    #     return action.ns_result


    class D(Domain, SingleAgent, Sequential, Environment, Actions, DeterministicInitialized, Markovian,
            FullyObservable, Rewards):  # TODO: check why DeterministicInitialized & PositiveCosts/Rewards?
        pass


    class RIW(Solver, DeterministicPolicies, Utilities):
        T_domain = D

        def __init__(self,
                     state_features: Callable[[D.T_state, Domain], Any],
                     use_state_feature_hash: bool = False,
                     use_simulation_domain = False,
                     time_budget: int = 3600000,
                     rollout_budget: int = 100000,
                     parallel: bool = True,
                     debug_logs: bool = False) -> None:
            self._solver = None
            self._domain = None
            self._state_features = state_features
            self._use_state_feature_hash = use_state_feature_hash
            self._use_simulation_domain = use_simulation_domain
            self._time_budget = time_budget
            self._rollout_budget = rollout_budget
            self._parallel = parallel
            self._debug_logs = debug_logs

        def _init_solve(self, domain_factory: Callable[[], D]) -> None:
            self._domain = domain_factory()
            # if self._parallel:
            #     global riw_pool
            #     riw_pool = multiprocessing.Pool()
            #     setattr(self._domain.__class__, 'wrapped_get_applicable_actions',
            #             RIWDomain_parallel_get_applicable_actions)
            #     setattr(self._domain.__class__, 'wrapped_compute_next_state',
            #             RIWDomain_parallel_compute_next_state)
            #     setattr(self._domain.__class__, 'wrapped_get_next_state',
            #             RIWDomain_parallel_get_next_state)
            # else:
            #     setattr(self._domain.__class__, 'wrapped_get_applicable_actions',
            #             RIWDomain_sequential_get_applicable_actions)
            #     setattr(self._domain.__class__, 'wrapped_compute_next_state',
            #             RIWDomain_sequential_compute_next_state)
            #     setattr(self._domain.__class__, 'wrapped_get_next_state',
            #             RIWDomain_sequential_get_next_state)
            self._solver = riw_solver(domain=self._domain,
                                      state_features=lambda o: self._state_features(o, self._domain),
                                      use_state_feature_hash=self._use_state_feature_hash,
                                      use_simulation_domain=self._use_simulation_domain,
                                      time_budget=self._time_budget,
                                      rollout_budget=self._rollout_budget,
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
            return self._solver.get_next_action(observation)
        
        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            return self._solver.get_utility(observation)
    
except ImportError:
    sys.path = record_sys_path
    print('AIRLAPS C++ hub library not found. Please check it is installed in "airlaps/hub/lib".')
    raise
