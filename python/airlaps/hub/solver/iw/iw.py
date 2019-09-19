import multiprocessing
import os
import sys
from typing import Callable

from airlaps import Domain, Solver
from airlaps import hub
from airlaps.builders.domain import SingleAgent, Sequential, DeterministicTransitions, Actions, \
    DeterministicInitialized, Markovian, FullyObservable, Rewards
from airlaps.builders.solver import DeterministicPolicies, Utilities

record_sys_path = sys.path
airlaps_cpp_extension_lib_path = os.path.join(hub.__path__[0], 'lib')
if airlaps_cpp_extension_lib_path not in sys.path:
    sys.path.append(airlaps_cpp_extension_lib_path)

try:

    from __airlaps_hub_cpp import _IWSolver_ as iw_solver

    iw_pool = None  # must be separated from the domain since it cannot be pickled
    iw_ns_results = None  # must be separated from the domain since it cannot be pickled


    def IWDomain_parallel_get_applicable_actions(self, state):  # self is a domain
        global iw_ns_results
        actions = self.get_applicable_actions(state)
        iw_ns_results = {a: None for a in actions.get_elements()}
        return actions


    def IWDomain_sequential_get_applicable_actions(self, state):  # self is a domain
        global iw_ns_results
        actions = self.get_applicable_actions(state)
        iw_ns_results = {a: None for a in actions.get_elements()}
        return actions


    def IWDomain_pickable_get_next_state(domain, state, action):
        return domain.get_next_state(state, action)


    def IWDomain_parallel_compute_next_state(self, state, action):  # self is a domain
        global iw_pool, iw_ns_results
        iw_ns_results[action] = iw_pool.apply_async(IWDomain_pickable_get_next_state, (self, state, action))


    def IWDomain_sequential_compute_next_state(self, state, action):  # self is a domain
        global iw_ns_results
        iw_ns_results[action] = self.get_next_state(state, action)


    def IWDomain_parallel_get_next_state(self, state, action):  # self is a domain
        global iw_ns_results
        return iw_ns_results[action].get()


    def IWDomain_sequential_get_next_state(self, state, action):  # self is a domain
        global iw_ns_results
        return iw_ns_results[action]


    class D(Domain, SingleAgent, Sequential, DeterministicTransitions, Actions, DeterministicInitialized, Markovian,
            FullyObservable, Rewards):  # TODO: check why DeterministicInitialized & PositiveCosts/Rewards?
        pass


    class IW(Solver, DeterministicPolicies, Utilities):
        T_domain = D

        def __init__(self,
                     nb_of_binary_features: Callable[[Domain], int],
                     state_binarizer: Callable[[D.T_state, Domain, Callable[[int], None], None]],  # TODO: correct hint
                     use_state_feature_hash: bool = False,
                     parallel: bool = True,
                     debug_logs: bool = False) -> None:
            self._solver = None
            self._domain = None
            self._nb_of_binary_features = nb_of_binary_features
            self._state_binarizer = state_binarizer
            self._use_state_feature_hash = use_state_feature_hash
            self._parallel = parallel
            self._debug_logs = debug_logs

        def _init_solve(self, domain_factory: Callable[[], D]) -> None:
            self._domain = domain_factory()
            if self._parallel:
                global iw_pool
                iw_pool = multiprocessing.Pool()
                setattr(self._domain.__class__, 'wrapped_get_applicable_actions',
                        IWDomain_parallel_get_applicable_actions)
                setattr(self._domain.__class__, 'wrapped_compute_next_state',
                        IWDomain_parallel_compute_next_state)
                setattr(self._domain.__class__, 'wrapped_get_next_state',
                        IWDomain_parallel_get_next_state)
            else:
                setattr(self._domain.__class__, 'wrapped_get_applicable_actions',
                        IWDomain_sequential_get_applicable_actions)
                setattr(self._domain.__class__, 'wrapped_compute_next_state',
                        IWDomain_sequential_compute_next_state)
                setattr(self._domain.__class__, 'wrapped_get_next_state',
                        IWDomain_sequential_get_next_state)
            self._solver = iw_solver(domain=self._domain,
                                     nb_of_binary_features=self._nb_of_binary_features(self._domain),
                                     state_binarizer=lambda o, f: self._state_binarizer(o, self._domain, f),
                                     use_state_feature_hash=self._use_state_feature_hash,
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
    print('AIRLAPS C++ hub library not found. Please check it is installed in your AIRLAPS_HOME.')
    raise
