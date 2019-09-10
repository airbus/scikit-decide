from __future__ import annotations

import math
import multiprocessing
from typing import Optional, Callable

import numpy as np

from airlaps import Domain, Solver
from airlaps.builders.domain import SingleAgent, Sequential, DeterministicTransitions, Actions, Goals, \
    DeterministicInitialized, Markovian, FullyObservable, Rewards
from airlaps.builders.solver import DeterministicPolicies, Utilities

import sys, os
from airlaps import hub

record_sys_path = sys.path
airlaps_cpp_extension_lib_path = os.path.join(hub._get_airlaps_home(), 'lib')
if airlaps_cpp_extension_lib_path not in sys.path:
    sys.path.append(airlaps_cpp_extension_lib_path)

try:

    from __airlaps_hub_cpp import _BFWSParSolver_ as bfws_par_solver
    from __airlaps_hub_cpp import _BFWSSeqSolver_ as bfws_seq_solver

    bfws_pool = None  # must be separated from the domain since it cannot be pickled
    bfws_ns_results = None  # must be separated from the domain since it cannot be pickled


    def BFWSDomain_parallel_get_applicable_actions(self, state):  # self is a domain
        global bfws_ns_results
        actions = self.get_applicable_actions(state)
        bfws_ns_results = {a: None for a in actions.get_elements()}
        return actions


    def BFWSDomain_sequential_get_applicable_actions(self, state):  # self is a domain
        global bfws_ns_results
        actions = self.get_applicable_actions(state)
        bfws_ns_results = {a: None for a in actions.get_elements()}
        return actions


    def BFWSDomain_pickable_get_next_state(domain, state, action):
        return domain.get_next_state(state, action)


    def BFWSDomain_parallel_compute_next_state(self, state, action):  # self is a domain
        global bfws_pool, bfws_ns_results
        bfws_ns_results[action] = bfws_pool.apply_async(BFWSDomain_pickable_get_next_state, (self, state, action))


    def BFWSDomain_sequential_compute_next_state(self, state, action):  # self is a domain
        global bfws_ns_results
        bfws_ns_results[action] = self.get_next_state(state, action)


    def BFWSDomain_parallel_get_next_state(self, state, action):  # self is a domain
        global bfws_ns_results
        return bfws_ns_results[action].get()


    def BFWSDomain_sequential_get_next_state(self, state, action):  # self is a domain
        global bfws_ns_results
        return bfws_ns_results[action]


    class D(Domain, SingleAgent, Sequential, DeterministicTransitions, Actions, DeterministicInitialized, Markovian,
            FullyObservable, Rewards):  # TODO: check why DeterministicInitialized & PositiveCosts/Rewards?
        pass


    class BFWS(Solver, DeterministicPolicies, Utilities):
        T_domain = D

        def __init__(self,
                     state_binarizer: Callable[[D.T_state, Domain, Callable[[int], None], None]],
                     heuristic: Callable[[D.T_state, Domain], float],
                     termination_checker: Callable[[D.T_state, Domain], bool],
                     parallel: bool = True,
                     debug_logs: bool = False) -> None:
            self._solver = None
            self._domain = None
            self._state_binarizer = state_binarizer
            self._heuristic = heuristic
            self._termination_checker = termination_checker
            self._parallel = parallel
            self._debug_logs = debug_logs

        def _init_solve(self, domain_factory: Callable[[], D]) -> None:
            self._domain = domain_factory()
            if self._parallel:
                global bfws_pool
                bfws_pool = multiprocessing.Pool()
                setattr(self._domain.__class__, 'wrapped_get_applicable_actions',
                        BFWSDomain_parallel_get_applicable_actions)
                setattr(self._domain.__class__, 'wrapped_compute_next_state',
                        BFWSDomain_parallel_compute_next_state)
                setattr(self._domain.__class__, 'wrapped_get_next_state',
                        BFWSDomain_parallel_get_next_state)
                self._solver = bfws_par_solver(domain=self._domain,
                                               state_binarizer=lambda o, f: self._state_binarizer(o, self._domain, f),
                                               heuristic=lambda o: self._heuristic(o, self._domain),
                                               termination_checker=lambda o: self._termination_checker(o, self._domain),
                                               debug_logs=self._debug_logs)
            else:
                setattr(self._domain.__class__, 'wrapped_get_applicable_actions',
                        BFWSDomain_sequential_get_applicable_actions)
                setattr(self._domain.__class__, 'wrapped_compute_next_state',
                        BFWSDomain_sequential_compute_next_state)
                setattr(self._domain.__class__, 'wrapped_get_next_state',
                        BFWSDomain_sequential_get_next_state)
                self._solver = bfws_seq_solver(domain=self._domain,
                                               state_binarizer=lambda o, f: self._state_binarizer(o, self._domain, f),
                                               heuristic=lambda o: self._heuristic(o, self._domain),
                                               termination_checker=lambda o: self._termination_checker(o, self._domain),
                                               debug_logs=self._debug_logs)
            self._solver.clear()

        def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
            self._init_solve(domain_factory)
            self._solve_from(self._domain.get_initial_state())

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
