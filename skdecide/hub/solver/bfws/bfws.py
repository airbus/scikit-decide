# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import multiprocessing
import os
import sys
from typing import Callable

from skdecide import Domain, Solver
from skdecide import hub
from skdecide.domains import PipeParallelDomain, ShmParallelDomain
from skdecide.builders.domain import SingleAgent, Sequential, DeterministicTransitions, Actions, \
    DeterministicInitialized, Markovian, FullyObservable, Rewards
from skdecide.builders.solver import DeterministicPolicies, Utilities

record_sys_path = sys.path
skdecide_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if skdecide_cpp_extension_lib_path not in sys.path:
    sys.path.append(skdecide_cpp_extension_lib_path)

try:

    from __skdecide_hub_cpp import _BFWSSolver_ as bfws_solver


    class D(Domain, SingleAgent, Sequential, DeterministicTransitions, Actions, DeterministicInitialized, Markovian,
            FullyObservable, Rewards):  # TODO: check why DeterministicInitialized & PositiveCosts/Rewards?
        pass


    class BFWS(Solver, DeterministicPolicies, Utilities):
        T_domain = D

        def __init__(self,
                     state_binarizer: Callable[[D.T_state, Domain, Callable[[int], None]], None],
                     heuristic: Callable[[D.T_state, Domain], float],
                     termination_checker: Callable[[D.T_state, Domain], bool],
                     parallel: bool = True,
                     shared_memory_proxy = None,
                     debug_logs: bool = False) -> None:
            self._solver = None
            self._domain = None
            self._state_binarizer = state_binarizer
            self._heuristic = heuristic
            self._termination_checker = termination_checker
            self._parallel = parallel
            self._shared_memory_proxy = shared_memory_proxy
            self._debug_logs = debug_logs

        def _init_solve(self, domain_factory: Callable[[], D]) -> None:
            if self._parallel:
                if self._shared_memory_proxy is None:
                    self._domain = PipeParallelDomain(domain_factory)
                else:
                    self._domain = ShmParallelDomain(domain_factory, self._shared_memory_proxy)
            else:
                self._domain = domain_factory()
            self._solver = bfws_solver(domain=self._domain,
                                       state_binarizer=lambda o, f: self._state_binarizer(o, self._domain, f),
                                       heuristic=lambda o: self._heuristic(o, self._domain),
                                       termination_checker=lambda o: self._termination_checker(o, self._domain),
                                       parallel=self._parallel,
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
    print('Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".')
    raise
