# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import multiprocessing
import os
import sys
from typing import Optional, Callable

from skdecide import Domain, Solver
from skdecide import hub
from skdecide.domains import PipeParallelDomain, ShmParallelDomain
from skdecide.builders.domain import SingleAgent, Sequential, EnumerableTransitions, Actions, Goals, Markovian, \
    FullyObservable, PositiveCosts
from skdecide.builders.solver import DeterministicPolicies, Utilities

record_sys_path = sys.path
skdecide_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if skdecide_cpp_extension_lib_path not in sys.path:
    sys.path.append(skdecide_cpp_extension_lib_path)

try:

    from __skdecide_hub_cpp import _AOStarSolver_ as aostar_solver

    # TODO: remove Markovian req?
    class D(Domain, SingleAgent, Sequential, EnumerableTransitions, Actions, Goals, Markovian, FullyObservable,
            PositiveCosts):
        pass

    class AOstar(Solver, DeterministicPolicies, Utilities):
        T_domain = D
        
        def __init__(self,
                     heuristic: Optional[Callable[[Domain, D.T_state], float]] = None,
                     discount: float = 1.,
                     max_tip_expanions: int = 1,
                     parallel: bool = True,
                     shared_memory_proxy = None,
                     detect_cycles: bool = False,
                     debug_logs: bool = False) -> None:
            self._solver = None
            self._heuristic = heuristic
            self._discount = discount
            self._max_tip_expansions = max_tip_expanions
            self._parallel = parallel
            self._shared_memory_proxy = shared_memory_proxy
            self._detect_cycles = detect_cycles
            self._debug_logs = debug_logs

        def _init_solve(self, domain_factory: Callable[[], Domain]) -> None:
            if self._parallel:
                if self._shared_memory_proxy is None:
                    self._domain = PipeParallelDomain(domain_factory)
                else:
                    self._domain = ShmParallelDomain(domain_factory, self._shared_memory_proxy)
            else:
                self._domain = domain_factory()
            if self._heuristic is None:
                heuristic = lambda d, s: 0
            else:
                heuristic = self._heuristic
            self._solver = aostar_solver(domain=self._domain,
                                         goal_checker=lambda d, s: d.is_goal(s),
                                         heuristic=lambda d, s: heuristic(d, s) if not self._parallel else d.call(None, heuristic, s),
                                         discount=self._discount,
                                         max_tip_expansions=self._max_tip_expansions,
                                         detect_cycles=self._detect_cycles,
                                         parallel=self._parallel,
                                         debug_logs=self._debug_logs)
            self._solver.clear()
        
        def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
            self._init_solve(domain_factory)

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
            if self._parallel:
                self._domain.start_session(ipc_notify=True)
            self._solver.solve(memory)
            if self._parallel:
                self._domain.end_session()
        
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
