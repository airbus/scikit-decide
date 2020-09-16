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
from skdecide.builders.domain import SingleAgent, Sequential, DeterministicTransitions, Actions, Goals, Markovian, \
    FullyObservable, PositiveCosts
from skdecide.builders.solver import ParallelSolver, DeterministicPolicies, Utilities

record_sys_path = sys.path
skdecide_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if skdecide_cpp_extension_lib_path not in sys.path:
    sys.path.append(skdecide_cpp_extension_lib_path)
    
# GAL
current_path = os.path.abspath(os.path.dirname(__file__))

try:
    if os.name == 'nt':
        print("current path: ", current_path)
        print("skdecide_cpp_extension_lib_path: ", skdecide_cpp_extension_lib_path)

        os.environ['path'] = skdecide_cpp_extension_lib_path + ';' + os.environ['path']
        sys.path.insert(0, skdecide_cpp_extension_lib_path)
        
        # Note: from python3.8, PATH will not take effect
        # https://github.com/python/cpython/pull/12302
        # Use add_dll_directory to specify dll resolution path
        if sys.version_info[:2] >= (3, 8):
            os.add_dll_directory(skdecide_cpp_extension_lib_path)

except ImportError as e:
    from __skdecide_hub_cpp import _AStarSolver_ as astar_solver
    if os.name == 'nt':
        executable_path = os.path.abspath(os.path.dirname(sys.executable))
        raise ImportError(
            """NOTE: You may need to run \"set PATH=%s;%%PATH%%\"
        if you encounters \"DLL load failed\" errors. If you have python
        installed in other directory, replace \"%s\" with your own
        directory. The original error is: \n %s""" %
            (executable_path, executable_path, astar_solver.get_exception_message(e)))
    else:
        raise ImportError(
            """NOTE: You may need to run \"export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH\"
        if you encounters \"libmkldnn.so not found\" errors. If you have python
        installed in other directory, replace \"/usr/local/lib\" with your own
        directory. The original error is: \n""" + astar_solver.get_exception_message(e))
except Exception as e:
    raise e

try:

    from __skdecide_hub_cpp import _AStarSolver_ as astar_solver

    # TODO: remove Markovian req?
    class D(Domain, SingleAgent, Sequential, DeterministicTransitions, Actions, Goals, Markovian, FullyObservable,
            PositiveCosts):
        pass


    class Astar(ParallelSolver, Solver, DeterministicPolicies, Utilities):
        T_domain = D
        
        def __init__(self,
                     domain_factory: Callable[[], Domain] = None,
                     heuristic: Optional[Callable[[Domain, D.T_state], float]] = None,
                     parallel: bool = False,
                     shared_memory_proxy = None,
                     debug_logs: bool = False) -> None:
            ParallelSolver.__init__(self,
                                    domain_factory=domain_factory,
                                    parallel=parallel,
                                    shared_memory_proxy=shared_memory_proxy)
            self._solver = None
            self._debug_logs = debug_logs
            if heuristic is None:
                self._heuristic = lambda d, s: 0
            else:
                self._heuristic = heuristic
            self._lambdas = [self._heuristic]
            self._ipc_notify = True

        def _init_solve(self, domain_factory: Callable[[], Domain]) -> None:
            self._domain_factory = domain_factory
            self._solver = astar_solver(domain=self.get_domain(),
                                        goal_checker=lambda d, s: d.is_goal(s),
                                        heuristic=lambda d, s: self._heuristic(d, s) if not self._parallel else d.call(None, 0, s),
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
    print('Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".')
    raise
