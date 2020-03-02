# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import multiprocessing
import os
import sys
from typing import Callable, Any, List, Tuple

from skdecide import Domain, Solver
from skdecide import hub
from skdecide.domains import PipeParallelDomain, ShmParallelDomain
from skdecide.builders.domain import SingleAgent, Sequential, DeterministicTransitions, Actions, \
    DeterministicInitialized, Markovian, FullyObservable, Rewards
from skdecide.builders.solver import DeterministicPolicies, Utilities
from skdecide.hub.space.gym import ListSpace

record_sys_path = sys.path
skdecide_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if skdecide_cpp_extension_lib_path not in sys.path:
    sys.path.append(skdecide_cpp_extension_lib_path)

try:

    from __skdecide_hub_cpp import _IWSolver_ as iw_solver

    class D(Domain, SingleAgent, Sequential, DeterministicTransitions, Actions, DeterministicInitialized, Markovian,
            FullyObservable, Rewards):  # TODO: check why DeterministicInitialized & PositiveCosts/Rewards?
        pass


    class IW(Solver, DeterministicPolicies, Utilities):
        T_domain = D

        def __init__(self,
                     state_features: Callable[[Domain, D.T_state], Any],
                     use_state_feature_hash: bool = False,
                     node_ordering: Callable[[float, int, int, float, int, int], bool] = None,
                     time_budget: int = 0,  # time budget to continue searching for better plans after a goal has been reached
                     parallel: bool = True,
                     shared_memory_proxy = None,
                     debug_logs: bool = False) -> None:
            self._solver = None
            self._domain = None
            self._state_features = state_features
            self._use_state_feature_hash = use_state_feature_hash
            self._node_ordering = node_ordering
            self._time_budget = time_budget
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
            self._solver = iw_solver(domain=self._domain,
                                     state_features=lambda d, s: self._state_features(d, s) if not self._parallel else d.call(None, self._state_features, s),
                                     use_state_feature_hash=self._use_state_feature_hash,
                                     node_ordering=self._node_ordering,
                                     time_budget=self._time_budget,
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
        
        def _reset(self) -> None:
            self._solver.clear()
        
        def get_nb_of_explored_states(self) -> int:
            return self._solver.get_nb_of_explored_states()
        
        def get_nb_of_pruned_states(self) -> int:
            return self._solver.get_nb_of_pruned_states()
        
        def get_intermediate_scores(self) -> List[Tuple[int, float]]:
            return self._solver.get_intermediate_scores()
    
except ImportError:
    sys.path = record_sys_path
    print('Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".')
    raise
