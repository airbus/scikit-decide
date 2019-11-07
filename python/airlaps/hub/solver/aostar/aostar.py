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
from typing import Optional, Callable

from airlaps import Domain, Solver
from airlaps import hub
from airlaps.builders.domain import SingleAgent, Sequential, EnumerableTransitions, Actions, Goals, Markovian, \
    FullyObservable, PositiveCosts
from airlaps.builders.solver import DeterministicPolicies, Utilities

record_sys_path = sys.path
airlaps_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if airlaps_cpp_extension_lib_path not in sys.path:
    sys.path.append(airlaps_cpp_extension_lib_path)

try:

    from __airlaps_hub_cpp import _AOStarParSolver_ as aostar_par_solver
    from __airlaps_hub_cpp import _AOStarSeqSolver_ as aostar_seq_solver

    # TODO: remove Markovian req?
    class D(Domain, SingleAgent, Sequential, EnumerableTransitions, Actions, Goals, Markovian, FullyObservable,
            PositiveCosts):
        pass

    aostar_pool = None  # must be separated from the domain since it cannot be pickled
    aostar_nsd_results = None  # must be separated from the domain since it cannot be pickled

    def AOStarDomain_parallel_get_applicable_actions(self, state):  # self is a domain
        global aostar_nsd_results
        actions = self.get_applicable_actions(state)
        aostar_nsd_results = {a: None for a in actions.get_elements()}
        return actions

    def AOStarDomain_sequential_get_applicable_actions(self, state):  # self is a domain
        global aostar_nsd_results
        actions = self.get_applicable_actions(state)
        aostar_nsd_results = {a: None for a in actions.get_elements()}
        return actions

    def AOStarDomain_pickable_get_next_state_distribution(domain, state, action):
        return domain.get_next_state_distribution(state, action)

    def AOStarDomain_parallel_compute_next_state_distribution(self, state, action):  # self is a domain
        global aostar_pool
        aostar_nsd_results[action] = aostar_pool.apply_async(AOStarDomain_pickable_get_next_state_distribution,
                                                            (self, state, action))

    def AOStarDomain_sequential_compute_next_state_distribution(self, state, action):  # self is a domain
        aostar_nsd_results[action] = self.get_next_state_distribution(state, action)

    def AOStarDomain_parallel_get_next_state_distribution(self, state, action):  # self is a domain
        return aostar_nsd_results[action].get()

    def AOStarDomain_sequential_get_next_state_distribution(self, state, action):  # self is a domain
        return aostar_nsd_results[action]

    class AOstar(Solver, DeterministicPolicies, Utilities):
        T_domain = D
        
        def __init__(self, heuristic: Optional[Callable[[D.T_state, Domain], float]] = None,
                     discount: float = 1.,
                     max_tip_expanions: int = 1,
                     parallel: bool = True,
                     detect_cycles: bool = False,
                     debug_logs: bool = False) -> None:
            self._solver = None
            self._heuristic = heuristic
            self._discount = discount
            self._max_tip_expansions = max_tip_expanions
            self._parallel = parallel
            self._detect_cycles = detect_cycles
            self._debug_logs = debug_logs

        def _init_solve(self, domain_factory: Callable[[], Domain]) -> None:
            self._domain = domain_factory()
            if self._parallel:
                global aostar_pool
                aostar_pool = multiprocessing.Pool()
                setattr(self._domain.__class__, 'wrapped_get_applicable_actions',
                        AOStarDomain_parallel_get_applicable_actions)
                setattr(self._domain.__class__, 'wrapped_compute_next_state_distribution',
                        AOStarDomain_parallel_compute_next_state_distribution)
                setattr(self._domain.__class__, 'wrapped_get_next_state_distribution',
                        AOStarDomain_parallel_get_next_state_distribution)
                self._solver = aostar_par_solver(domain=self._domain,
                                                 goal_checker=lambda o: self._domain.is_goal(o),
                                                 heuristic=(lambda o: self._heuristic(o, self._domain)) if self._heuristic is not None else (lambda o: 0),
                                                 discount=self._discount,
                                                 max_tip_expansions=self._max_tip_expansions,
                                                 detect_cycles=self._detect_cycles,
                                                 debug_logs=self._debug_logs)
            else:
                setattr(self._domain.__class__, 'wrapped_get_applicable_actions',
                        AOStarDomain_sequential_get_applicable_actions)
                setattr(self._domain.__class__, 'wrapped_compute_next_state_distribution',
                        AOStarDomain_sequential_compute_next_state_distribution)
                setattr(self._domain.__class__, 'wrapped_get_next_state_distribution',
                        AOStarDomain_sequential_get_next_state_distribution)
                self._solver = aostar_seq_solver(domain=self._domain,
                                                 goal_checker=lambda o: self._domain.is_goal(o),
                                                 heuristic=(lambda o: self._heuristic(o, self._domain)) if self._heuristic is not None else (lambda o: 0),
                                                 discount=self._discount,
                                                 max_tip_expansions=self._max_tip_expansions,
                                                 detect_cycles=self._detect_cycles,
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
            return self._solver.get_next_action(observation)
        
        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            return self._solver.get_utility(observation)

except ImportError:
    sys.path = record_sys_path
    print('AIRLAPS C++ hub library not found. Please check it is installed in "airlaps/hub".')
    raise
