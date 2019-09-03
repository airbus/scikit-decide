from typing import Optional, Callable, Any, Iterable
import multiprocessing

from airlaps import Domain, Solver
from airlaps.builders.domain import SingleAgent, Sequential, DeterministicTransitions, Actions, Goals, Markovian, \
    FullyObservable, PositiveCosts
from airlaps.builders.solver import DeterministicPolicies, Utilities

import sys, os
from airlaps import hub

record_sys_path = sys.path
airlaps_cpp_extension_lib_path = os.path.join(hub._get_airlaps_home(), 'lib')
if airlaps_cpp_extension_lib_path not in sys.path:
    sys.path.append(airlaps_cpp_extension_lib_path)

try:
    from __airlaps_hub_cpp import _AStarParSolver_ as astar_par_solver
    from __airlaps_hub_cpp import _AStarSeqSolver_ as astar_seq_solver

    # TODO: remove Markovian req?
    class D(Domain, SingleAgent, Sequential, DeterministicTransitions, Actions, Goals, Markovian, FullyObservable,
            PositiveCosts):
        pass

    astar_pool = None  # must be separated from the domain since it cannot be pickled
    astar_nsd_results = None  # must be separated from the domain since it cannot be pickled

    def AStarDomain_parallel_get_applicable_actions(self, state):  # self is a domain
        global astar_nsd_results
        actions = self.get_applicable_actions(state)
        astar_nsd_results = {a: None for a in actions.get_elements()}
        return actions

    def AStarDomain_sequential_get_applicable_actions(self, state):  # self is a domain
        actions = self.get_applicable_actions(state)
        astar_nsd_results = {a: None for a in actions.get_elements()}
        return actions

    def AStarDomain_pickable_get_next_state(domain, state, action):
        return domain.get_next_state(state, action)

    def AStarDomain_parallel_compute_next_state(self, state, action):  # self is a domain
        global astar_pool
        astar_nsd_results[action] = astar_pool.apply_async(AStarDomain_pickable_get_next_state,
                                                        (self, state, action))

    def AStarDomain_sequential_compute_next_state(self, state, action):  # self is a domain
        astar_nsd_results[action] = self.get_next_state(state, action)

    def AStarDomain_parallel_get_next_state(self, state, action):  # self is a domain
        return astar_nsd_results[action].get()

    def AStarDomain_sequential_get_next_state(self, state, action):  # self is a domain
        return astar_nsd_results[action]

    class Astar(Solver, DeterministicPolicies, Utilities):
        T_domain = D
        
        def __init__(self, heuristic: Optional[Callable[[D.T_state, Domain], float]] = None,
                    parallel: bool = True,
                    debug_logs: bool = False) -> None:
            self._solver = None
            self._domain = None
            self._heuristic = heuristic
            self._parallel = parallel
            self._debug_logs = debug_logs

        def _init_solve(self, domain_factory: Callable[[], Domain]) -> Domain:
            self._domain = domain_factory()
            if self._parallel:
                global astar_pool
                astar_pool = multiprocessing.Pool()
                setattr(self._domain.__class__, 'wrapped_get_applicable_actions',
                        AStarDomain_parallel_get_applicable_actions)
                setattr(self._domain.__class__, 'wrapped_compute_next_state',
                        AStarDomain_parallel_compute_next_state)
                setattr(self._domain.__class__, 'wrapped_get_next_state',
                        AStarDomain_parallel_get_next_state)
                self._solver = astar_par_solver(domain=self._domain,
                                                goal_checker=lambda o: self._domain.is_goal(o),
                                                heuristic=(lambda o: self._heuristic(o, self._domain)) if self._heuristic is not None else (lambda o: 0),
                                                debug_logs=self._debug_logs)
            else:
                setattr(self._domain.__class__, 'wrapped_get_applicable_actions',
                        AStarDomain_sequential_get_applicable_actions)
                setattr(self._domain.__class__, 'wrapped_compute_next_state',
                        AStarDomain_sequential_compute_next_state)
                setattr(self._domain.__class__, 'wrapped_get_next_state',
                        AStarDomain_sequential_get_next_state)
                self._solver = astar_seq_solver(domain=self._domain,
                                                goal_checker=lambda o: self._domain.is_goal(o),
                                                heuristic=(lambda o: self._heuristic(o, self._domain)) if self._heuristic is not None else (lambda o: 0),
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
