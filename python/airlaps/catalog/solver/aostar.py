from typing import Optional, Callable, Any, Iterable
import multiprocessing

from airlaps import Memory, T_state, T_event, Domain
from airlaps.builders.domain import EnumerableTransitionDomain, ActionDomain, GoalDomain, \
    DeterministicInitializedDomain, MarkovianDomain, PositiveCostDomain, FullyObservableDomain
from airlaps.builders.solver import DomainSolver, DeterministicPolicySolver, SolutionSolver, UtilitySolver
from __airlaps import __AOStarParSolver as aostar_par_solver
from __airlaps import __AOStarSeqSolver as aostar_seq_solver

aostar_pool = None  # must be separated from the domain since it cannot be pickled
aostar_nsd_results = None  # must be separated from the domain since it cannot be pickled

def AOStarDomain_parallel_get_applicable_actions(self, state):  # self is a domain
    global aostar_nsd_results
    actions = self.get_applicable_actions(state)
    aostar_nsd_results = {a: None for a in actions.get_elements()}
    return actions

def AOStarDomain_sequential_get_applicable_actions(self, state):  # self is a domain
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

class AOstar(DomainSolver, DeterministicPolicySolver, SolutionSolver, UtilitySolver):
    
    def __init__(self, heuristic: Optional[Callable[[T_state, Domain], float]] = None,
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

    def _reset(self) -> None:
        self._domain = self._new_domain()
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
    
    def get_domain_requirements(self) -> Iterable[type]:
        return [Domain, EnumerableTransitionDomain, ActionDomain, GoalDomain, DeterministicInitializedDomain,
                MarkovianDomain, FullyObservableDomain, PositiveCostDomain]
    
    def _check_domain(self, domain: Domain) -> bool:
        return True # TODO : check that there are no cycles in the MDP graph (should be probably done during the search if the state space is too large)
    
    def get_next_action(self, memory: Memory[T_state]) -> T_event:
        return self._solver.get_next_action(self._domain.get_last_state(memory))
    
    def solve(self, from_observation: Optional[Memory[T_state]] = None,
              on_update: Optional[Callable[..., bool]] = None, max_time: Optional[float] = None,
              **kwargs: Any) -> dict:
        self._solver.solve(self._domain.get_last_state(from_observation))
    
    def get_utility(self, memory: Memory[T_state]) -> float:
        return self._solver.get_utility(self._domain.get_last_state(memory))