from typing import Optional, Callable, Any, Iterable

from airlaps import Memory, T_state, T_event, Domain
from airlaps.builders.domain import EnumerableTransitionDomain, ActionDomain, GoalDomain, \
    DeterministicInitializedDomain, MarkovianDomain, PositiveCostDomain, FullyObservableDomain
from airlaps.builders.solver import DomainSolver, DeterministicPolicySolver, SolutionSolver, UtilitySolver
from __airlaps import __AOStarSolver as aostar_solver

class AOstar(DomainSolver, DeterministicPolicySolver, SolutionSolver, UtilitySolver):
    
    def __init__(self, heuristic: Optional[Callable[[T_state, Domain], float]] = None,
                 discount: float = 1.,
                 max_tip_expanions: int = 1,
                 weight: float = 1.,
                 detect_cycles: bool = False,
                 debug_logs: bool = False) -> None:
        self._solver = None
        self._heuristic = heuristic
        self._discount = discount
        self._max_tip_expansions = max_tip_expanions
        self._detect_cycles = detect_cycles
        self._debug_logs = debug_logs

    def _reset(self) -> None:
        self._domain = self._new_domain()
        self._solver = aostar_solver(domain=self._domain,
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