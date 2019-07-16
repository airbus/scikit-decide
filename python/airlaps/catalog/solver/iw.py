from typing import Optional, Callable, Any, Iterable
import multiprocessing
import numpy as np
import math

from airlaps import DeterministicPlanningDomain, Memory, T_state, T_event, Domain
from airlaps.builders.domain import DeterministicTransitionDomain, ActionDomain, \
    DeterministicInitializedDomain, MarkovianDomain, PositiveCostDomain, FullyObservableDomain
from airlaps.builders.solver import DomainSolver, DeterministicPolicySolver, SolutionSolver, UtilitySolver
from __airlaps import _IWParSolver_ as iw_par_solver
from __airlaps import _IWSeqSolver_ as iw_seq_solver

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
    iw_ns_results[action] = iw_pool.apply_async(IWDomain_pickable_get_next_state,
                                                (self, state, action))

def IWDomain_sequential_compute_next_state(self, state, action):  # self is a domain
    global iw_ns_results
    iw_ns_results[action] = self.get_next_state(state, action)

def IWDomain_parallel_get_next_state(self, state, action):  # self is a domain
    global iw_ns_results
    return iw_ns_results[action].get()

def IWDomain_sequential_get_next_state(self, state, action):  # self is a domain
    global iw_ns_results
    return iw_ns_results[action]

class IW(DomainSolver, DeterministicPolicySolver, SolutionSolver, UtilitySolver):
    
    def __init__(self,
                 planner: str = 'bfs',
                 state_to_feature_atoms_encoder: Optional[Callable[[T_state, Domain], np.array]] = None,
                 num_tracked_atoms=0,
                 default_encoding_type: str = 'byte',
                 default_encoding_space_relative_precision: float = 0.001,
                 frameskip: int = 15,
                 simulator_budget: int = 150000,
                 time_budget: int = math.inf,
                 novelty_subtables: bool = False,
                 random_actions: bool = False,
                 max_rep: int = 30,
                 nodes_threshold: int = 50000,
                 lookahead_caching: int = 2,
                 discount: float = 1.,
                 max_depth: int = 1500,
                 break_ties_using_rewards: bool = False,
                 parallel: bool = True,
                 debug_logs: bool = False) -> None:
        self._solver = None
        self._planner = planner
        self._state_to_feature_atoms_encoder = state_to_feature_atoms_encoder
        self._num_tracked_atoms = num_tracked_atoms
        self._default_encoding_type = default_encoding_type
        self._default_encoding_space_relative_precision = default_encoding_space_relative_precision
        self._frameskip = frameskip
        self._simulator_budget = simulator_budget
        self._time_budget = time_budget
        self._novelty_subtables = novelty_subtables
        self._random_actions = random_actions
        self._max_rep = max_rep
        self._nodes_threshold = nodes_threshold
        self._lookahead_caching = lookahead_caching
        self._discount = discount
        self._max_depth = max_depth
        self._break_ties_using_rewards = break_ties_using_rewards
        self._parallel = parallel
        self._debug_logs = debug_logs

    def _reset(self) -> None:
        self._domain = self._new_domain()
        if self._parallel:
            global iw_pool
            iw_pool = multiprocessing.Pool()
            setattr(self._domain.__class__, 'wrapped_get_applicable_actions',
                    IWDomain_parallel_get_applicable_actions)
            setattr(self._domain.__class__, 'wrapped_compute_next_state',
                    IWDomain_parallel_compute_next_state)
            setattr(self._domain.__class__, 'wrapped_get_next_state',
                    IWDomain_parallel_get_next_state)
            self._solver = iw_par_solver(domain=self._domain,
                                         planner=self._planner,
                                         state_to_feature_atoms_encoder=(lambda o: self._state_to_feature_atoms_encoder(o, self._domain)) if self._state_to_feature_atoms_encoder is not None else (lambda o: np.array([], dtype=np.int64)),
                                         num_tracked_atoms=self._num_tracked_atoms,
                                         default_encoding_type=self._default_encoding_type,
                                         default_encoding_space_relative_precision=self._default_encoding_space_relative_precision,
                                         frameskip=self._frameskip,
                                         simulator_budget=self._simulator_budget,
                                         time_budget=self._time_budget,
                                         novelty_subtables=self._novelty_subtables,
                                         random_actions=self._random_actions,
                                         max_rep=self._max_rep,
                                         nodes_threshold=self._nodes_threshold,
                                         lookaheead_caching=self._lookahead_caching,
                                         discount=self._discount,
                                         max_depth=self._max_depth,
                                         break_ties_using_rewards=self._break_ties_using_rewards,
                                         debug_logs=self._debug_logs)
        else:
            setattr(self._domain.__class__, 'wrapped_get_applicable_actions',
                    IWDomain_sequential_get_applicable_actions)
            setattr(self._domain.__class__, 'wrapped_compute_next_state',
                    IWDomain_sequential_compute_next_state)
            setattr(self._domain.__class__, 'wrapped_get_next_state',
                    IWDomain_sequential_get_next_state)
            self._solver = iw_seq_solver(domain=self._domain,
                                         planner=self._planner,
                                         state_to_feature_atoms_encoder=(lambda o: self._state_to_feature_atoms_encoder(o, self._domain)) if self._state_to_feature_atoms_encoder is not None else (lambda o: np.array([], dtype=np.int64)),
                                         num_tracked_atoms=self._num_tracked_atoms,
                                         default_encoding_type=self._default_encoding_type,
                                         default_encoding_space_relative_precision = self._default_encoding_space_relative_precision,
                                         frameskip=self._frameskip,
                                         simulator_budget=self._simulator_budget,
                                         time_budget=self._time_budget,
                                         novelty_subtables=self._novelty_subtables,
                                         random_actions=self._random_actions,
                                         max_rep=self._max_rep,
                                         nodes_threshold=self._nodes_threshold,
                                         lookahead_caching=self._lookahead_caching,
                                         discount=self._discount,
                                         max_depth=self._max_depth,
                                         break_ties_using_rewards=self._break_ties_using_rewards,
                                         debug_logs=self._debug_logs)
    
    def get_domain_requirements(self) -> Iterable[type]:
        return [Domain, DeterministicTransitionDomain, ActionDomain, DeterministicInitializedDomain,
                MarkovianDomain, FullyObservableDomain, PositiveCostDomain]
    
    def _check_domain(self, domain: Domain) -> bool:
        return True # TODO : check that there are no cycles in the MDP graph (should be probably done during the search if the state space is too large)
    
    def get_next_action(self, memory: Memory[T_state]) -> T_event:
        try :
            return self._solver.get_next_action(self._domain.get_last_state(memory))
        except:
            self.solve(memory)
            return self._solver.get_next_action(self._domain.get_last_state(memory))
    
    def solve(self, from_observation: Optional[Memory[T_state]] = None,
              on_update: Optional[Callable[..., bool]] = None, max_time: Optional[float] = None,
              **kwargs: Any) -> dict:
        self._solver.solve(self._domain.get_last_state(from_observation))
    
    def get_utility(self, memory: Memory[T_state]) -> float:
        return self._solver.get_utility(self._domain.get_last_state(memory))