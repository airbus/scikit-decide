from __future__ import annotations

import math
import multiprocessing
from typing import Optional, Callable

import numpy as np
from __airlaps import _IWParSolver_ as iw_par_solver
from __airlaps import _IWSeqSolver_ as iw_seq_solver

from airlaps import Domain, Solver
from airlaps.builders.domain import SingleAgent, Sequential, DeterministicTransitions, Actions, Goals, \
    DeterministicInitialized, Markovian, FullyObservable, Rewards
from airlaps.builders.solver import DeterministicPolicies, Utilities


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
    iw_ns_results[action] = iw_pool.apply_async(IWDomain_pickable_get_next_state, (self, state, action))


def IWDomain_sequential_compute_next_state(self, state, action):  # self is a domain
    global iw_ns_results
    iw_ns_results[action] = self.get_next_state(state, action)


def IWDomain_parallel_get_next_state(self, state, action):  # self is a domain
    global iw_ns_results
    return iw_ns_results[action].get()


def IWDomain_sequential_get_next_state(self, state, action):  # self is a domain
    global iw_ns_results
    return iw_ns_results[action]


class D(Domain, SingleAgent, Sequential, DeterministicTransitions, Actions, DeterministicInitialized, Markovian,
        FullyObservable, Rewards):  # TODO: check why DeterministicInitialized & PositiveCosts/Rewards?
    pass


class IW(Solver, DeterministicPolicies, Utilities):
    T_domain = D

    def __init__(self,
                 planner: str = 'bfs',
                 state_to_feature_atoms_encoder: Optional[Callable[[D.T_state, Domain], np.array]] = None,
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
                 max_depth: int = 1500,
                 break_ties_using_rewards: bool = False,
                 discount: float = 1.,
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
        self._max_depth = max_depth
        self._break_ties_using_rewards = break_ties_using_rewards
        self._discount = discount
        self._parallel = parallel
        self._debug_logs = debug_logs

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        self._domain = domain_factory()
        self._from_state = self._domain.get_initial_state()
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
                                         max_depth=self._max_depth,
                                         break_ties_using_rewards=self._break_ties_using_rewards,
                                         discount=self._discount,
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
                                         max_depth=self._max_depth,
                                         break_ties_using_rewards=self._break_ties_using_rewards,
                                         discount=self._discount,
                                         debug_logs=self._debug_logs)
        self._solver.solve(self._from_state)

    def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        return self._solver.get_next_action(observation)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True

    def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
        return self._solver.get_utility(observation)
