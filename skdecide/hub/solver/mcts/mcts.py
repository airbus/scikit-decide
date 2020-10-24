# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import sys
from math import sqrt
from typing import Callable, Dict, Tuple, List

from skdecide import Domain, Solver
from skdecide import hub
from skdecide.domains import PipeParallelDomain, ShmParallelDomain
from skdecide.builders.domain import SingleAgent, Sequential, Environment, Actions, \
    DeterministicInitialized, Markovian, FullyObservable, Rewards
from skdecide.builders.solver import ParallelSolver, DeterministicPolicies, Utilities

record_sys_path = sys.path
skdecide_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if skdecide_cpp_extension_lib_path not in sys.path:
    sys.path.append(skdecide_cpp_extension_lib_path)

try:

    from __skdecide_hub_cpp import _MCTSSolver_ as mcts_solver
    from __skdecide_hub_cpp import _MCTSOptions_ as mcts_options

    class D(Domain, SingleAgent, Sequential, Environment, Actions, DeterministicInitialized, Markovian,
            FullyObservable, Rewards):  # TODO: check why DeterministicInitialized & PositiveCosts/Rewards?
        pass
    
    
    class MCTS(ParallelSolver, Solver, DeterministicPolicies, Utilities):
        T_domain = D

        Options = mcts_options

        def __init__(self,
                     domain_factory: Callable[[], Domain],
                     time_budget: int = 3600000,
                     rollout_budget: int = 100000,
                     max_depth: int = 1000,
                     discount: float = 1.0,
                     uct_mode: bool = True,
                     ucb_constant: float = 1.0 / sqrt(2.0),
                     online_node_garbage: bool = False,
                     custom_policy: Callable[[Domain, D.T_agent[D.T_observation]], D.T_agent[D.T_concurrency[D.T_event]]] = None,
                     heuristic: Callable[[Domain, D.T_agent[D.T_observation]], Tuple[float, int]] = None,
                     transition_mode: Options.TransitionMode = Options.TransitionMode.Distribution,
                     tree_policy: Options.TreePolicy = Options.TreePolicy.Default,
                     expander: Options.Expander = Options.Expander.Full,
                     action_selector_optimization: Options.ActionSelector = Options.ActionSelector.UCB1,
                     action_selector_execution: Options.ActionSelector = Options.ActionSelector.BestQValue,
                     rollout_policy: Options.RolloutPolicy = Options.RolloutPolicy.Random,
                     back_propagator: Options.BackPropagator = Options.BackPropagator.Graph,
                     continuous_planning: bool = True,
                     parallel: bool = False,
                     shared_memory_proxy = None,
                     debug_logs: bool = False) -> None:
            ParallelSolver.__init__(self,
                                    domain_factory=domain_factory,
                                    parallel=parallel,
                                    shared_memory_proxy=shared_memory_proxy)
            self._solver = None
            self._domain = None
            self._time_budget = time_budget
            self._rollout_budget = rollout_budget
            self._max_depth = max_depth
            self._discount = discount
            self._uct_mode = uct_mode
            self._ucb_constant = ucb_constant
            self._online_node_garbage = online_node_garbage
            self._custom_policy = custom_policy
            self._heuristic = heuristic
            self._transition_mode = transition_mode
            self._tree_policy = tree_policy
            self._expander = expander
            self._action_selector_optimization = action_selector_optimization
            self._action_selector_execution = action_selector_execution
            self._rollout_policy = rollout_policy
            self._back_propagator = back_propagator
            self._continuous_planning = continuous_planning
            self._debug_logs = debug_logs
            self._lambdas = [self._custom_policy, self._heuristic]
            self._ipc_notify = True
        
        def _init_solve(self, domain_factory: Callable[[], D]) -> None:
            self._domain_factory = domain_factory
            self._solver = mcts_solver(domain=self.get_domain(),
                                       time_budget=self._time_budget,
                                       rollout_budget=self._rollout_budget,
                                       max_depth=self._max_depth,
                                       discount=self._discount,
                                       uct_mode=self._uct_mode,
                                       ucb_constant=self._ucb_constant,
                                       online_node_garbage=self._online_node_garbage,
                                       custom_policy=None if self._custom_policy is None \
                                                          else lambda d, s, i= \
                                                               None: self._custom_policy(d, s) if not self._parallel \
                                                                                               else d.call(i, 0, s),
                                       heuristic=None if self._heuristic is None \
                                                      else lambda d, s, i= \
                                                           None: self._heuristic(d, s) if not self._parallel \
                                                                                       else d.call(i, 1, s),
                                       transition_mode=self._transition_mode,
                                       tree_policy=self._tree_policy,
                                       expander=self._expander,
                                       action_selector_optimization=self._action_selector_optimization,
                                       action_selector_execution=self._action_selector_execution,
                                       rollout_policy=self._rollout_policy,
                                       back_propagator=self._back_propagator,
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
            if self._continuous_planning or not self._is_solution_defined_for(observation):
                self._solve_from(observation)
            action = self._solver.get_next_action(observation)
            if action is None:
                print('\x1b[3;33;40m' + 'No best action found in observation ' +
                      str(observation) + ', applying random action' + '\x1b[0m')
                return self.call_domain_method('get_action_space').sample()
            else:
                return action
        
        def _reset(self) -> None:
            self._solver.clear()
        
        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            return self._solver.get_utility(observation)
        
        def get_nb_of_explored_states(self) -> int:
            return self._solver.get_nb_of_explored_states()
        
        def get_nb_rollouts(self) -> int:
            return self._solver.get_nb_rollouts()
        
        def get_policy(self) -> Dict[D.T_agent[D.T_observation], Tuple[D.T_agent[D.T_concurrency[D.T_event]], float]]:
            return self._solver.get_policy()
        
        def get_action_prefix(self) -> List[D.T_agent[D.T_observation]]:
            return self._solver.get_action_prefix()
    

    class UCT(MCTS):
        def __init__(self,
                     domain_factory: Callable[[], Domain],
                     time_budget: int = 3600000,
                     rollout_budget: int = 100000,
                     max_depth: int = 1000,
                     discount: float = 1.0,
                     ucb_constant: float = 1.0 / sqrt(2.0),
                     online_node_garbage: float = False,
                     custom_policy: Callable[[Domain, D.T_agent[D.T_observation]], D.T_agent[D.T_concurrency[D.T_event]]] = None,
                     heuristic: Callable[[Domain, D.T_agent[D.T_observation]], Tuple[float, int]] = None,
                     transition_mode: mcts_options.TransitionMode = mcts_options.TransitionMode.Distribution,
                     rollout_policy: mcts_options.RolloutPolicy = mcts_options.RolloutPolicy.Random,
                     continuous_planning: bool = True,
                     parallel: bool = False,
                     shared_memory_proxy = None,
                     debug_logs: bool = False) -> None:
            super().__init__(domain_factory=domain_factory,
                             time_budget=time_budget,
                             rollout_budget=rollout_budget,
                             max_depth=max_depth,
                             discount=discount,
                             uct_mode=True,
                             ucb_constant=ucb_constant,
                             online_node_garbage=online_node_garbage,
                             custom_policy=custom_policy,
                             heuristic=heuristic,
                             transition_mode=transition_mode,
                             rollout_policy=rollout_policy,
                             continuous_planning=continuous_planning,
                             parallel=parallel,
                             shared_memory_proxy=shared_memory_proxy,
                             debug_logs=debug_logs)
    
except ImportError:
    sys.path = record_sys_path
    print('Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".')
    raise
