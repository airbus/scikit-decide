# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import sys
from math import sqrt
from typing import Callable, Dict, Tuple, Any

from skdecide import Domain, Solver
from skdecide import hub
from skdecide.domains import PipeParallelDomain, ShmParallelDomain
from skdecide.builders.domain import SingleAgent, Sequential, Environment, Actions, \
    DeterministicInitialized, Markovian, FullyObservable, Rewards
from skdecide.builders.solver import DeterministicPolicies, Utilities

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
    
    
    class MCTS(Solver, DeterministicPolicies, Utilities):
        T_domain = D

        Options = mcts_options

        def __init__(self,
                     time_budget: int = 3600000,
                     rollout_budget: int = 100000,
                     max_depth: int = 1000,
                     discount: float = 1.0,
                     uct_mode: bool = True,
                     ucb_constant: float = 1.0 / sqrt(2.0),
                     rollout_policy_functor: Callable[[Domain, D.T_agent[D.T_observation]], D.T_agent[D.T_concurrency[D.T_event]]] = None,
                     transition_mode: Options.TransitionMode = Options.TransitionMode.Distribution,
                     tree_policy: Options.TreePolicy = Options.TreePolicy.Default,
                     expander: Options.Expander = Options.Expander.Full,
                     action_selector_optimization: Options.ActionSelector = Options.ActionSelector.UCB1,
                     action_selector_execution: Options.ActionSelector = Options.ActionSelector.BestQValue,
                     rollout_policy: Options.RolloutPolicy = Options.RolloutPolicy.Random,
                     back_propagator: Options.BackPropagator = Options.BackPropagator.Graph,
                     continuous_planning: bool = True,
                     parallel: bool = True,
                     shared_memory_proxy = None,
                     debug_logs: bool = False) -> None:
            self._solver = None
            self._domain = None
            self._time_budget = time_budget
            self._rollout_budget = rollout_budget
            self._max_depth = max_depth
            self._discount = discount
            self._uct_mode = uct_mode
            self._ucb_constant = ucb_constant
            self._rollout_policy_functor = rollout_policy_functor
            self._transition_mode = transition_mode
            self._tree_policy = tree_policy
            self._expander = expander
            self._action_selector_optimization = action_selector_optimization
            self._action_selector_execution = action_selector_execution
            self._rollout_policy = rollout_policy
            self._back_propagator = back_propagator
            self._continuous_planning = continuous_planning
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
            self._solver = mcts_solver(domain=self._domain,
                                       time_budget=self._time_budget,
                                       rollout_budget=self._rollout_budget,
                                       max_depth=self._max_depth,
                                       discount=self._discount,
                                       uct_mode=self._uct_mode,
                                       ucb_constant=self._ucb_constant,
                                       rollout_policy_functor=lambda d, s, i=None: self._rollout_policy_functor(d, s) if not self._parallel else d.call(i, self._rollout_policy_functor, s),
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
            if self._parallel:
                self._domain.start_session(ipc_notify=True)
            self._solver.solve(memory)
            if self._parallel:
                self._domain.end_session()
        
        def _is_solution_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
            return self._solver.is_solution_defined_for(observation)
        
        def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
            if self._continuous_planning or not self._is_solution_defined_for(observation):
                self._solve_from(observation)
            action = self._solver.get_next_action(observation)
            if action is None:
                print('\x1b[3;33;40m' + 'No best action found in observation ' +
                      str(observation) + ', applying random action' + '\x1b[0m')
                if not self._parallel:
                    return self._domain.get_action_space().sample()
                else:
                    self._domain.start_session(ipc_notify=False)
                    domain_id = self._domain.get_action_space()
                    self._domain.wait_job(domain_id)
                    r = self._domain.get_result(domain_id).sample()
                    self._domain.end_session()
                    return r
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
                     time_budget: int = 3600000,
                     rollout_budget: int = 100000,
                     max_depth: int = 1000,
                     discount: float = 1.0,
                     ucb_constant: float = 1.0 / sqrt(2.0),
                     rollout_policy_functor: Callable[[Domain, D.T_agent[D.T_observation]], D.T_agent[D.T_concurrency[D.T_event]]] = None,
                     transition_mode: mcts_options.TransitionMode = mcts_options.TransitionMode.Distribution,
                     rollout_policy: Options.RolloutPolicy = mcts_options.RolloutPolicy.Random,
                     continuous_planning: bool = True,
                     parallel: bool = True,
                     shared_memory_proxy = None,
                     debug_logs: bool = False) -> None:
            super().__init__(time_budget=time_budget,
                             rollout_budget=rollout_budget,
                             max_depth=max_depth,
                             discount=discount,
                             uct_mode=True,
                             ucb_constant=ucb_constant,
                             rollout_policy_functor=rollout_policy_functor,
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
