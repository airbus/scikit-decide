# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import Callable, Any

from skdecide import Solver
from skdecide import hub

from skdecide import Domain, RLDomain, StatelessSimulatorDomain, \
                    Space, EnvironmentOutcome, TransitionValue
from skdecide.builders.domain import Simulation, Environment, \
                                    Initializable, Rewards, Memoryless, \
                                    FullyObservable, Sequential, SingleAgent
from skdecide.builders.solver import Policies

record_sys_path = sys.path
skdecide_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if skdecide_cpp_extension_lib_path not in sys.path:
    sys.path.append(skdecide_cpp_extension_lib_path)

try:

    from __skdecide_hub_cpp import _WRLDomainFilter_ as wrl_domain_filter
    from __skdecide_hub_cpp import _WRLSolver_ as wrl_solver

    class D(RLDomain, FullyObservable):
        pass


    class WidthEnvironmentDomain(D):
        T_domain = D

        def __init__(self, domain: T_domain,
                           state_features: Callable[[D.T_observation, Domain], Any],
                           initial_pruning_probability: float = 0.999,
                           temperature_increase_rate: float = 0.01,
                           width_increase_resilience: int = 10,
                           max_depth: int = 1000,
                           use_state_feature_hash: bool = False,
                           cache_transitions: bool = False,
                           debug_logs: bool = False) -> None:
            self._original_domain = domain
            self._domain = wrl_domain_filter(domain,
                                             lambda o: state_features(o, domain),
                                             initial_pruning_probability,
                                             temperature_increase_rate,
                                             width_increase_resilience,
                                             max_depth,
                                             use_state_feature_hash,
                                             cache_transitions,
                                             debug_logs)
        
        def get_original_domain(self):
            return self._original_domain
        
        def get_action_space(self) -> D.T_agent[Space[D.T_event]]:
            return self._original_domain.get_action_space()
        
        def get_observation_space(self) -> D.T_agent[Space[D.T_observation]]:
            return self._original_domain.get_observation_space()
        
        def reset(self) -> D.T_agent[D.T_observation]:
            return self._domain.reset()
        
        def step(self, action: D.T_agent[D.T_concurrency[D.T_event]]) -> EnvironmentOutcome[
                    D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
            return self._domain.step(action)
    

    class D(StatelessSimulatorDomain, FullyObservable):
        pass

    class WidthSimulationDomain(D):
        T_domain = D

        def __init__(self, domain: T_domain,
                           state_features: Callable[[D.T_state, Domain], Any],
                           initial_pruning_probability: float = 0.999,
                           temperature_increase_rate: float = 0.01,
                           width_increase_resilience: int = 10,
                           max_depth: int = 1000,
                           use_state_feature_hash: bool = False,
                           cache_transitions: bool = False,
                           debug_logs: bool = False) -> None:
            self._original_domain = domain
            self._domain = wrl_domain_filter(domain,
                                             lambda o: state_features(o, domain),
                                             initial_pruning_probability,
                                             temperature_increase_rate,
                                             width_increase_resilience,
                                             max_depth,
                                             use_state_feature_hash,
                                             cache_transitions,
                                             debug_logs)
        
        def get_original_domain(self):
            return self._original_domain
        
        def get_action_space(self) -> D.T_agent[Space[D.T_event]]:
            return self._original_domain.get_action_space()
        
        def get_observation_space(self) -> D.T_agent[Space[D.T_observation]]:
            return self._original_domain.get_observation_space()
        
        def reset(self) -> D.T_agent[D.T_observation]:
            return self._domain.reset()
        
        def sample(self, memory: D.T_memory[D.T_state], action: D.T_agent[D.T_concurrency[D.T_event]]) -> \
                    EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
            return self._domain.sample(state, action)
    

    class S(Solver, Policies):
        pass

    class WidthMetaSolver(S):
        T_solver = S

        def __init__(self, solver: T_solver,
                           state_features: Callable[[D.T_state, Domain], Any],
                           initial_pruning_probability: float = 0.999,
                           temperature_increase_rate: float = 0.01,
                           width_increase_resilience: int = 10,
                           max_depth: int = 1000,
                           use_state_feature_hash: bool = False,
                           cache_transitions: bool = False,
                           debug_logs: bool = False) -> None:
            self._solver = wrl_solver(solver,
                                      lambda o: state_features(o, domain),
                                      initial_pruning_probability,
                                      temperature_increase_rate,
                                      width_increase_resilience,
                                      max_depth,
                                      use_state_feature_hash,
                                      cache_transitions,
                                      debug_logs)
        
        def reset(self) -> None:
            self._solver.reset()
        
        def solve_domain(self, domain_factory: Callable[[], D]) -> None:
            self._solver.solve(domain_factory)

except ImportError:
    sys.path = record_sys_path
    print('Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".')
    raise