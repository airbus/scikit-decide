# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Any

from airlaps import Solver
from airlaps.core import TransitionOutcome, TransitionValue
from airlaps.builders.domain import Simulation, Environment, Initializable
from airlaps.builders.solver import Policies

record_sys_path = sys.path
airlaps_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if airlaps_cpp_extension_lib_path not in sys.path:
    sys.path.append(airlaps_cpp_extension_lib_path)

try:

    from __airlaps_hub_cpp import _WRLDomainFilter_ as wrl_domain_filter
    from __airlaps_hub_cpp import _WRLSolver_ as wrl_solver

    class D(Environment, Initializable, Rewards):
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
            self._domain = wrl_domain_filter(domain,
                                             state_features,
                                             initial_pruning_probability,
                                             temperature_increase_rate,
                                             width_increase_resilience,
                                             max_depth,
                                             use_state_feature_hash,
                                             cache_transitions,
                                             debug_logs)
        
        def _state_reset(self) -> D.T_state:
            return self._domain.reset()
        
        def _state_step(self, action: D.T_agent[D.T_concurrency[D.T_event]]) -> TransitionOutcome[
                D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
            return self._domain.step(action)
    

    class D(Simulation, Initializable, Rewards):
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
            self._domain = wrl_domain_filter(domain,
                                             state_features,
                                             initial_pruning_probability,
                                             temperature_increase_rate,
                                             width_increase_resilience,
                                             max_depth,
                                             use_state_feature_hash,
                                             cache_transitions,
                                             debug_logs)
        
        def _state_reset(self) -> D.T_state:
            return self._domain.reset()
        
        def _state_sample(self, memory: D.T_memory[D.T_state], action: D.T_agent[D.T_concurrency[D.T_event]]) -> \
                TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
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
                                      state_features,
                                      initial_pruning_probability,
                                      temperature_increase_rate,
                                      width_increase_resilience,
                                      max_depth,
                                      use_state_feature_hash,
                                      cache_transitions,
                                      debug_logs)
        
        def _reset(self) -> None:
            self._solver.reset()
        
        def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
            self._solver.solve(domain_factory)

except ImportError:
    sys.path = record_sys_path
    print('AIRLAPS C++ hub library not found. Please check it is installed in "airlaps/hub".')
    raise