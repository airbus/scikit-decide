# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
# from stable_baselines import PPO2
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

import numpy as np
from typing import Callable, Any

from airlaps.core import EnvironmentOutcome, TransitionValue
from airlaps.hub.domain.gym import GymDomain, GymDomainHashable, GymWidthDomain
from airlaps.hub.solver.stable_baselines import StableBaseline
from airlaps.hub.solver.wrl import WidthEnvironmentDomain
from airlaps.utils import rollout

ENV_NAME = 'MountainCarContinuous-v0'
HORIZON = 200

class D(GymDomainHashable, GymWidthDomain):
    pass

class GymWRLDomain(D):
    def __init__(self, gym_env: gym.Env,
                       continuous_feature_fidelity: int = 1):
        GymDomainHashable.__init__(self, gym_env=gym_env)
        GymWidthDomain.__init__(self, continuous_feature_fidelity=continuous_feature_fidelity)


class MyWidthEnvironmentDomain(WidthEnvironmentDomain):
    def __init__(self, domain: GymWRLDomain,
                        state_features: Callable[[D.T_observation, GymWRLDomain], Any],
                        initial_pruning_probability: float = 0.999,
                        temperature_increase_rate: float = 0.01,
                        width_increase_resilience: int = 10,
                        max_depth: int = 1000,
                        use_state_feature_hash: bool = False,
                        cache_transitions: bool = False,
                        debug_logs: bool = False) -> None:
        super().__init__(domain, state_features, initial_pruning_probability,
                         temperature_increase_rate, width_increase_resilience,
                         max_depth, use_state_feature_hash, cache_transitions, debug_logs)
    
    def reset(self) -> D.T_agent[D.T_observation]:
        return super().reset()._state
    
    def step(self, action: D.T_agent[D.T_concurrency[D.T_event]]) -> EnvironmentOutcome[
                D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
        return super().step(action)._state


proxy_domain_factory = lambda: MyWidthEnvironmentDomain(domain=GymWRLDomain(gym.make(ENV_NAME)),
                                                      state_features=lambda s, d: d.bee_features(s),
                                                      initial_pruning_probability=0.999,
                                                      temperature_increase_rate=0.001,
                                                      width_increase_resilience=10,
                                                      max_depth=HORIZON,
                                                      use_state_feature_hash=False,
                                                      cache_transitions=True,
                                                      debug_logs=True)
proxy_domain = proxy_domain_factory()

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()

if StableBaseline.check_domain(proxy_domain.get_original_domain()):
    # the noise objects for DDPG
    n_actions = domain.get_action_space().shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    # solver_factory = lambda: StableBaseline(PPO2, MlpPolicy, learn_config={'total_timesteps': 50000}, verbose=1)
    solver_factory = lambda: StableBaseline(DDPG, MlpPolicy, learn_config={'total_timesteps': 50000}, param_noise=param_noise, action_noise=action_noise, verbose=1)
    solver = WidthEnvironmentDomain.solve_with(solver_factory, proxy_domain_factory)
    solver.save('TEMP_Baselines')
    rollout(proxy_domain, solver, num_episodes=1, max_steps=HORIZON, max_framerate=30, outcome_formatter=None)

    # Restore and re-run
    solver = GymDomain.solve_with(solver_factory, domain_factory, load_path='TEMP_Baselines')
    rollout(domain, solver, num_episodes=1, max_steps=HORIZON, max_framerate=30, outcome_formatter=None)
