# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
# from stable_baselines3 import PPO
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG

import numpy as np
from typing import Callable, Any

from skdecide.core import EnvironmentOutcome, TransitionValue
from skdecide.hub.domain.gym import GymDomain, GymWidthDomain
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.wrl import WidthEnvironmentDomain
from skdecide.utils import rollout

ENV_NAME = 'MountainCarContinuous-v0'
HORIZON = 200

class D(GymDomain, GymWidthDomain):
    pass

class GymWRLDomain(D):
    def __init__(self, gym_env: gym.Env,
                       continuous_feature_fidelity: int = 1):
        GymDomain.__init__(self, gym_env=gym_env)
        GymWidthDomain.__init__(self, continuous_feature_fidelity=continuous_feature_fidelity)
    
    def _state_reset(self) -> D.T_state:
        GymWidthDomain._reset_features(self)
        return GymDomain._state_reset(self)


proxy_domain_factory = lambda: WidthEnvironmentDomain(domain=GymWRLDomain(gym.make(ENV_NAME)),
                                                      state_features=lambda d, s: d.bee1_features(s),
                                                      initial_pruning_probability=0.999,
                                                      temperature_increase_rate=0.001,
                                                      width_increase_resilience=10,
                                                      max_depth=HORIZON,
                                                      use_state_feature_hash=False,
                                                      cache_transitions=False,
                                                      debug_logs=False)
proxy_domain = proxy_domain_factory()

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()

if StableBaseline.check_domain(proxy_domain.get_original_domain()):
    # the noise objects for DDPG
    n_actions = domain.get_action_space().shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    # solver_factory = lambda: StableBaseline(PPO, 'MlpPolicy', learn_config={'total_timesteps': 50000}, verbose=1)
    solver_factory = lambda: StableBaseline(DDPG, 'MlpPolicy', learn_config={'total_timesteps': 50000}, action_noise=action_noise, verbose=1)
    solver = WidthEnvironmentDomain.solve_with(solver_factory, proxy_domain_factory)
    solver.save('TEMP_Baselines')
    rollout(proxy_domain, solver, num_episodes=1, max_steps=HORIZON, max_framerate=30, outcome_formatter=None)

    # Restore and re-run
    solver = GymDomain.solve_with(solver_factory, domain_factory, load_path='TEMP_Baselines')
    rollout(domain, solver, num_episodes=1, max_steps=HORIZON, max_framerate=30, outcome_formatter=None)
