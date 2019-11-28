# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

import numpy as np
from typing import Callable

from airlaps.hub.domain.gym import GymDomain, GymWidthDomain
from airlaps.hub.solver.stable_baselines import StableBaseline
from airlaps.hub.solver.wrl import WidthEnvironmentDomain
from airlaps.utils import rollout

ENV_NAME = 'Acrobot-v1'
HORIZON = 500

class D(GymDomain, GymWidthDomain):
    pass

class GymWRLDomain(D):
    def __init__(self, gym_env: gym.Env,
                       continuous_feature_fidelity: int = 1):
        GymDomain.__init__(self, gym_env=gym_env)
        GymWidthDomain.__init__(self, continuous_feature_fidelity=continuous_feature_fidelity)


domain_factory = lambda: WidthEnvironmentDomain(domain=GymWRLDomain(gym.make(ENV_NAME)),
                                                state_features=lambda s, d: d.bee_features(s),
                                                initial_pruning_probability=0.999,
                                                temperature_increase_rate=0.01,
                                                width_increase_resilience=10,
                                                max_depth=1000,
                                                use_state_feature_hash=False,
                                                cache_transitions=False,
                                                debug_logs=True)

domain = domain_factory()
if StableBaseline.check_domain(domain.get_original_domain()):
    solver_factory = lambda: StableBaseline(PPO2, MlpPolicy, learn_config={'total_timesteps': 50000}, verbose=1)
    solver = WidthEnvironmentDomain.solve_with(solver_factory, domain_factory)
    solver.save('TEMP_Baselines')
    rollout(domain, solver, num_episodes=1, max_steps=1000, max_framerate=30, outcome_formatter=None)

    # Restore and re-run
    solver = WidthEnvironmentDomain.solve_with(solver_factory, domain_factory, load_path='TEMP_Baselines')
    rollout(domain, solver, num_episodes=1, max_steps=1000, max_framerate=30, outcome_formatter=None)
