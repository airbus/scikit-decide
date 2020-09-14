# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Example 2: Solve a Gym environment with Reinforcement Learning"""

# %%
'''
Import modules.
'''

# %%
import gym
from stable_baselines3 import PPO

from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.utils import rollout

# %%
'''
Select a [Gym environment](https://gym.openai.com/envs) and solve it with a [Stable Baselines](https://stable-baselines3.readthedocs.io/en/master/index.html) solver wrapped in scikit-decide.
The solution is then saved (for later reuse) and assessed in rollout.
'''

# %%
ENV_NAME = 'CartPole-v1'

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()
if StableBaseline.check_domain(domain):
    solver_factory = lambda: StableBaseline(PPO, 'MlpPolicy', learn_config={'total_timesteps': 30000}, verbose=1)
    with solver_factory() as solver:
        GymDomain.solve_with(solver, domain_factory)
        solver.save('TEMP_Baselines')
        rollout(domain, solver, num_episodes=1, max_steps=1000, max_framerate=30, outcome_formatter=None)

# %%
'''
Restore saved solution and re-run rollout.
'''

# %%
with solver_factory() as solver:
    GymDomain.solve_with(solver, domain_factory, load_path='TEMP_Baselines')
    rollout(domain, solver, num_episodes=1, max_steps=1000, max_framerate=30, outcome_formatter=None)
