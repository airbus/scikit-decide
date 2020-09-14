# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Example 3: Solve a Gym environment with Cartesian Genetic Programming"""

# %%
'''
Import modules.
'''

# %%
import gym

from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.solver.cgp import CGP  # Cartesian Genetic Programming
from skdecide.utils import rollout

# %%
'''
Select a [Gym environment](https://gym.openai.com/envs) and solve it with Cartesian Genetic Programming in scikit-decide.
The solution is then assessed in rollout.
'''

# %%
ENV_NAME = 'MountainCarContinuous-v0'

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()
if CGP.check_domain(domain):
    solver_factory = lambda: CGP('TEMP_CGP', n_it=25)
    with solver_factory() as solver:
        GymDomain.solve_with(solver, domain_factory)
        rollout(domain, solver, num_episodes=5, max_steps=1000, max_framerate=30, outcome_formatter=None)
