# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym

from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.solver.cgp import CGP  # Cartesian Genetic Programming
from skdecide.utils import rollout


ENV_NAME = 'MountainCarContinuous-v0'
# ENV_NAME = 'DuplicatedInput-v0'

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()
print("init")
if CGP.check_domain(domain):
    print("start")
    solver_factory = lambda: CGP('TEMP', n_it=25)
    solver = GymDomain.solve_with(solver_factory, domain_factory)
    rollout(domain, solver, num_episodes=5, max_steps=1000, max_framerate=30, outcome_formatter=None)
    print("end")