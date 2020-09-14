# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
from ray.rllib.agents.ppo import PPOTrainer

from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.utils import rollout


ENV_NAME = 'CartPole-v1'

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()

# Check domain compatibility
if RayRLlib.check_domain(domain):

    # Start solving
    solver_factory = lambda: RayRLlib(PPOTrainer, train_iterations=5, config={'framework': 'torch'})
    with solver_factory() as solver:
        GymDomain.solve_with(solver, domain_factory)
        solver.save('TEMP_RLlib')  # Save results

        # Continue solving (just to demonstrate the capability to learn further)
        solver.solve(domain_factory)
        solver.save('TEMP_RLlib')  # Save updated results

        # Test solution
        rollout(domain, solver, num_episodes=1, max_steps=1000, max_framerate=30, outcome_formatter=None)

        # Restore (latest results) from scratch and re-run
        GymDomain.solve_with(solver, domain_factory, load_path='TEMP_RLlib')
        rollout(domain, solver, num_episodes=1, max_steps=1000, max_framerate=30, outcome_formatter=None)
