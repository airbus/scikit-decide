import gym

from airlaps.hub.domain.gym import GymDomain
from airlaps.hub.solver.cgp import CGP  # Cartesian Genetic Programming
from airlaps.utils import rollout


ENV_NAME = 'MountainCarContinuous-v0'

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()
if CGP.check_domain(domain):
    solver_factory = lambda: CGP('TEMP', n_it=25)
    solver = GymDomain.solve_with(solver_factory, domain_factory)
    rollout(domain, solver, num_episodes=5, max_steps=1000, max_framerate=30, outcome_formatter=None)
