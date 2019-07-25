import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from airlaps.utils import rollout
from airlaps.wrappers.domain.gym import GymDomain
from airlaps.wrappers.solver.baselines import BaselinesSolver

ENV_NAME = 'CartPole-v1'

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()
if BaselinesSolver.check_domain(domain):
    solver_factory = lambda: BaselinesSolver(PPO2, MlpPolicy, learn_config={'total_timesteps': 50000}, verbose=1)
    solver = GymDomain.solve_with(solver_factory, domain_factory)
    rollout(domain, solver, num_episodes=5, max_steps=1000, max_framerate=30, outcome_formatter=None)
