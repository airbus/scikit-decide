import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from airlaps.hub.domain.gym import GymDomain
from airlaps.hub.solver.stable_baselines import StableBaseline
from airlaps.utils import rollout


ENV_NAME = 'CartPole-v1'

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()
if StableBaseline.check_domain(domain):
    solver_factory = lambda: StableBaseline(PPO2, MlpPolicy, learn_config={'total_timesteps': 50000}, verbose=1)
    solver = GymDomain.solve_with(solver_factory, domain_factory)
    rollout(domain, solver, num_episodes=5, max_steps=1000, max_framerate=30, outcome_formatter=None)
