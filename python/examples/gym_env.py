import gym

from airlaps.utils import rollout
from airlaps.wrappers.domain.gym import GymDomain

ENV_NAME = 'MsPacman-v0'  # 'CartPole-v1'

gym_domain = GymDomain(gym.make(ENV_NAME))
rollout(gym_domain, num_episodes=5, max_steps=1000, max_framerate=30, outcome_formatter=None)
gym_domain.close()
