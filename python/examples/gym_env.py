import gym

from airlaps.hub.domain.gym import GymDomain
from airlaps.utils import rollout

ENV_NAME = 'CartPole-v1'  # 'MsPacman-v4'

gym_domain = GymDomain(gym.make(ENV_NAME))
rollout(gym_domain, num_episodes=5, max_steps=1000, max_framerate=30, outcome_formatter=None)
gym_domain.close()
