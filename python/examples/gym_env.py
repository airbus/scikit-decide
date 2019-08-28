import gym
from airlaps import hub
from airlaps.utils import rollout

GymDomain = hub.load('GymDomain', folder='hub/domain/gym')


ENV_NAME = 'CartPole-v1'  # 'MsPacman-v4'

gym_domain = GymDomain(gym.make(ENV_NAME))
rollout(gym_domain, num_episodes=5, max_steps=1000, max_framerate=30, outcome_formatter=None)
gym_domain.close()
