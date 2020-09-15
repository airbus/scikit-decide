# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Example 1: Run a Gym environment"""

# %%
'''
Import modules.
'''

# %%
import gym

from skdecide.hub.domain.gym import GymDomain
from skdecide.utils import rollout

# %%
'''
Select a [Gym environment](https://gym.openai.com/envs) and run 5 episodes.
'''

# %%
ENV_NAME = 'CartPole-v1'  # or any other installed environment ('MsPacman-v4'...)

gym_domain = GymDomain(gym.make(ENV_NAME))
rollout(gym_domain, num_episodes=5, max_steps=1000, max_framerate=30, outcome_formatter=None)
gym_domain.close()  # optional but recommended to avoid Gym errors at the end
