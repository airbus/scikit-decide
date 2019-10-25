# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO: update to new API or remove

import gym

from airlaps import Memory
from airlaps.wrappers.domain.gym import GymDomain
from airlaps.catalog.solver.iw import IW

ENV_NAME = 'CartPole-v1'

iw_solver = IWSolver()
iw_solver.reset(lambda: GymDomain(gym.make(ENV_NAME)))

if iw_solver.check_domain():
    iteration = 0


    def on_update(*args, **kwargs):
        global iteration
        iteration += 1
        print('===> Iteration', iteration)


    # Train
    iw_solver.solve(on_update=on_update, total_timesteps=10000)

    # Test
    gym_domain = GymDomain(gym.make(ENV_NAME))
    # TODO: update with rollout util
    for i_episode in range(5):
        observation = gym_domain.reset()
        for t in range(1000):
            gym_domain.render()
            action = iw_solver.sample_action(Memory([observation]))
            outcome = gym_domain.step(action)
            observation = outcome.observation
            print(outcome)
            if outcome.termination:
                print(f'Episode finished after {t + 1} timesteps')
                break
