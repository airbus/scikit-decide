import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from airlaps import Memory
from airlaps.wrappers.domain.gym import GymDomain
from airlaps.wrappers.solver.baselines import BaselinesSolver

ENV_NAME = 'CartPole-v1'

baselines_solver = BaselinesSolver(PPO2, MlpPolicy, verbose=1)
baselines_solver.reset(lambda: GymDomain(gym.make(ENV_NAME)))

if baselines_solver.check_domain():
    iteration = 0


    def on_update(*args, **kwargs):
        global iteration
        iteration += 1
        print('===> Iteration', iteration)


    # Train
    baselines_solver.solve(on_update=on_update, total_timesteps=10000)

    # Test
    gym_domain = GymDomain(gym.make(ENV_NAME))
    # TODO: update with rollout util
    for i_episode in range(5):
        observation = gym_domain.reset()
        for t in range(1000):
            gym_domain.render()
            action = baselines_solver.sample_action(Memory([observation]))
            outcome = gym_domain.step(action)
            observation = outcome.observation
            print(outcome)
            if outcome.termination:
                print(f'Episode finished after {t + 1} timesteps')
                break
