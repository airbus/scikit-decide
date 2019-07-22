import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from airlaps import Memory
from airlaps.wrappers.domain.gym import GymDomain
from airlaps.wrappers.solver.baselines import BaselinesSolver
from airlaps.utils import rollout

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

    # Test solver solution on domain
    print('######################### TEST {} SOLVER #########################'.format(baselines_solver))
    rollout(GymDomain(gym.make(ENV_NAME)),
            baselines_solver,
            max_steps=1000,
            verbose=True,
            outcome_formatter=lambda o: f'Observation [{o.observation}] - transition value = {o.value}')
