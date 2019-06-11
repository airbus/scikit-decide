import gym

from airlaps.wrappers.domain.gym import GymDomain

ENV_NAME = 'MsPacman-v0'  # 'CartPole-v1'

gym_domain = GymDomain(gym.make(ENV_NAME))
# TODO: update with rollout util & randomwalk solver
for i_episode in range(5):
    observation = gym_domain.reset()
    for t in range(1000):
        gym_domain.render()
        action = gym_domain.get_applicable_actions().sample()
        outcome = gym_domain.step(action)
        print(outcome)
        if outcome.termination:
            print(f'Episode finished after {t + 1} timesteps')
            break
