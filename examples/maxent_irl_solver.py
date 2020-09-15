import gym

from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.solver.maxent_irl import MaxentIRL  # maximum entropy inverse reinforcement learning
from skdecide.utils import rollout

ENV_NAME = 'MountainCar-v0'

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()
print('===>', domain.get_action_space().unwrapped())
if MaxentIRL.check_domain(domain):
    solver_factory = lambda: MaxentIRL(n_states=400, n_actions=3, one_feature=20,
                                       expert_trajectories="expert_mountain.npy", n_epochs=10000)
    with solver_factory() as solver:
        GymDomain.solve_with(solver, domain_factory)
        rollout(domain, solver, num_episodes=5, max_steps=500, max_framerate=30, outcome_formatter=None,
                action_formatter=None)

domain.close()
