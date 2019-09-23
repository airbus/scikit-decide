import gym

from airlaps.hub.domain.gym import GymWidthPlanningDomain
from airlaps.hub.solver.iw import IW
from airlaps.utils import rollout

ENV_NAME = 'MountainCar-v0'
HORIZON = 30

domain_factory = lambda: GymWidthPlanningDomain(gym_env=gym.make(ENV_NAME),
                                                discretization_factor=10,
                                                max_depth=HORIZON)
domain = domain_factory()

if IW.check_domain(domain):
    solver_factory = lambda: IW(state_features=lambda s, d: s.state.round(3),
                                use_state_feature_hash=True,
                                parallel=False, debug_logs=False)
    solver = GymWidthPlanningDomain.solve_with(solver_factory, domain_factory)
    rollout(domain, solver, num_episodes=5, max_steps=HORIZON, max_framerate=30)
