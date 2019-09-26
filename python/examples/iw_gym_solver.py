import gym
import numpy as np

from airlaps.hub.domain.gym import GymWidthPlanningDomain
from airlaps.hub.solver.iw import IW
from airlaps.utils import rollout

ENV_NAME = 'MountainCar-v0'
HORIZON = 500

domain_factory = lambda: GymWidthPlanningDomain(gym_env=gym.make(ENV_NAME),
                                                discretization_factor=10,
                                                max_depth=HORIZON)
domain = domain_factory()

if IW.check_domain(domain):
    solver_factory = lambda: IW(state_features=lambda s, d: d.state_features(s),
                                use_state_feature_hash=False,
                                parallel=False, debug_logs=False)
    # solver_factory = lambda: IW(state_features=lambda s, d: s._state,
    #                             use_state_feature_hash=True,
    #                             parallel=False, debug_logs=False)
    solver = GymWidthPlanningDomain.solve_with(solver_factory, domain_factory)
    rollout(domain, solver, num_episodes=1, max_steps=HORIZON, max_framerate=30)
