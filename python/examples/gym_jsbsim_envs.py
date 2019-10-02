import gym
import gym_jsbsim
import numpy as np

from airlaps.hub.domain.gym import GymWidthPlanningDomain
from airlaps.hub.solver.iw import IW
from airlaps.utils import rollout

ENV_NAME = 'GymJsbsim-HeadingControlTask-v0'
# ENV_NAME = 'GymJsbsim-TaxiControlTask-v0'
HORIZON = 200

gym_env = gym.make(ENV_NAME)
gym_env._max_episode_steps = HORIZON

domain_factory = lambda: GymWidthPlanningDomain(gym_env=gym_env,
                                                set_state=lambda e, s: e.set_full_state(s),
                                                get_state=lambda e: e.get_full_state(),
                                                termination_is_goal=False,
                                                discretization_factor=3,
                                                max_depth=HORIZON)
domain = domain_factory()

if IW.check_domain(domain):
    solver_factory = lambda: IW(state_features=lambda s, d: d.state_features(s),
                                use_state_feature_hash=False,
                                parallel=False, debug_logs=True)
    # solver_factory = lambda: IW(state_features=lambda s, d: s._state,
    #                             use_state_feature_hash=True,
    #                             parallel=False, debug_logs=False)
    solver = GymWidthPlanningDomain.solve_with(solver_factory, domain_factory)
    rollout(domain, solver, num_episodes=1, max_steps=HORIZON, max_framerate=30, outcome_formatter=None, action_formatter=None)
