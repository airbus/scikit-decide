# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import numpy as np
from typing import Callable

from airlaps.hub.domain.gym import GymPlanningDomain, GymWidthDomain, GymDiscreteActionDomain
from airlaps.hub.solver.iw import IW
# from airlaps.hub.solver.riw import RIW
from airlaps.utils import rollout

ENV_NAME = 'MountainCar-v0'
HORIZON = 200


class D(GymPlanningDomain, GymWidthDomain, GymDiscreteActionDomain):
    pass


class GymIWDomain(D):
    """This class wraps a cost-based deterministic OpenAI Gym environment as a domain
        usable by a width-based planner

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
                       get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None,
                       termination_is_goal: bool = True,
                       continuous_feature_fidelity: int = 1,
                       discretization_factor: int = 10,
                       branching_factor: int = None,
                       max_depth: int = 50) -> None:
        """Initialize GymIWDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        set_state: Function to call to set the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        get_state: Function to call to get the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        termination_is_goal: True if the termination condition is a goal (and not a dead-end)
        continuous_feature_fidelity: Number of integers to represent a continuous feature
                                     in the interval-based feature abstraction (higher is more precise)
        discretization_factor: Number of discretized action variable values per continuous action variable
        branching_factor: if not None, sample branching_factor actions from the resulting list of discretized actions
        max_depth: maximum depth of states to explore from the initial state
        """
        GymPlanningDomain.__init__(self,
                                   gym_env=gym_env,
                                   set_state=set_state,
                                   get_state=get_state,
                                   termination_is_goal=termination_is_goal,
                                   max_depth=max_depth)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        GymWidthDomain.__init__(self, continuous_feature_fidelity=continuous_feature_fidelity)
        gym_env._max_episode_steps = max_depth


domain_factory = lambda: GymIWDomain(gym_env=gym.make(ENV_NAME),
                                     termination_is_goal=True,
                                     continuous_feature_fidelity=2,
                                     discretization_factor=3,
                                     max_depth=HORIZON)
domain = domain_factory()

if IW.check_domain(domain):
    solver_factory = lambda: IW(state_features=lambda s, d: d.state_features(s),
                                use_state_feature_hash=False,
                                node_ordering=lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: a_novelty > b_novelty,
                                # node_ordering=lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: a_depth < b_depth,
                                # node_ordering=lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: True if a_novelty > b_novelty else False if a_novelty < b_novelty else a_depth > b_depth,
                                # node_ordering=lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: True if a_novelty > b_novelty else False if a_novelty < b_novelty else a_gscore > b_gscore,
                                # node_ordering=lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: True if a_gscore > b_gscore else False if a_gscore < b_gscore else a_novelty > b_novelty,
                                parallel=False, debug_logs=False)

    # solver_factory = lambda: IW(state_features=lambda s, d: s._state,
    #                             use_state_feature_hash=True,
    #                             # node_ordering=lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: a_depth < b_depth,
    #                             node_ordering=lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: True if a_novelty > b_novelty else False if a_novelty < b_novelty else a_gscore > b_gscore,
    #                             parallel=False, debug_logs=False)

    solver = GymIWDomain.solve_with(solver_factory, domain_factory)
    rollout(domain, solver, num_episodes=1, max_steps=HORIZON, max_framerate=30,
            outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
