# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import numpy as np
from typing import Callable

from airlaps.hub.domain.gym import DeterministicInitializedGymDomain, GymWidthDomain, GymDiscreteActionDomain
from airlaps.hub.solver.riw import RIW
from airlaps.utils import rollout

ENV_NAME = 'CartPole-v0'
HORIZON = 200

gym_env = gym.make(ENV_NAME)
gym_env._max_episode_steps = HORIZON


class D(DeterministicInitializedGymDomain, GymWidthDomain, GymDiscreteActionDomain):
    pass


class GymRIWDomain(D):
    """This class wraps a cost-based deterministic OpenAI Gym environment as a domain
        usable by a width-based planner

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
                       get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None,
                       continuous_feature_fidelity: int = 1,
                       discretization_factor: int = 10,
                       branching_factor: int = None) -> None:
        """Initialize GymRIWDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        set_state: Function to call to set the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        get_state: Function to call to get the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        continuous_feature_fidelity: Number of integers to represent a continuous feature
                                     in the interval-based feature abstraction (higher is more precise)
        discretization_factor: Number of discretized action variable values per continuous action variable
        branching_factor: if not None, sample branching_factor actions from the resulting list of discretized actions
        max_depth: maximum depth of states to explore from the initial state
        """
        DeterministicInitializedGymDomain.__init__(self,
                                                   gym_env=gym_env,
                                                   set_state=set_state,
                                                   get_state=get_state)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        GymWidthDomain.__init__(self, continuous_feature_fidelity=continuous_feature_fidelity)


domain_factory = lambda: GymRIWDomain(gym_env=gym_env,
                                      continuous_feature_fidelity=2,
                                      discretization_factor=3)
domain = domain_factory()

if RIW.check_domain(domain):
    solver_factory = lambda: RIW(state_features=lambda s, d: d.state_features(s),
                                 use_state_feature_hash=False,
                                 use_simulation_domain=False,
                                 time_budget=200,
                                 rollout_budget=1000,
                                 max_depth=HORIZON-1,
                                 max_cost=10,
                                 exploration=0.25,
                                 parallel=False,
                                 debug_logs=False)
    solver = GymRIWDomain.solve_with(solver_factory, domain_factory)
    initial_state = solver._domain.reset()
    rollout(domain, solver, from_memory=initial_state, num_episodes=1, max_steps=HORIZON-1, max_framerate=30, outcome_formatter=None, action_formatter=None)
