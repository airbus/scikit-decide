# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Example 6: IW online planning with Gym environment"""

# %%
'''
Import modules.
'''

# %%
import gym
import numpy as np
from typing import Callable

from skdecide.hub.domain.gym import GymPlanningDomain, GymWidthDomain, GymDiscreteActionDomain
from skdecide.hub.solver.iw import IW
from skdecide.utils import rollout

# %%
'''
Select a [Gym environment](https://gym.openai.com/envs) and horizon parameter.
'''

# %%
ENV_NAME = 'MountainCar-v0'
HORIZON = 500

# %%
'''
Define a specific IW domain by combining Gym domain templates.
'''

# %%
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
                       discretization_factor: int = 3,
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

# %%
'''
Solve the domain with IW solver in "realtime".
'''

# %%
domain_factory = lambda: GymIWDomain(gym_env=gym.make(ENV_NAME),
                                     termination_is_goal=True,
                                     continuous_feature_fidelity=1,
                                     discretization_factor=3,
                                     max_depth=HORIZON)
domain = domain_factory()

if IW.check_domain(domain):
    solver_factory = lambda: IW(domain_factory=domain_factory,
                                state_features=lambda d, s: d.bee1_features(
                                                                np.append(
                                                                    s._state,
                                                                    s._context[3].value.reward if s._context[3] is not None else 0)),
                                use_state_feature_hash=False,
                                node_ordering=lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: a_novelty > b_novelty,
                                parallel=False, debug_logs=False)

    with solver_factory() as solver:
        GymIWDomain.solve_with(solver, domain_factory)
        rollout(domain, solver, num_episodes=1, max_steps=HORIZON, max_framerate=30,
                outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
        # value, steps = simple_rollout(domain_factory(), solver, HORIZON)
        # print('value:', value)
        # print('steps:', steps)
        print('explored:', solver.get_nb_of_explored_states())
        print('pruned:', solver.get_nb_of_pruned_states())
        filter_intermediate_scores = []
        current_score = None
        for score in solver.get_intermediate_scores():
            if current_score is None or current_score != score[2]:
                current_score = score[2]
                filter_intermediate_scores.append(score)
        print('Intermediate scores:' + str(filter_intermediate_scores))
