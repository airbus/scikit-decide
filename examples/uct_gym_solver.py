# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Example 5: UCT online planning with Gym environment"""

# %%
'''
Import modules.
'''

# %%
import gym
import numpy as np
from typing import Callable

from skdecide.hub.domain.gym import DeterministicGymDomain, GymDiscreteActionDomain
from skdecide.hub.solver.mcts import UCT
from skdecide.utils import rollout

# %%
'''
Select a [Gym environment](https://gym.openai.com/envs) and horizon parameter.
'''

# %%
ENV_NAME = 'CartPole-v0'
HORIZON = 200

# %%
'''
Define a specific UCT domain by combining Gym domain templates.
'''

# %%
class D(DeterministicGymDomain, GymDiscreteActionDomain):
    pass


class GymUCTDomain(D):
    """This class wraps a cost-based deterministic OpenAI Gym environment as a domain
        usable by a UCT planner

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
                       get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None,
                       discretization_factor: int = 3,
                       branching_factor: int = None,
                       max_depth: int = 50) -> None:
        """Initialize GymUCTDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        set_state: Function to call to set the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        get_state: Function to call to get the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        discretization_factor: Number of discretized action variable values per continuous action variable
        branching_factor: if not None, sample branching_factor actions from the resulting list of discretized actions
        max_depth: maximum depth of states to explore from the initial state
        """
        DeterministicGymDomain.__init__(self,
                                        gym_env=gym_env,
                                        set_state=set_state,
                                        get_state=get_state)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        gym_env._max_episode_steps = max_depth

# %%
'''
Solve the domain with UCT solver in "realtime".
'''

# %%
domain_factory = lambda: GymUCTDomain(gym_env=gym.make(ENV_NAME),
                                      discretization_factor=3,
                                      max_depth=HORIZON)
domain = domain_factory()

if UCT.check_domain(domain):
    solver_factory = lambda: UCT(domain_factory=domain_factory,
                                 time_budget=200,  # 200 ms,
                                 rollout_budget=100,
                                 transition_mode=UCT.Options.TransitionMode.Sample,
                                 continuous_planning=True,
                                 parallel=False, debug_logs=False)

    with solver_factory() as solver:
        GymUCTDomain.solve_with(solver, domain_factory)
        rollout(domain, solver, num_episodes=1, max_steps=HORIZON, max_framerate=30,
                outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
