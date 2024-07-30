# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import gymnasium as gym

from skdecide import Value
from skdecide.hub.domain.gym import (
    GymDiscreteActionDomain,
    GymPlanningDomain,
    GymWidthDomain,
)


class D(GymPlanningDomain, GymWidthDomain, GymDiscreteActionDomain):
    pass


class GymDomainForWidthSolvers(D):
    def __init__(
        self,
        gym_env: gym.Env,
        set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
        get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None,
        gym_env_for_rendering: Optional[gym.Env] = None,
        termination_is_goal: bool = True,
        continuous_feature_fidelity: int = 5,
        discretization_factor: int = 3,
        branching_factor: int = None,
        max_depth: int = 1000,
    ) -> None:
        GymPlanningDomain.__init__(
            self,
            gym_env=gym_env,
            set_state=set_state,
            get_state=get_state,
            gym_env_for_rendering=gym_env_for_rendering,
            termination_is_goal=termination_is_goal,
            max_depth=max_depth,
        )
        GymDiscreteActionDomain.__init__(
            self,
            discretization_factor=discretization_factor,
            branching_factor=branching_factor,
        )
        GymWidthDomain.__init__(
            self, continuous_feature_fidelity=continuous_feature_fidelity
        )
        gym_env._max_episode_steps = max_depth

    def state_features(self, s):
        return self.bee2_features(s)

    def heuristic(self, s):
        return Value(cost=0)


def get_state_continuous_mountain_car(env):
    return env.unwrapped.state


def set_state_continuous_mountain_car(env, state):
    env.unwrapped.state = state
