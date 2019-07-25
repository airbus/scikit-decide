# TODO: update to new API or remove (quite redundant with other examples)

import random
from enum import Enum
from typing import Dict

from airlaps import RLDomain, Distribution, TransitionOutcome, TransitionValue, ImplicitDistribution, nocopy
from airlaps.dataclasses import dataclass, \
    replace  # TODO: replace 'airlaps.dataclasses' by 'dataclasses' once transitioned to Python 3.7


# @nocopy avoids copy to optimize memory since State is recursively immutable
# (i.e. copy.copy(state) or copy.deepcopy(state) will actually return state instead of a copy)
# @dataclass: frozen=True makes State immutable (avoiding potential side effects of modifying it directly) and hashable
# if all his attributes are (useful e.g. for putting it in a Set or as Dict keys)
@nocopy
@dataclass(frozen=True)
class State:
    x: int
    y: int


class Actions(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


T_state = State  # Type of states
T_observation = Dict[str, float]  # Type of observations
T_event = Actions  # Type of events
T_value = float  # Type of transition values (rewards or costs)
T_info = None  # Type of additional information given as part of an environment outcome


class NoisyGridEnvDomain(RLDomain):

    def __init__(self, n_rows: int, n_cols: int) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols

    def _step(self, event: T_event) -> TransitionOutcome[T_state, T_value, T_info]:
        move_reward = -1  # default penalty for moving the robot (e.g. because of energy consumption)

        if event == Actions.up and self.state.y > 0:
            self.state = replace(self.state, y=self.state.y - 1)
        elif event == Actions.down and self.state.y < self.n_rows - 1:
            self.state = replace(self.state, y=self.state.y + 1)
        elif event == Actions.left and self.state.x > 0:
            self.state = replace(self.state, x=self.state.x - 1)
        elif event == Actions.right and self.state.x < self.n_cols - 1:
            self.state = replace(self.state, x=self.state.x + 1)
        else:
            move_reward = 0  # reset penalty if robot actually did not move

        goal_reward = 10 if self._is_in_top_left_corner() else 0
        total_reward = move_reward + goal_reward

        return TransitionOutcome(self.state, TransitionValue(reward=total_reward), self._is_in_top_left_corner())

    def _reset(self) -> T_state:
        self.state = State(x=random.randrange(self.n_cols), y=random.randrange(self.n_rows))
        return self.state

    def get_observation_distribution(self, state: T_state, event: T_event) -> Distribution[T_observation]:
        def noisy_sampling():
            x_noise = random.random()  # noise in range [0, 1[
            y_noise = random.random()  # noise in range [0, 1[
            return {'x': state.x + x_noise, 'y': state.y + y_noise}

        return ImplicitDistribution(noisy_sampling)

    def _is_in_top_left_corner(self):
        return self.state.x == 0 and self.state.y == 0


# Test the domain
gridEnvDomain = NoisyGridEnvDomain(n_rows=5, n_cols=10)
print('Initial observation:', gridEnvDomain.reset())
while True:
    action = input('Please enter action (%s): ' % ', '.join([a.name for a in Actions]))
    try:
        print(gridEnvDomain.step(getattr(Actions, action)))
    except AttributeError:
        print('This action is unknown')
