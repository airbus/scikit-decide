# TODO: remove this example if redundant with grid_solve_planning_rl.py?

import math
from enum import Enum
from typing import NamedTuple

from gym.spaces import MultiDiscrete

from airlaps import Memory, Space, TransitionValue, T_value
from airlaps.builders.domain import UnrestrictedActionDomain
from airlaps.domains import DeterministicPlanningDomain
from airlaps.wrappers.space import gym


class State(NamedTuple):
    x: int
    y: int


class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3
    up_right = 4
    up_left = 5
    down_right = 6
    down_left = 7


T_state = State  # Type of states
T_observation = State  # Type of observations
T_event = Action


class MyDomain(DeterministicPlanningDomain, UnrestrictedActionDomain):
    def __init__(self, n_x=10, n_y=10):
        self.n_x = n_x
        self.n_y = n_y

    def get_next_state(self, memory: Memory[T_state], event: T_event) -> T_state:
        last = memory[-1]
        if event == Action.up:
            return State(min(last.x + 1, self.n_x - 1),
                         last.y)
        if event == Action.down:
            return State(max(last.x - 1, 0), last.y)
        if event == Action.left:
            return State(last.x, max(0, last.y - 1))
        if event == Action.right:
            return State(last.x, min(self.n_y - 1, last.y + 1))
        if event == Action.up_left:
            return State(min(last.x + 1, self.n_x - 1), max(0, last.y - 1))
        if event == Action.down_left:
            return State(max(last.x - 1, 0), max(0, last.y - 1))
        if event == Action.up_right:
            return State(min(last.x + 1, self.n_x - 1), min(self.n_y - 1, last.y + 1))
        if event == Action.down_right:
            return State(max(last.x - 1, 0), min(self.n_y - 1, last.y + 1))

    def get_transition_value(self, memory: Memory[T_state], event: T_event, next_state: T_state) -> TransitionValue[
            T_value]:
        l = memory[-1]
        cost = 10 if l.x == next_state.x and l.y == next_state.y else math.sqrt(
            (l.x - next_state.x) ** 2 + (l.y - next_state.y) ** 2)
        return TransitionValue(cost=cost)

    def is_terminal(self, state: T_state) -> bool:
        return self.is_goal(state)

    def _get_action_space_(self) -> Space[T_event]:
        return gym.EnumSpace(T_event)

    def _get_goals_(self) -> Space[T_observation]:
        return gym.ListSpace([State(self.n_x - 1, self.n_y - 1)])

    def _get_initial_state_(self) -> T_state:
        return State(x=0, y=0)

    def _get_observation_space_(self) -> Space[T_observation]:
        return gym.GymSpace(gym_space=MultiDiscrete([self.n_x, self.n_y]))


def random_walk():
    gym_domain = MyDomain(10, 10)
    for i_episode in range(5):
        observation = gym_domain.reset()
        for t in range(1000):
            action = gym_domain.get_applicable_actions().sample()
            outcome = gym_domain.step(action)
            print(outcome)
            if outcome.termination:
                print(f'Episode finished after {t + 1} timesteps')
                break


def solve_by_astar():
    from airlaps.catalog.solver import lazyastar

    def heuristic(state_1, state_2):
        return abs(state_1.x - state_2.x) + abs(state_1.y - state_2.y)

    def heuristic_l2(state_1, state_2):
        return math.sqrt((state_1.x - state_2.x) ** 2 + (state_1.y - state_2.y) ** 2)

    grid_domain = MyDomain(1000, 1000)
    lazyastar_solver = lazyastar.LazyAstar(heuristic=lambda x, d:
    heuristic_l2(x, State(d.n_x - 1, d.n_y - 1)),
                                           weight=1.)
    lazyastar_solver.reset(lambda: grid_domain)
    # Solve
    lazyastar_solver.solve(from_observation=Memory([State(0, 0)],
                                                   maxlen=1),
                           verbose=True,
                           render=False)

    for i_episode in range(1):
        state = State(0, 0)
        memory = Memory(maxlen=1)  # Markovian memory (only keeps last state)
        for t in range(3000):
            print(state)
            memory.append(state)
            action = lazyastar_solver.get_next_action(memory)
            state = grid_domain.get_next_state(memory, action)
            value = grid_domain.get_transition_value(memory, action, state)
            print(f'ESAD cost: {value.cost:.2f} NM')
            if grid_domain.is_terminal(state):
                print(f'Episode {i_episode + 1} finished after {t + 1} timesteps.')
                break
        print(
            f'The goal was{"" if state in grid_domain.get_goals() else " not"} reached in episode {i_episode + 1}.')


def solve_with_rl():
    import airlaps.wrappers.solver.baselines as baselines
    from stable_baselines import PPO2
    from stable_baselines.common.policies import MlpPolicy

    grid_domain = MyDomain(100, 100)

    solver = baselines.BaselinesSolver(PPO2, baselines_policy=MlpPolicy)
    solver.reset(MyDomain)
    assert solver.check_domain()
    # Solve
    solver.solve(total_timesteps=30000)
    for i_episode in range(1):
        state = State(0, 0)
        memory = Memory(maxlen=1)  # Markovian memory (only keeps last state)
        for t in range(3000):
            print(state)
            memory.append(state)
            action = solver.sample_action(memory)
            if action == 0:
                action = Action.up
            if action == 1:
                action = Action.down
            if action == 2:
                action = Action.left
            if action == 3:
                action = Action.right
            if action == 4:
                action = Action.up_right
            if action == 5:
                action = Action.up_left
            if action == 6:
                action = Action.down_right
            if action == 7:
                action = Action.down_left
            state = grid_domain.get_next_state(memory, action)
            value = grid_domain.get_transition_value(memory, action, state)
            print(f'ESAD cost: {value.cost:.2f} NM')
            if grid_domain.is_terminal(state):
                print(f'Episode {i_episode + 1} finished after {t + 1} timesteps.')
                break
        print(
            f'The goal was{"" if state in grid_domain.get_goals() else " not"} reached in episode {i_episode + 1}.')


if __name__ == '__main__':
    solve_by_astar()
    # TODO : this doesn't work yet
    solve_with_rl()
