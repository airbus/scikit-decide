# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import NamedTuple, Optional

import pytest
from stable_baselines3 import PPO

from skdecide import DeterministicPlanningDomain, ImplicitSpace, Space, Value
from skdecide.builders.domain import UnrestrictedActions
from skdecide.hub.space.gym import EnumSpace, MultiDiscreteSpace
from skdecide.utils import load_registered_solver


# Must be defined outside the grid_domain() fixture
# so that parallel domains can pickle it
# /!\ Is it worth defining the domain as a fixture?
class State(NamedTuple):
    x: int
    y: int
    s: int  # step => to make the domain cycle-free for algorithms like AO*


# Must be defined outside the grid_domain() fixture
# so that parallel domains can pickle it
# /!\ Is it worth defining the domain as a fixture?
class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


class D(DeterministicPlanningDomain, UnrestrictedActions):
    T_state = State  # Type of states
    T_observation = T_state  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class GridDomain(D):
    def __init__(self, num_cols=10, num_rows=10):
        self.num_cols = num_cols
        self.num_rows = num_rows

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:

        if action == Action.left:
            next_state = State(max(memory.x - 1, 0), memory.y, memory.s + 1)
        if action == Action.right:
            next_state = State(
                min(memory.x + 1, self.num_cols - 1), memory.y, memory.s + 1
            )
        if action == Action.up:
            next_state = State(memory.x, max(memory.y - 1, 0), memory.s + 1)
        if action == Action.down:
            next_state = State(
                memory.x, min(memory.y + 1, self.num_rows - 1), memory.s + 1
            )

        return next_state

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:

        if next_state.x == memory.x and next_state.y == memory.y:
            cost = 2  # big penalty when hitting a wall
        else:
            cost = abs(next_state.x - memory.x) + abs(
                next_state.y - memory.y
            )  # every move costs 1

        return Value(cost=cost)

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        return self._is_goal(state) or state.s >= 100

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return EnumSpace(Action)

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return ImplicitSpace(
            lambda state: state.x == (self.num_cols - 1)
            and state.y == (self.num_rows - 1)
        )

    def _get_initial_state_(self) -> D.T_state:
        return State(x=0, y=0, s=0)

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return MultiDiscreteSpace([self.num_cols, self.num_rows, 100])


# FIXTURES


@pytest.fixture(
    params=[
        {"entry": "LazyAstar", "config": {"verbose": False}, "optimal": True},
        {
            "entry": "StableBaseline",
            "config": {
                "baselines_policy": "MlpPolicy",
                "learn_config": {"total_timesteps": 10},
                "verbose": 1,
            },
            "optimal": False,
        },
    ]
)
def solver_python(request):
    return request.param


# HELPER FUNCTION


def get_plan(domain, solver):
    plan = []
    cost = 0
    observation = domain.reset()
    nb_steps = 0
    while (not domain.is_goal(observation)) and nb_steps < 20:
        plan.append(solver.sample_action(observation))
        outcome = domain.step(plan[-1])
        cost += outcome.value.cost
        observation = outcome.observation
        nb_steps += 1
    return plan, cost


def test_solve_python(solver_python):
    noexcept = True

    try:
        dom = GridDomain()
        solver_type = load_registered_solver(solver_python["entry"])
        solver_args = deepcopy(solver_python["config"])
        if solver_python["entry"] == "StableBaseline":
            solver_args["algo_class"] = PPO

        with solver_type(**solver_args) as slv:
            GridDomain.solve_with(slv)
            plan, cost = get_plan(dom, slv)
    except Exception as e:
        print(e)
        noexcept = False
    assert (
        solver_type.check_domain(dom)
        and noexcept
        and ((not solver_python["optimal"]) or (cost == 18 and len(plan) == 18))
    )
