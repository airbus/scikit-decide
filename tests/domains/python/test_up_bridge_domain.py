# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
from skdecide.hub.domain.up import UPDomain
from skdecide.hub.solver.lazy_astar import LazyAstar

import unified_planning
from unified_planning.shortcuts import (
    Fluent,
    InstantaneousAction,
    Not,
)


def test_up_bridge_domain():
    noexcept = True

    try:
        x = Fluent("x")
        y = Fluent("y")

        a = InstantaneousAction("a")
        a.add_precondition(Not(x))
        a.add_effect(x, True)

        b = InstantaneousAction("b")
        b.add_precondition(Not(y))
        b.add_effect(y, True)

        c = InstantaneousAction("c")
        c.add_precondition(y)
        c.add_effect(x, True)

        problem = unified_planning.model.Problem("simple_with_costs")

        problem.add_fluent(x)
        problem.add_fluent(y)

        problem.add_action(a)
        problem.add_action(b)
        problem.add_action(c)

        problem.set_initial_value(x, False)
        problem.set_initial_value(y, False)

        problem.add_goal(x)

        problem.add_quality_metric(
            unified_planning.model.metrics.MinimizeActionCosts({a: 10, b: 1, c: 1})
        )

        domain_factory = lambda: UPDomain(problem)
        domain = domain_factory()
        action_space = domain.get_action_space()
        observation_space = domain.get_observation_space()

        with LazyAstar() as solver:
            UPDomain.solve_with(solver, domain_factory)
            s = domain.get_initial_state()
            step = 0
            p = []
            while not domain.is_goal(s) and step < 10:
                p.append(solver.get_next_action(s))
                s = domain.get_next_state(s, p[-1])
                step += 1
    except Exception as e:
        print(e)
        noexcept = False
    assert (
        LazyAstar.check_domain(domain)
        and len(action_space._elements) == 3
        and all(
            isinstance(s, gym.spaces.Discrete) and s.n == 2
            for s in observation_space._gym_space.spaces.values()
        )
        and noexcept
        and step == 2
        and p[0].up_action == b
        and p[1].up_action == c
    )
