# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import gymnasium as gym
import pytest


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_up_bridge_domain_random():
    import unified_planning
    from unified_planning.shortcuts import Fluent, InstantaneousAction, Not

    from skdecide.hub.domain.up import UPDomain

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

    domain_factory = lambda: UPDomain(problem, state_encoding="dictionary")
    domain = domain_factory()
    action_space = domain.get_action_space()
    observation_space = domain.get_observation_space()

    s = domain.get_initial_state()
    step = 0
    while not domain.is_goal(s) and step < 10:
        s = domain.get_next_state(s, domain.get_applicable_actions(s).sample())
        step += 1
    assert len(action_space._elements) == 3
    assert all(
        isinstance(s, gym.spaces.Discrete) and s.n == 2
        for s in observation_space.unwrapped().spaces.values()
    )


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_up_bridge_domain_planning():
    import unified_planning
    from unified_planning.shortcuts import Fluent, InstantaneousAction, Not

    from skdecide.hub.domain.up import UPDomain
    from skdecide.hub.solver.lazy_astar import LazyAstar

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

    domain_factory = lambda: UPDomain(problem, state_encoding="native")
    domain = domain_factory()

    with LazyAstar(
        domain_factory=domain_factory,
    ) as solver:
        UPDomain.solve_with(solver)
        s = domain.get_initial_state()
        step = 0
        p = []
        while not domain.is_goal(s) and step < 10:
            p.append(solver.get_next_action(s))
            s = domain.get_next_state(s, p[-1])
            step += 1
    assert LazyAstar.check_domain(domain)
    assert step == 2
    assert p[0].up_action == b
    assert p[1].up_action == c


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.platform == "darwin",
    reason="requires python3.10 or higher, libomp segfault on MacOS",
)
def test_up_bridge_domain_rl():
    import unified_planning
    from ray.rllib.algorithms.dqn import DQN
    from unified_planning.shortcuts import (
        GE,
        BoolType,
        Int,
        IntType,
        SimulatedEffect,
        UserType,
    )

    from skdecide.hub.domain.up import UPDomain
    from skdecide.hub.solver.ray_rllib import RayRLlib
    from skdecide.utils import rollout

    Location = UserType("Location")
    robot_at = unified_planning.model.Fluent("robot_at", BoolType(), l=Location)
    connected = unified_planning.model.Fluent(
        "connected", BoolType(), l_from=Location, l_to=Location
    )
    battery_charge = unified_planning.model.Fluent("battery_charge", IntType(0, 150))

    move = unified_planning.model.InstantaneousAction(
        "move", l_from=Location, l_to=Location
    )
    l_from = move.parameter("l_from")
    l_to = move.parameter("l_to")
    move.add_precondition(connected(l_from, l_to))
    move.add_precondition(robot_at(l_from))
    move.add_precondition(GE(battery_charge(), 10))
    move.add_effect(robot_at(l_from), False)
    move.add_effect(robot_at(l_to), True)

    def fun(_, state, actual_params):
        value = state.get_value(battery_charge()).constant_value()
        return [Int(value - 10)]

    move.set_simulated_effect(SimulatedEffect([battery_charge()], fun))

    problem = unified_planning.model.Problem("robot")
    problem.add_fluent(robot_at, default_initial_value=False)
    problem.add_fluent(connected, default_initial_value=False)
    problem.add_fluent(battery_charge, default_initial_value=False)
    problem.add_action(move)

    NLOC = 10
    locations = [
        unified_planning.model.Object("l%s" % i, Location) for i in range(NLOC)
    ]
    problem.add_objects(locations)

    problem.set_initial_value(robot_at(locations[0]), True)
    for i in range(NLOC - 1):
        problem.set_initial_value(connected(locations[i], locations[i + 1]), True)
    problem.set_initial_value(battery_charge(), 150)

    problem.add_goal(robot_at(locations[-1]))

    problem.add_quality_metric(
        unified_planning.model.metrics.MinimizeActionCosts({move: 1})
    )

    domain_factory = lambda: UPDomain(
        problem,
        state_encoding="vector",
        action_encoding="int",
    )
    domain = domain_factory()
    action_space = domain.get_action_space()
    observation_space = domain.get_observation_space()

    with RayRLlib(
        domain_factory=domain_factory,
        algo_class=DQN,
        train_iterations=1,
    ) as solver:
        UPDomain.solve_with(solver)
        rollout(
            domain_factory(),
            solver,
            num_episodes=1,
            max_steps=100,
            max_framerate=30,
            outcome_formatter=None,
        )
    assert RayRLlib.check_domain(domain)
    assert isinstance(action_space.unwrapped(), gym.spaces.Discrete)
    assert action_space.unwrapped().n == 9
    assert isinstance(observation_space.unwrapped(), gym.spaces.Box)
    assert (
        len(observation_space.unwrapped().low)
        == len(observation_space.unwrapped().high)
        == 11
    )
    assert all(
        observation_space.unwrapped().low[i] == 0
        and observation_space.unwrapped().high[i] in {1, 150}
        for i in range(11)
    )
