# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import sys


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_up_bridge_solver_classic():
    noexcept = True

    try:
        from skdecide.hub.domain.up import UPDomain, SkUPAction
        from skdecide.hub.solver.up import UPSolver

        import unified_planning
        from unified_planning.shortcuts import UserType, BoolType, OneshotPlanner
        from unified_planning.plans import ActionInstance

        Location = UserType("Location")
        robot_at = unified_planning.model.Fluent("robot_at", BoolType(), l=Location)
        connected = unified_planning.model.Fluent(
            "connected", BoolType(), l_from=Location, l_to=Location
        )

        move = unified_planning.model.InstantaneousAction(
            "move", l_from=Location, l_to=Location
        )
        l_from = move.parameter("l_from")
        l_to = move.parameter("l_to")
        move.add_precondition(connected(l_from, l_to))
        move.add_precondition(robot_at(l_from))
        move.add_effect(robot_at(l_from), False)
        move.add_effect(robot_at(l_to), True)

        problem = unified_planning.model.Problem("robot")
        problem.add_fluent(robot_at, default_initial_value=False)
        problem.add_fluent(connected, default_initial_value=False)
        problem.add_action(move)

        NLOC = 10
        locations = [
            unified_planning.model.Object("l%s" % i, Location) for i in range(NLOC)
        ]
        problem.add_objects(locations)

        problem.set_initial_value(robot_at(locations[0]), True)
        for i in range(NLOC - 1):
            problem.set_initial_value(connected(locations[i], locations[i + 1]), True)

        problem.add_goal(robot_at(locations[-1]))

        domain_factory = lambda: UPDomain(problem)
        domain = domain_factory()

        with UPSolver(operation_mode=OneshotPlanner, name="pyperplan") as solver:
            UPDomain.solve_with(solver, domain_factory)
            s = domain.get_initial_state()
            step = 0
            p = []
            while not domain.is_goal(s) and step < 10:
                p.append(solver.get_next_action(s))
                s = domain.get_next_state(s, p[-1])
                step += 1
            ep = [
                SkUPAction(ActionInstance(move, (locations[i], locations[i + 1])))
                for i in range(9)
            ]
    except Exception as e:
        print(e)
        noexcept = False
    assert noexcept and UPSolver.check_domain(domain) and step == 9 and p == ep


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_up_bridge_solver_numeric():
    noexcept = True

    try:
        from skdecide.hub.domain.up import UPDomain
        from skdecide.hub.solver.up import UPSolver

        import unified_planning
        from unified_planning.shortcuts import (
            OneshotPlanner,
            Fluent,
            InstantaneousAction,
            Not,
        )

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

        with UPSolver(
            operation_mode=OneshotPlanner,
            name="fast-downward-opt",
        ) as solver:
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
        noexcept
        and UPSolver.check_domain(domain)
        and step == 2
        and p[0].up_action == b
        and p[1].up_action == c
    )
