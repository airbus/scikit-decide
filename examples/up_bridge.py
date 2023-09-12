# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from skdecide.hub.domain.up import UPDomain
from skdecide.hub.solver.up import UPSolver
from skdecide.hub.solver.lazy_astar import LazyAstar

import unified_planning
from unified_planning.shortcuts import (
    UserType,
    BoolType,
    OneshotPlanner,
    Fluent,
    InstantaneousAction,
    Not,
)

from skdecide.utils import rollout

# Example 1: Solving a basic example, the same as
# https://github.com/aiplan4eu/unified-planning/blob/master/docs/notebooks/01-basic-example.ipynb

print("\n\n=== EXAMPLE 1: Solving UP's basic example using skdecide's UP solver ===\n")

## Step 1: modeling the UP problem

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
locations = [unified_planning.model.Object("l%s" % i, Location) for i in range(NLOC)]
problem.add_objects(locations)

problem.set_initial_value(robot_at(locations[0]), True)
for i in range(NLOC - 1):
    problem.set_initial_value(connected(locations[i], locations[i + 1]), True)

problem.add_goal(robot_at(locations[-1]))

## Step 2: creating the scikit-decide's UPDomain

domain_factory = lambda: UPDomain(problem)
domain = domain_factory()

## Step 3: solving the UP problem with scikit-decide's UP engine

if UPSolver.check_domain(domain):
    with UPSolver(
        operation_mode=OneshotPlanner,
        name="pyperplan",
        engine_params={"output_stream": sys.stdout},
    ) as solver:
        UPDomain.solve_with(solver, domain_factory)
        rollout(
            domain,
            solver,
            num_episodes=1,
            max_steps=100,
            max_framerate=30,
            outcome_formatter=None,
        )


# Example 2: Solving a numeric example, the same as https://github.com/aiplan4eu/unified-planning/blob/master/docs/notebooks/02-optimal-planning.ipynb

print(
    "\n\n=== EXAMPLE 2: Solving UP's numeric example using skdecide's UP ENHSP solver ===\n"
)

## Step 1: modeling the UP problem

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

## Step 2: creating the scikit-decide's UPDomain

domain_factory = lambda: UPDomain(problem)
domain = domain_factory()

## Step 3: solving the UP problem with scikit-decide's UP engine

if UPSolver.check_domain(domain):
    with UPSolver(
        operation_mode=OneshotPlanner,
        name="enhsp-opt",
        engine_params={"output_stream": sys.stdout},
    ) as solver:
        UPDomain.solve_with(solver, domain_factory)
        rollout(
            domain,
            solver,
            num_episodes=1,
            max_steps=100,
            max_framerate=30,
            outcome_formatter=None,
        )

# Example 3: Solving the same UP numeric problem but with scikit-decide's A* algorithm

print(
    "\n\n=== EXAMPLE 3: Solving UP's numeric example using skdecide's LazyAstar solver ===\n"
)

if LazyAstar.check_domain(domain):
    with LazyAstar() as solver:
        UPDomain.solve_with(solver, domain_factory)
        rollout(
            domain,
            solver,
            num_episodes=1,
            max_steps=100,
            max_framerate=30,
            outcome_formatter=None,
        )
