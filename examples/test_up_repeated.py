import unified_planning
from unified_planning.shortcuts import (
    Or,
    GE,
    BoolType,
    Int,
    IntType,
    SimulatedEffect,
    UserType,
)

from skdecide.hub.domain.up import UPDomain
from skdecide.utils import rollout
from ray.rllib.algorithms.dqn import DQN
from skdecide.hub.solver.ray_rllib import RayRLlib
import ray

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
    state_encoding="repeated",
    action_encoding="int",
)

domain = domain_factory()

rollout(
    domain_factory(),
    None,
    num_episodes=1,
    max_steps=100,
    #max_framerate=30,
    outcome_formatter=None,
)

print('Initialise Solver ... \n')
solver = RayRLlib(
    domain_factory=domain_factory,
    algo_class=DQN,
    train_iterations=1,)

solver.solve()
print("done")

rollout(
    domain_factory(),
    solver,
    num_episodes=1,
    max_steps=100,
    #max_framerate=30,
    outcome_formatter=None,
)
