# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch

from skdecide.hub.solver.ray_rllib import RayRLlib

from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN

from gymnasium.spaces.box import Box

import unified_planning
from unified_planning.shortcuts import (
    BoolType,
    Fluent,
    InstantaneousAction,
    Not,
    OneshotPlanner,
    UserType,
)

from skdecide.hub.domain.up import UPDomain
from skdecide.hub.solver.lazy_astar import LazyAstar
from skdecide.hub.solver.up import UPSolver
from skdecide.utils import rollout


class TorchParametricActionsModel(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        true_obs_shape=(4,),
        action_embed_size=2,
        **kw
    ):
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )

        self.action_embed_model = TorchFC(
            Box(-1, 1, shape=true_obs_shape),
            action_space,
            action_embed_size,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_embed, _ = self.action_embed_model({"obs": input_dict["obs"]["cart"]})

        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        intent_vector = torch.unsqueeze(action_embed, 1)

        # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
        action_logits = torch.sum(avail_actions * intent_vector, dim=2)

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


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

problem.add_quality_metric(
    unified_planning.model.metrics.MinimizeActionCosts({move: 1})
)

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

# Example 2: Solving the same example but with RLLib's DQN

print(
    "\n\n=== EXAMPLE 2: Solving UP's basic example using skdecide's RLLib's DQN solver ===\n"
)

domain_factory = lambda: UPDomain(
    problem, state_encoding="vector", action_masking="vector"
)
domain = domain_factory()

ModelCatalog.register_custom_model("pa_model", TorchParametricActionsModel)

if True:  # RayRLlib.check_domain(domain):
    with RayRLlib(
        algo_class=DQN,
        train_iterations=5,
        config=DQNConfig()
        .training(model={"custom_model": "pa_model"}, hiddens=[], dueling=False)
        .framework("torch"),
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
