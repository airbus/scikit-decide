# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
from enum import Enum
from typing import Optional
import numpy as np
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from skdecide.builders.domain.events import Actions
from skdecide.core import Space, Value
from skdecide.domains import DeterministicPlanningDomain

from skdecide.hub.domain.simple_grid_world import SimpleGridWorld
from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.hub.space.gym import EnumSpace, ListSpace, SetSpace, BoxSpace
from skdecide.utils import rollout

# # === EXAMPLE 1: solve the cart pole OpenAI Gym domain using RLlib's PPO ===

# ENV_NAME = "CartPole-v1"

# domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
# domain = domain_factory()

# # Check domain compatibility
# if RayRLlib.check_domain(domain):
#     solver_factory = lambda: RayRLlib(PPO, train_iterations=5)

#     # Start solving
#     with solver_factory() as solver:
#         GymDomain.solve_with(solver, domain_factory)
#         solver.save("TEMP_RLlib")  # Save results

#         # Continue solving (just to demonstrate the capability to learn further)
#         solver.solve(domain_factory)
#         solver.save("TEMP_RLlib")  # Save updated results

#         # Test solution
#         rollout(
#             domain,
#             solver,
#             num_episodes=1,
#             max_steps=1000,
#             max_framerate=30,
#             outcome_formatter=None,
#         )

#     # Restore (latest results) from scratch and re-run
#     with solver_factory() as solver:
#         GymDomain.solve_with(solver, domain_factory, load_path="TEMP_RLlib")
#         rollout(
#             domain,
#             solver,
#             num_episodes=1,
#             max_steps=1000,
#             max_framerate=30,
#             outcome_formatter=None,
#         )


# # === EXAMPLE 2: solve the grid world domain with restricted actions using RLlib's DQN ===

# print(
#     "\n\033[94mEXAMPLE 2: solve the grid world domain with restricted actions using RLlib's DQN\n\033[0m"
# )

# domain_factory = lambda: SimpleGridWorld(num_cols=10, num_rows=10)
# domain = domain_factory()

# # Check domain compatibility
# if RayRLlib.check_domain(domain):
#     solver_factory = lambda: RayRLlib(PPO, train_iterations=5)

#     # Start solving
#     with solver_factory() as solver:
#         SimpleGridWorld.solve_with(solver, domain_factory)

#         # Test solution
#         rollout(
#             domain,
#             solver,
#             num_episodes=1,
#             max_steps=100,
#             max_framerate=30,
#             outcome_formatter=None,
#         )


# === EXAMPLE 3: solve the grid world domain with filtered actions using RLlib's DQN ===

print(
    "\n\033[94mEXAMPLE 3: solve the grid world domain with filtered actions using RLlib's DQN\n\033[0m"
)

# Allowed action handling in rllib requires to use Dict spaces for observations, which in turn
# don't support NamedTuple instances as sub-observations (cloudpickle error), therefore we use
# collections.namedtuple instead
State = namedtuple("State", ["x", "y"])


class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


class D(DeterministicPlanningDomain, Actions):
    T_state = State  # Type of states
    T_observation = T_state  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class GridWorldFilteredActions(D):
    def __init__(self, num_cols=10, num_rows=10):
        self.num_cols = num_cols
        self.num_rows = num_rows

    def _get_initial_state_(self) -> D.T_state:
        return State(x=0, y=0)

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        if action == Action.left:
            return State(max(memory.x - 1, 0), memory.y)
        if action == Action.right:
            return State(min(memory.x + 1, self.num_cols - 1), memory.y)
        if action == Action.up:
            return State(memory.x, max(memory.y - 1, 0))
        if action == Action.down:
            return State(memory.x, min(memory.y + 1, self.num_rows - 1))

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        allowed_actions = set()
        if memory.x > 0:
            allowed_actions.add(Action.left)
        if memory.x < self.num_cols - 1:
            allowed_actions.add(Action.right)
        if memory.y > 0:
            allowed_actions.add(Action.up)
        if memory.y < self.num_rows - 1:
            allowed_actions.add(Action.down)
        return SetSpace(allowed_actions)

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        cost = abs(next_state.x - memory.x) + abs(
            next_state.y - memory.y
        )  # every move c
        return Value(cost=cost)

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        return self._is_goal(state)

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return ListSpace([State(x=self.num_cols - 1, y=self.num_rows - 1)])

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return EnumSpace(Action)

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        # Allowed action handling in rllib requires to use Dict spaces for observations,
        # which in turn don't seem to support complex subspaces, which is why we use
        # BoxSpace instead of TupleSpace([sp.Discrete(self.num_cols), sp.Discrete(self.num_rows)])
        return BoxSpace(
            low=np.array([0, 0], dtype=np.int64),
            high=np.array([self.num_cols, self.num_rows], dtype=np.int64),
            dtype=np.int64,
        )


domain_factory = lambda: GridWorldFilteredActions(num_cols=10, num_rows=10)
domain = domain_factory()

# Check domain compatibility
if RayRLlib.check_domain(domain):
    solver_factory = lambda: RayRLlib(DQN, train_iterations=5)

    # Start solving
    with solver_factory() as solver:
        GridWorldFilteredActions.solve_with(solver, domain_factory)

        # Test solution
        rollout(
            domain,
            solver,
            num_episodes=1,
            max_steps=100,
            max_framerate=30,
            outcome_formatter=None,
        )
