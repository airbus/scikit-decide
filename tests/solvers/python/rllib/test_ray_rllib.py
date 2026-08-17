#  Copyright (c) AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
import logging
import os
import tempfile
from copy import deepcopy
from enum import Enum
from typing import NamedTuple, Optional

import gymnasium as gym
import pytest
import ray
from pytest import fixture
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.ppo import PPO

from skdecide import DeterministicPlanningDomain, ImplicitSpace, Space, Value
from skdecide.builders.domain import UnrestrictedActions
from skdecide.builders.domain.events import Actions
from skdecide.core import autocast_all
from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.domain.rock_paper_scissors import RockPaperScissors
from skdecide.hub.solver.ray_rllib.ray_rllib import AsRLlibMultiAgentEnv, RayRLlib
from skdecide.hub.space.gym import EnumSpace, MultiDiscreteSpace, SetSpace
from skdecide.utils import load_registered_solver, rollout

logger = logging.getLogger(__name__)


# Allowed action handling in rllib requires to use Dict spaces for observations, which in turn
# don't support NamedTuple instances as sub-observations (cloudpickle error), therefore we use
# collections.namedtuple instead
class State(NamedTuple):
    x: int
    y: int
    s: int  # step => to make the domain cycle-free for algorithms like AO*


class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


# Same domain as in test_python_solvers, but here in dedicated workdir for rllib


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
        return self._is_goal(state) or state.s >= 99

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
        return MultiDiscreteSpace(
            nvec=[self.num_cols, self.num_rows, 100], element_class=State
        )


# restricted action  version of the domain to test action masking
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
        return State(x=0, y=0, s=0)

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        if action == Action.left:
            next_state = State(memory.x - 1, memory.y, memory.s + 1)
        elif action == Action.right:
            next_state = State(memory.x + 1, memory.y, memory.s + 1)
        elif action == Action.up:
            next_state = State(memory.x, memory.y - 1, memory.s + 1)
        else:
            next_state = State(memory.x, memory.y + 1, memory.s + 1)

        if (
            next_state.x < 0
            or next_state.x > self.num_cols - 1
            or next_state.y < 0
            or next_state.y > self.num_rows - 1
        ):
            raise ValueError("Unapplicable action!")

        return next_state

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
        return ImplicitSpace(
            lambda state: state.x == (self.num_cols - 1)
            and state.y == (self.num_rows - 1)
        )

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return EnumSpace(Action)

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return MultiDiscreteSpace(
            nvec=[self.num_cols, self.num_rows], element_class=State
        )


def test_as_rllib_env():
    domain = RockPaperScissors()
    env = AsRLlibMultiAgentEnv(domain)

    # check action space
    assert isinstance(env.action_space, gym.spaces.Dict)
    assert env.get_agent_ids() == set(env.action_space)
    for subspace in env.action_space.values():
        assert isinstance(subspace, gym.spaces.Space)

    # check observation space
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert env.get_agent_ids() == set(env.observation_space)
    for subspace in env.observation_space.values():
        assert isinstance(subspace, gym.spaces.Space)


def test_as_rllib_env_with_autocast_from_singleagent_to_multiagents():
    ENV_NAME = "CartPole-v1"

    upcast_domain = GymDomain(gym.make(ENV_NAME))
    autocast_all(upcast_domain, GymDomain, RayRLlib.T_domain)
    env = AsRLlibMultiAgentEnv(upcast_domain)

    # check action space
    assert isinstance(env.action_space, gym.spaces.Dict)
    assert env.get_agent_ids() == set(env.action_space)
    for subspace in env.action_space.values():
        assert isinstance(subspace, gym.spaces.Space)

    # check observation space
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert env.get_agent_ids() == set(env.observation_space)
    for subspace in env.observation_space.values():
        assert isinstance(subspace, gym.spaces.Space)


def ppo_config_factory():
    return (
        PPO.get_default_config()
        .env_runners(
            # set num of CPU<1 to avoid hanging for ever in github actions on macos 11)
            num_cpus_per_env_runner=0.5
        )
        .training(
            # small batch size => fast (but bad) training
            minibatch_size=32
        )
        # # uncomment next lines to debug in local mode
        # .env_runners(num_env_runners=0)
        # .learners(num_learners=0)
    )


def dqn_config_factory():
    return (
        DQN.get_default_config()
        .env_runners(
            # set num of CPU<1 to avoid hanging for ever in github actions on macos 11)
            num_cpus_per_env_runner=0.5
        )
        .training(
            # small batch size => fast (but bad) training
            train_batch_size_per_learner=32
        )
        # # uncomment next lines to debug in local mode
        # .env_runners(num_env_runners=0)
        # .learners(num_learners=0)
    )


@fixture
def ray_init():
    # add module test_gnn_ray_rllib and thus GridWorldFilteredActions to ray runtimeenv
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"working_dir": os.path.dirname(__file__)},
    )


def test_ray_rllib_solver(ray_init):
    # define domain
    domain_factory = lambda: RockPaperScissors()
    domain = domain_factory()

    # check compatibility
    assert RayRLlib.check_domain(domain)

    # solver factory
    # NB: we use here a config_factory instead of instancing direcly the config,
    # as it cannot be reused later when loading the solver, because at that point
    # the config will have been "frozen" by the first training step
    solver_kwargs = dict(
        algo_class=PPO,
        train_iterations=1,
        gamma=0.95,
        train_batch_size_per_learner_log2=8,
    )
    solver_factory = lambda: RayRLlib(
        domain_factory=domain_factory, config=ppo_config_factory(), **solver_kwargs
    )

    # solve
    solver = solver_factory()
    solver.solve()
    assert hasattr(solver, "_algo")

    assert solver._algo.config.num_cpus_per_env_runner == 0.5
    assert solver._algo.config.gamma == 0.95
    assert solver._algo.config.train_batch_size_per_learner == 256

    # solve further
    solver.solve()

    # test get_policy()
    policy = solver.get_policy()

    with tempfile.TemporaryDirectory() as tmp_save_dir:
        # store
        solver.save(tmp_save_dir)

        # rollout
        rollout(
            domain,
            solver,
            max_steps=100,
            action_formatter=lambda a: str({k: v.name for k, v in a.items()}),
            outcome_formatter=lambda o: f"{ {k: v.name for k, v in o.observation.items()} }"
            f" - rewards: { {k: v.reward for k, v in o.value.items()} }",
        )

        # load and rollout
        solver2 = solver_factory()
        solver2.load(tmp_save_dir)
        rollout(
            domain,
            solver2,
            max_steps=100,
        )


def test_ray_rllib_solver_with_filtered_actions(ray_init):
    # define domain
    domain_factory = lambda: GridWorldFilteredActions()
    domain = domain_factory()

    # check compatibility
    assert RayRLlib.check_domain(domain)

    # define and solve
    # solver_kwargs = dict(algo_class=DQN, train_iterations=1)
    # config = dqn_config_factory()
    solver_kwargs = dict(algo_class=PPO, train_iterations=1)
    config = ppo_config_factory()
    solver = RayRLlib(domain_factory=domain_factory, config=config, **solver_kwargs)
    solver.solve()
    assert hasattr(solver, "_algo")

    # rollout
    rollout(domain, solver, max_steps=100)


def test_ray_rllib_solver_on_single_agent_domain(ray_init):
    # define domain
    ENV_NAME = "CartPole-v1"
    domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
    domain = domain_factory()

    # check compatibility
    assert RayRLlib.check_domain(domain)

    # define and solve
    solver_kwargs = dict(algo_class=DQN, train_iterations=1)
    config = dqn_config_factory()

    solver_kwargs = dict(algo_class=PPO, train_iterations=1)
    config = ppo_config_factory()
    solver = RayRLlib(domain_factory=domain_factory, config=config, **solver_kwargs)
    solver.solve()
    assert hasattr(solver, "_algo")

    # rollout
    rollout(
        domain,
        solver,
        max_steps=100,
    )


def get_plan(domain, solver):
    plan = []
    cost = 0
    observation = domain.reset()
    nb_steps = 0
    while (not domain.is_goal(observation)) and nb_steps < 20:
        plan.append(solver.sample_action(observation, domain=domain))
        outcome = domain.step(plan[-1])
        cost += outcome.value.cost
        observation = outcome.observation
        nb_steps += 1
    return plan, cost


@fixture
def solver_python():
    return {
        "entry": "RayRLlib",
        "config": {"train_iterations": 1, "algo_class": DQN},
        "optimal": False,
    }


def test_solve_python(ray_init, solver_python):
    dom = GridDomain()
    solver_type = load_registered_solver(solver_python["entry"])
    solver_args = deepcopy(solver_python["config"])
    solver_args["domain_factory"] = GridDomain

    with solver_type(**solver_args) as slv:
        slv.solve()
        plan, cost = get_plan(dom, slv)
        # test get_plan and get_policy
        if hasattr(slv, "get_policy"):
            slv.get_policy()
        if hasattr(slv, "get_plan"):
            slv.get_plan()

    assert solver_type.check_domain(dom) and (
        (not solver_python["optimal"]) or (cost == 18 and len(plan) == 18)
    )


class MyCallback:
    """Callback for testing.

    - displays iteration number
    - stops after max iteration reached
    - check classes of domain and solver

    """

    def __init__(self, solver_cls, max_iter=2):
        self.solver_cls = solver_cls
        self.max_iter = max_iter
        self.iter = 0

    def __call__(self, solver, *args):
        self.iter += 1
        logger.warning(f"End of iteration #{self.iter}.")
        assert isinstance(solver, self.solver_cls)
        stopping = self.iter >= self.max_iter
        return stopping


def test_solve_python_with_cb(solver_python, caplog):
    solver_type = load_registered_solver(solver_python["entry"])
    if "callback" not in inspect.signature(solver_type.__init__).parameters:
        pytest.skip(
            f"Solver {solver_python['entry']} is not yet implementing callbacks."
        )
    solver_args = deepcopy(solver_python["config"])
    solver_args["domain_factory"] = GridDomain
    # Adding the callback
    solver_args["callback"] = MyCallback(solver_cls=solver_type)

    with solver_type(**solver_args) as slv:
        slv.solve()

    # Check that 2 iterations only were done and messages logged by callback
    assert "End of iteration #2" in caplog.text
    assert "End of iteration #3" not in caplog.text
