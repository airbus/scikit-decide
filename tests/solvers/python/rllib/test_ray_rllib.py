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
from typing import NamedTuple, Optional, Union

import gymnasium as gym
import numpy as np
import ray
import torch
from gymnasium.envs.classic_control import PendulumEnv
from pytest import fixture
from pytest_cases import parametrize
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.dqn.default_dqn_rl_module import (
    QF_LOGITS,
    QF_PREDS,
    QF_PROBS,
)
from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import (
    DefaultDQNTorchRLModule,
)
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core import Columns
from ray.rllib.core.models.base import Encoder, Model
from ray.rllib.core.models.torch.encoder import (
    TorchActorCriticEncoder,
    TorchCNNEncoder,
    TorchMLPEncoder,
)
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.utils.torch_utils import FLOAT_MIN, nn
from ray.rllib.utils.typing import TensorType

from skdecide import DeterministicPlanningDomain, ImplicitSpace, Space, Value
from skdecide.builders.domain import UnrestrictedActions
from skdecide.builders.domain.events import Actions
from skdecide.core import autocast_all
from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.domain.rock_paper_scissors import RockPaperScissors
from skdecide.hub.solver.ray_rllib.action_masking.algorithms.dqn.torch.action_masking_dqn_torch_rl_module import (
    ActionMaskingDQNTorchRLModule,
)
from skdecide.hub.solver.ray_rllib.action_masking.algorithms.ppo.torch.action_masking_ppo_torch_rl_module import (
    ActionMaskingPPOTorchRLModule,
)
from skdecide.hub.solver.ray_rllib.action_masking.rl_module.base import (
    ActionMaskingRLModule,
)
from skdecide.hub.solver.ray_rllib.ray_rllib import (
    SK_DEFAULT_MODULE_ID,
    AsRLlibMultiAgentEnv,
    RayRLlib,
)
from skdecide.hub.solver.utils.torch.utils import extract_module_parameters_values
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
        # small batch size => fast (but bad) training
        .training(minibatch_size=16, train_batch_size_per_learner=32)
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
        train_batch_size_per_learner_log2=4,
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
    assert solver._algo.config.train_batch_size_per_learner == 16

    # store weigths to see if they update during training
    rl_module = solver.get_policy()[SK_DEFAULT_MODULE_ID]
    assert isinstance(rl_module, DefaultPPOTorchRLModule)
    value_init_params = extract_module_parameters_values(rl_module.vf)
    action_init_params = extract_module_parameters_values(rl_module.pi)

    # solve further
    solver.solve()

    # weights have been updated?
    value_final_params = extract_module_parameters_values(rl_module.vf)
    action_final_params = extract_module_parameters_values(rl_module.pi)
    assert not (
        all(
            np.allclose(action_init_params[name], action_final_params[name])
            for name in action_init_params
        )
    )

    # assert not (
    #     all(
    #         np.allclose(value_init_params[name], value_final_params[name])
    #         for name in value_init_params
    #     )
    # ), value_init_params

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


@parametrize(
    "algo_class, config_factory, expected_rl_module_cls",
    [
        (PPO, ppo_config_factory, ActionMaskingPPOTorchRLModule),
        (DQN, dqn_config_factory, ActionMaskingDQNTorchRLModule),
    ],
)
def test_ray_rllib_solver_with_filtered_actions(
    ray_init, algo_class, config_factory, expected_rl_module_cls
):
    # define domain
    domain_factory = lambda: GridWorldFilteredActions()
    domain = domain_factory()

    # check compatibility
    assert RayRLlib.check_domain(domain)

    # define and solve
    solver_kwargs = dict(algo_class=algo_class, train_iterations=1)
    config = config_factory()
    solver = RayRLlib(domain_factory=domain_factory, config=config, **solver_kwargs)
    solver.solve()
    assert hasattr(solver, "_algo")
    assert isinstance(
        solver._algo.get_module(SK_DEFAULT_MODULE_ID), expected_rl_module_cls
    )

    # rollout
    rollout(domain, solver, max_steps=100)


class MyModule(ActionMaskingRLModule, DefaultDQNTorchRLModule):
    """Custom RL module for DQN + masking."""

    def _mask_action_logits(
        self,
        batch: dict[str, TensorType],
        action_mask: TensorType,
    ) -> dict[str, TensorType]:
        """Masks the action logits for the output of `forward` methods

        Args:
            batch: A dictionary containing tensors (at least action logits).
            action_mask: A tensor containing the action mask for the current
                observations.

        Returns:
            A modified batch with masked action logits for the action distribution
            inputs.
        """
        # Convert action mask into an `[0.0][-inf]`-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)

        # Mask the logits.
        for column in (QF_PREDS, QF_LOGITS, Columns.ACTION_DIST_INPUTS):
            if column in batch:
                batch[column] += inf_mask

        if QF_PROBS in batch:
            if QF_LOGITS in batch:
                batch[QF_PROBS] = nn.functional.softmax(batch[QF_LOGITS], dim=-1)
            else:
                raise RuntimeError()

        # Return the batch with the masked action logits.
        return batch

    def _qf_forward_helper(
        self,
        batch: dict[str, TensorType],
        encoder: Encoder,
        head: Union[Model, dict[str, Model]],
    ) -> dict[str, TensorType]:
        # Check, if the observations are still in `dict` form.
        # if isinstance(batch[Columns.OBS], dict):
        action_mask, batch = self.preprocess_batch(batch)
        # # NOTE: Because we manipulate the batch we need to add the `action_mask`
        # # to the batch to access them in `_forward_train`.
        # batch["action_mask"] = action_mask
        outs = super()._qf_forward_helper(batch, encoder, head)
        return self._mask_action_logits(outs, action_mask)

    def _forward_exploration(
        self, batch: dict[str, TensorType], t: int
    ) -> dict[str, TensorType]:
        """Forward pass during exploration.

        We need to override as a multinomial on Q-values is taken, originally excluding *0* valued logits.
        But after masking, we should have FLOAT_MIN (approximating -inf) values excluded instead.

        """
        # Define the return dictionary.
        output = {}

        # Q-network forward pass.
        qf_outs = self.compute_q_values(batch)

        # Get action distribution.
        action_dist_cls = self.get_exploration_action_dist_cls()
        action_dist = action_dist_cls.from_logits(qf_outs[QF_PREDS])
        # Note, the deterministic version of the categorical distribution
        # outputs directly the `argmax` of the logits.
        exploit_actions = action_dist.to_deterministic().sample()

        # We need epsilon greedy to support exploration.
        # TODO (simon): Implement sampling for nested spaces.
        # Update scheduler.
        self.epsilon_schedule.update(t)
        # Get the actual epsilon,
        epsilon = self.epsilon_schedule.get_current_value()
        # Apply epsilon-greedy exploration.
        B = qf_outs[QF_PREDS].shape[0]
        random_actions = torch.squeeze(
            torch.multinomial(
                (
                    torch.nan_to_num(
                        qf_outs[QF_PREDS].reshape(-1, qf_outs[QF_PREDS].size(-1)),
                        neginf=FLOAT_MIN,
                    )
                    != FLOAT_MIN
                ).float(),
                num_samples=1,
            ),
            dim=1,
        )

        actions = torch.where(
            torch.rand((B,)) < epsilon,
            random_actions,
            exploit_actions,
        )

        # Add the actions to the return dictionary.
        output[Columns.ACTIONS] = actions

        # If this is a stateful module, add output states.
        if Columns.STATE_OUT in qf_outs:
            output[Columns.STATE_OUT] = qf_outs[Columns.STATE_OUT]

        return output


def test_ray_rllib_solver_with_custom_rlmodule(ray_init):
    # define domain
    domain_factory = lambda: GridWorldFilteredActions()
    domain = domain_factory()

    # define and solve
    agent2module_id = {agent: "my_module_id" for agent in domain.get_agents()}
    rl_module_spec = MultiRLModuleSpec(
        rl_module_specs={
            "my_module_id": RLModuleSpec(
                module_class=MyModule,
                model_config={},
            )
        }
    )
    solver_kwargs = dict(
        algo_class=DQN,
        train_iterations=1,
        rl_module_spec=rl_module_spec,
        agent2module_id=agent2module_id,
    )
    config = dqn_config_factory()
    solver = RayRLlib(domain_factory=domain_factory, config=config, **solver_kwargs)
    solver.solve()
    assert hasattr(solver, "_algo")
    assert isinstance(solver._algo.get_module("my_module_id"), MyModule)

    # rollout
    rollout(domain, solver, max_steps=100)


class Pendulum3dEnv(PendulumEnv):
    obs_shape = (3, 1, 1)
    observation_space: gym.spaces.Box

    def __init__(self, render_mode: str | None = None, g=10.0):
        super().__init__(render_mode, g)
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.reshape(self.obs_shape),
            high=self.observation_space.low.reshape(self.obs_shape),
            dtype=self.observation_space.dtype,
            shape=self.obs_shape,
        )

    def _get_obs(self):
        return super()._get_obs().reshape(self.obs_shape)


@parametrize("flatten_3d_obs", [True, False])
def test_ray_rllib_solver_on_single_agent_domain(ray_init, flatten_3d_obs):
    # define domain
    domain_factory = lambda: GymDomain(
        gym.make(gym.envs.registration.EnvSpec("pendulum", entry_point=Pendulum3dEnv))
    )
    domain = domain_factory()

    # check compatibility
    assert RayRLlib.check_domain(domain)

    # define and solve
    config = ppo_config_factory()
    solver = RayRLlib(
        domain_factory=domain_factory,
        algo_class=PPO,
        config=config,
        flatten_3d_obs=flatten_3d_obs,
        train_iterations=1,
        model_configs={SK_DEFAULT_MODULE_ID: {"conv_filters": [(2, (2, 1), (2, 1))]}},
    )
    solver.solve()
    assert hasattr(solver, "_algo")
    rl_module = solver.get_policy()
    agent_rl_module = rl_module.get(SK_DEFAULT_MODULE_ID)
    assert isinstance(agent_rl_module, DefaultPPOTorchRLModule)
    assert isinstance(agent_rl_module.encoder, TorchActorCriticEncoder)
    if flatten_3d_obs:
        assert isinstance(agent_rl_module.encoder.actor_encoder, TorchMLPEncoder)
    else:
        assert isinstance(agent_rl_module.encoder.actor_encoder, TorchCNNEncoder)

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
