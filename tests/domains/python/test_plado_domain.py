# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import random

import numpy as np
import pytest
from pytest_cases import fixture, fixture_union, param_fixture
from ray.rllib.algorithms.dqn import DQN

from skdecide import rollout
from skdecide.hub.domain.plado import (
    ActionEncoding,
    PladoPddlDomain,
    PladoPPddlDomain,
    StateEncoding,
)
from skdecide.hub.solver.lazy_astar import LazyAstar
from skdecide.hub.solver.ray_rllib import RayRLlib

try:
    import plado
except ImportError:
    pytest.skip("plado not available", allow_module_level=True)


test_dir = os.path.dirname(os.path.abspath(__file__))


@fixture
def blocksworld_domain_problem_paths():
    domain_path = f"{test_dir}/pddl_domains/blocks/domain.pddl"
    problem_path = f"{test_dir}/pddl_domains/blocks/probBLOCKS-3-0.pddl"
    return domain_path, problem_path


@fixture
def agricola_domain_problem_paths():
    domain_path = f"{test_dir}/pddl_domains/agricola-opt18/domain.pddl"
    problem_path = f"{test_dir}/pddl_domains/agricola-opt18/p01.pddl"
    return domain_path, problem_path


@fixture
def tireworld_domain_problem_paths():
    domain_path = f"{test_dir}/pddl_domains/tireworld/domain.pddl"
    problem_path = f"{test_dir}/pddl_domains/tireworld/p01.pddl"
    return domain_path, problem_path


pddl_domain_problem_paths = fixture_union(
    "pddl_domain_problem_paths",
    (
        blocksworld_domain_problem_paths,
        agricola_domain_problem_paths,
        tireworld_domain_problem_paths,
    ),
)


state_encoding = param_fixture("state_encoding", list(StateEncoding))
action_encoding = param_fixture("action_encoding", list(ActionEncoding))


@fixture
def plado_domain_factory(pddl_domain_problem_paths, state_encoding, action_encoding):
    domain_path, problem_path = pddl_domain_problem_paths
    if "agricola" in domain_path and action_encoding == ActionEncoding.GYM:
        pytest.skip("Discrete action encoding not tractable for agricola domain.")
    if "tireworld" in domain_path:
        plado_domain_cls = PladoPPddlDomain
    else:
        plado_domain_cls = PladoPddlDomain
    return lambda: plado_domain_cls(
        domain_path=domain_path,
        problem_path=problem_path,
        state_encoding=state_encoding,
        action_encoding=action_encoding,
    )


@fixture
def plado_ppddl_domain_factory(
    tireworld_domain_problem_paths, state_encoding, action_encoding
):
    domain_path, problem_path = tireworld_domain_problem_paths
    return lambda: PladoPPddlDomain(
        domain_path=domain_path,
        problem_path=problem_path,
        state_encoding=state_encoding,
        action_encoding=action_encoding,
    )


@fixture
def plado_pddl_domain_factory(
    blocksworld_domain_problem_paths, state_encoding, action_encoding
):
    domain_path, problem_path = blocksworld_domain_problem_paths
    return lambda: PladoPddlDomain(
        domain_path=domain_path,
        problem_path=problem_path,
        state_encoding=state_encoding,
        action_encoding=action_encoding,
    )


@fixture
def plado_native_domain_factory(blocksworld_domain_problem_paths):
    domain_path, problem_path = blocksworld_domain_problem_paths
    return lambda: PladoPddlDomain(
        domain_path=domain_path,
        problem_path=problem_path,
        state_encoding=StateEncoding.NATIVE,
        action_encoding=ActionEncoding.NATIVE,
    )


@fixture
def plado_gym_naive_domain_factory(blocksworld_domain_problem_paths):
    domain_path, problem_path = blocksworld_domain_problem_paths
    return lambda: PladoPddlDomain(
        domain_path=domain_path,
        problem_path=problem_path,
        state_encoding=StateEncoding.GYM_VECTOR,
        action_encoding=ActionEncoding.GYM,
    )


def test_plado_domain_random(plado_domain_factory):
    domain_factory = plado_domain_factory

    domain = domain_factory()
    obs = domain.reset()
    assert obs in domain.get_observation_space()

    actions = domain.get_applicable_actions()
    action = actions.get_elements()[0]
    assert action in domain.get_action_space()

    outcome = domain.step(action)
    assert outcome.observation in domain.get_observation_space()

    # check conversions
    pladostate = domain.task.initial_state
    new_pladostate = domain._translate_state_to_plado(
        domain._translate_state_from_plado(pladostate)
    )
    assert new_pladostate.fluents == pladostate.fluents
    assert new_pladostate.atoms == pladostate.atoms
    if domain.state_encoding == StateEncoding.NATIVE:
        assert (
            domain._translate_state_from_plado(domain._translate_state_to_plado(obs))
            == obs
        )
    elif domain.state_encoding == StateEncoding.GYM_VECTOR:
        assert (
            domain._translate_state_from_plado(domain._translate_state_to_plado(obs))
            == obs
        ).all()

    assert (
        domain._translate_action_from_plado(domain._translate_action_to_plado(action))
        == action
    )
    assert domain._translate_action_to_plado(
        domain._translate_action_from_plado(domain._translate_action_to_plado(action))
    ) == domain._translate_action_to_plado(action)

    # rollout with random walk
    rollout(
        domain,
        max_steps=5,
        num_episodes=1,
        render=False,
    )


def test_plado_state_sample_ppddl(plado_ppddl_domain_factory):
    def reset() -> tuple[
        PladoPPddlDomain, PladoPPddlDomain.T_state, PladoPPddlDomain.T_event
    ]:
        random.seed(42)
        domain = plado_ppddl_domain_factory()
        domain.reset()
        memory = domain._memory
        action = domain.get_applicable_actions().get_elements()[0]
        return domain, memory, action

    domain, memory, action = reset()
    outcome = domain._state_sample(memory, action)
    state1 = outcome.state
    value1 = outcome.value

    domain, memory, action = reset()
    state2 = domain._get_next_state_distribution(memory, action).sample()
    value2 = domain._get_transition_value(memory, action, state2)

    domain, memory, action = reset()
    value3 = domain._get_transition_value(memory, action, state2)
    state3 = domain._get_next_state_distribution(memory, action).sample()

    if isinstance(state1, np.ndarray):
        assert (state1 == state2).all()
        assert (state3 == state2).all()
    else:
        assert state1 == state2
        assert state3 == state2

    assert value1 == value2
    assert value3 == value2


def test_plado_state_sample_pddl(plado_pddl_domain_factory):
    def reset() -> tuple[
        PladoPddlDomain, PladoPddlDomain.T_state, PladoPddlDomain.T_event
    ]:
        domain = plado_pddl_domain_factory()
        domain.reset()
        memory = domain._memory
        action = domain.get_applicable_actions().get_elements()[0]
        return domain, memory, action

    domain, memory, action = reset()
    outcome = domain._state_sample(memory, action)
    state1 = outcome.state
    value1 = outcome.value

    domain, memory, action = reset()
    state2 = domain._get_next_state(memory, action)
    value2 = domain._get_transition_value(memory, action, state2)

    domain, memory, action = reset()
    value3 = domain._get_transition_value(memory, action, state2)
    state3 = domain._get_next_state(memory, action)

    if isinstance(state1, np.ndarray):
        assert (state1 == state2).all()
        assert (state3 == state2).all()
    else:
        assert state1 == state2
        assert state3 == state2

    assert value1 == value2
    assert value3 == value2


def test_plado_domain_planning(plado_native_domain_factory):
    domain_factory = plado_native_domain_factory

    assert LazyAstar.check_domain(domain_factory())
    with LazyAstar(
        domain_factory=domain_factory,
    ) as solver:
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
        )


def test_plado_domain_blocksworld_rl(plado_gym_naive_domain_factory):
    domain_factory = plado_gym_naive_domain_factory

    assert RayRLlib.check_domain(domain_factory())
    with RayRLlib(
        domain_factory=domain_factory,
        algo_class=DQN,
        train_iterations=1,
    ) as solver:
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
        )
