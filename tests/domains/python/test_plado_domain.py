# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import random

import gymnasium as gym
import numpy as np
import pytest
from pytest_cases import fixture, fixture_union, param_fixture
from ray.rllib.algorithms.dqn import DQN

from skdecide import rollout
from skdecide.hub.domain.plado import (
    ActionEncoding,
    ObservationEncoding,
    PladoPddlDomain,
    PladoPPddlDomain,
    PladoTransformedObservablePddlDomain,
    PladoTransformedObservablePPddlDomain,
    StateEncoding,
)
from skdecide.hub.domain.plado.llg_encoder import decode_llg
from skdecide.hub.domain.plado.plado import BasePladoDomain
from skdecide.hub.solver.lazy_astar import LazyAstar
from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.autoregressive.ppo.autoregressive_ppo import (
    AutoregressiveGraphPPO,
    AutoregressivePPO,
)

try:
    import plado
except ImportError:
    pytest.skip("plado not available", allow_module_level=True)
else:
    from plado.semantics.task import State as PladoState

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
obs_encoding = param_fixture("obs_encoding", list(ObservationEncoding))


@fixture
def plado_fully_observable_domain_factory(
    pddl_domain_problem_paths, state_encoding, action_encoding
):
    domain_path, problem_path = pddl_domain_problem_paths
    if "agricola" in domain_path:
        if action_encoding == ActionEncoding.GYM_DISCRETE:
            pytest.skip("Discrete action encoding not tractable for agricola domain.")
        if state_encoding == StateEncoding.GYM_GRAPH_LLG:
            pytest.skip("LLG encoding not implemented yet for domain with fluents.")
    if "tireworld" in domain_path:
        if state_encoding == StateEncoding.GYM_GRAPH_LLG:
            pytest.skip("LLG encoding not implemented yet for PPDDL.")
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
def plado_partiallyobservable_domain_factory(
    pddl_domain_problem_paths, obs_encoding, action_encoding
):
    domain_path, problem_path = pddl_domain_problem_paths
    if "agricola" in domain_path:
        if action_encoding == ActionEncoding.GYM_DISCRETE:
            pytest.skip("Discrete action encoding not tractable for agricola domain.")
    if "tireworld" in domain_path:
        plado_domain_cls = PladoTransformedObservablePPddlDomain
    else:
        plado_domain_cls = PladoTransformedObservablePddlDomain
    return lambda: plado_domain_cls(
        domain_path=domain_path,
        problem_path=problem_path,
        obs_encoding=obs_encoding,
        action_encoding=action_encoding,
    )


plado_domain_factory = fixture_union(
    "plado_domain_factory",
    (plado_fully_observable_domain_factory, plado_partiallyobservable_domain_factory),
)


@fixture
def plado_ppddl_domain_factory(
    tireworld_domain_problem_paths, state_encoding, action_encoding
):
    if state_encoding == StateEncoding.GYM_GRAPH_LLG:
        pytest.skip("LLG encoding not implemented yet for PPDDL.")
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
        action_encoding=ActionEncoding.GYM_DISCRETE,
    )


@fixture
def plado_gym_autoregressive_domain_factory(blocksworld_domain_problem_paths):
    domain_path, problem_path = blocksworld_domain_problem_paths
    return lambda: PladoPddlDomain(
        domain_path=domain_path,
        problem_path=problem_path,
        state_encoding=StateEncoding.GYM_VECTOR,
        action_encoding=ActionEncoding.GYM_MULTIDISCRETE,
    )


@fixture
def plado_gym_gnn_autoregressive_domain_factory(
    blocksworld_domain_problem_paths, obs_encoding
):
    domain_path, problem_path = blocksworld_domain_problem_paths
    return lambda: PladoTransformedObservablePddlDomain(
        domain_path=domain_path,
        problem_path=problem_path,
        obs_encoding=obs_encoding,
        action_encoding=ActionEncoding.GYM_MULTIDISCRETE,
    )


@fixture
def plado_ppddl_gym_autoregressive_domain_factory(tireworld_domain_problem_paths):
    domain_path, problem_path = tireworld_domain_problem_paths
    return lambda: PladoPPddlDomain(
        domain_path=domain_path,
        problem_path=problem_path,
        state_encoding=StateEncoding.GYM_VECTOR,
        action_encoding=ActionEncoding.GYM_MULTIDISCRETE,
    )


def are_graphs_equal(
    g1: gym.spaces.GraphInstance, g2: gym.spaces.GraphInstance
) -> bool:
    return (
        (g1.edge_links == g2.edge_links).all()
        and (g1.nodes == g2.nodes).all()
        and (g1.edges is None and g2.edges is None)
        or (g1.edges == g2.edges).all()
    )


def are_pladostates_equal(s1: PladoState, s2: PladoState) -> bool:
    return s1.atoms == s2.atoms and s1.fluents == s2.fluents


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
    if isinstance(domain, BasePladoDomain):
        pladostate = domain.task.initial_state
        new_pladostate = domain._translate_state_to_plado(
            domain._translate_state_from_plado(pladostate)
        )
        assert are_pladostates_equal(new_pladostate, pladostate)
        if domain.state_encoding == StateEncoding.NATIVE:
            assert (
                domain._translate_state_from_plado(
                    domain._translate_state_to_plado(obs)
                )
                == obs
            )
        elif domain.state_encoding == StateEncoding.GYM_VECTOR:
            assert (
                domain._translate_state_from_plado(
                    domain._translate_state_to_plado(obs)
                )
                == obs
            ).all()
        elif domain.state_encoding == StateEncoding.GYM_GRAPH_LLG:
            assert are_graphs_equal(
                domain._translate_state_from_plado(
                    domain._translate_state_to_plado(obs)
                ),
                obs,
            )
            # check decode_llg
            assert are_pladostates_equal(
                decode_llg(obs), domain._translate_state_to_plado(obs)
            )
            assert are_pladostates_equal(
                decode_llg(outcome.observation),
                domain._translate_state_to_plado(outcome.observation),
            )
        else:
            raise NotImplementedError()

        if domain.action_encoding == ActionEncoding.GYM_MULTIDISCRETE:
            assert (
                domain._translate_action_from_plado(
                    domain._translate_action_to_plado(action)
                )
                == action
            ).all()
        else:
            assert (
                domain._translate_action_from_plado(
                    domain._translate_action_to_plado(action)
                )
                == action
            )
        assert domain._translate_action_to_plado(
            domain._translate_action_from_plado(
                domain._translate_action_to_plado(action)
            )
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
    elif isinstance(state1, gym.spaces.GraphInstance):
        assert are_graphs_equal(state1, state2)
        assert are_graphs_equal(state2, state3)
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


def test_plado_domain_blocksworld_autoregressive_sb3(
    plado_gym_autoregressive_domain_factory,
):
    domain_factory = plado_gym_autoregressive_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=AutoregressivePPO,
        baselines_policy="MlpPolicy",
        autoregressive_action=True,
        learn_config={"total_timesteps": 300},
        n_steps=100,
    ) as solver:
        solver.solve()
        max_steps = 20
        episodes = rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=max_steps,
            num_episodes=1,
            render=False,
            return_episodes=True,
        )
        observations, actions, values = episodes[0]
        #  assert len(actions) < max_steps - 1  # unsucessful to reach goal


def test_plado_domain_ppddl_autoregressive_sb3(
    plado_ppddl_gym_autoregressive_domain_factory,
):
    domain_factory = plado_ppddl_gym_autoregressive_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=AutoregressivePPO,
        baselines_policy="MlpPolicy",
        autoregressive_action=True,
        learn_config={"total_timesteps": 300},
        n_steps=100,
    ) as solver:
        solver.solve()
        max_steps = 20
        episodes = rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=max_steps,
            num_episodes=1,
            render=False,
            return_episodes=True,
        )
        observations, actions, values = episodes[0]
        #  assert len(actions) < max_steps - 1  # unsucessful to reach goal


def test_plado_domain_blocksworld_autoregressive_gnn_sb3(
    plado_gym_gnn_autoregressive_domain_factory,
):
    domain_factory = plado_gym_gnn_autoregressive_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=AutoregressiveGraphPPO,
        baselines_policy="GraphInputPolicy",
        autoregressive_action=True,
        learn_config={"total_timesteps": 300},
        n_steps=100,
    ) as solver:
        solver.solve()
        max_steps = 20
        episodes = rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=max_steps,
            num_episodes=1,
            render=False,
            return_episodes=True,
        )
        observations, actions, values = episodes[0]
        #  assert len(actions) < max_steps - 1  # unsucessful to reach goal


def test_plado_domain_blocksworld_autoregressive_graph2node_sb3(
    plado_gym_gnn_autoregressive_domain_factory,
):
    domain_factory = plado_gym_gnn_autoregressive_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=AutoregressiveGraphPPO,
        baselines_policy="Graph2NodePolicy",
        autoregressive_action=True,
        learn_config={"total_timesteps": 300},
        n_steps=100,
    ) as solver:
        solver.solve()
        max_steps = 20
        episodes = rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=max_steps,
            num_episodes=1,
            render=False,
            return_episodes=True,
        )
        observations, actions, values = episodes[0]
        #  assert len(actions) < max_steps - 1  # unsucessful to reach goal
