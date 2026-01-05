# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import random

import gymnasium as gym
import numpy as np
import pytest
import torch as th
import torch_geometric as thg
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
from skdecide.hub.solver.utils.gnn.advanced_gnn import AdvancedGNN
from skdecide.hub.solver.utils.gnn.torch_layers import Graph2NodeLayer
from skdecide.hub.solver.utils.torch.utils import extract_module_parameters_values

try:
    from plado.semantics.task import State as PladoState
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
obs_encoding = param_fixture("obs_encoding", list(ObservationEncoding))
llg_encoder_kwargs = param_fixture(
    "llg_encoder_kwargs",
    [
        dict(encode_actions=True, simplify_encoding=False),
        dict(encode_actions=True, simplify_encoding=True),
        dict(encode_actions=False, simplify_encoding=False),
        dict(encode_static_facts=False, encode_actions=False),
    ],
    ids=[
        "full-llg",
        "actions-simplified-llg",
        "no-actions-all-features-llg",
        "no-actions-no-static-facts",
    ],
)
llg_encoder_kwargs2 = param_fixture(
    "llg_encoder_kwargs2",
    [
        dict(encode_actions=True, simplify_encoding=False),
        dict(encode_actions=False, simplify_encoding=True),
    ],
    ids=["full-llg", "simplified-wo-actions-llg"],
)


@fixture
def plado_fully_observable_domain_factory_no_llg_options(
    pddl_domain_problem_paths, state_encoding, action_encoding
):
    domain_path, problem_path = pddl_domain_problem_paths
    if "agricola" in domain_path:
        if action_encoding == ActionEncoding.GYM_DISCRETE:
            pytest.skip("Discrete action encoding not tractable for agricola domain.")
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
def plado_fully_observable_domain_factory_llg_options(
    pddl_domain_problem_paths, action_encoding, llg_encoder_kwargs
):
    state_encoding = StateEncoding.GYM_GRAPH_LLG
    domain_path, problem_path = pddl_domain_problem_paths
    if "agricola" in domain_path:
        if action_encoding == ActionEncoding.GYM_DISCRETE:
            pytest.skip("Discrete action encoding not tractable for agricola domain.")
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
        llg_encoder_kwargs=llg_encoder_kwargs,
    )


plado_fully_observable_domain_factory = fixture_union(
    "plado_fully_observable_domain_factory",
    (
        plado_fully_observable_domain_factory_no_llg_options,
        plado_fully_observable_domain_factory_llg_options,
    ),
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
def plado_pddl_domain_factory_no_llg_options(
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
def plado_pddl_domain_factory_llg_options(
    blocksworld_domain_problem_paths, llg_encoder_kwargs, action_encoding
):
    domain_path, problem_path = blocksworld_domain_problem_paths
    state_encoding = StateEncoding.GYM_GRAPH_LLG
    return lambda: PladoPddlDomain(
        domain_path=domain_path,
        problem_path=problem_path,
        state_encoding=state_encoding,
        action_encoding=action_encoding,
        llg_encoder_kwargs=llg_encoder_kwargs,
    )


plado_pddl_domain_factory = fixture_union(
    "plado_pddl_domain_factory",
    (plado_pddl_domain_factory_no_llg_options, plado_pddl_domain_factory_llg_options),
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
def plado_llg_domain_factory(blocksworld_domain_problem_paths, llg_encoder_kwargs2):
    domain_path, problem_path = blocksworld_domain_problem_paths
    return lambda: PladoPddlDomain(
        domain_path=domain_path,
        problem_path=problem_path,
        state_encoding=StateEncoding.GYM_GRAPH_LLG,
        action_encoding=ActionEncoding.GYM_MULTIDISCRETE,
        llg_encoder_kwargs=llg_encoder_kwargs2,
    )


@fixture
def plado_graph_object_domain_factory(blocksworld_domain_problem_paths):
    domain_path, problem_path = blocksworld_domain_problem_paths
    return lambda: PladoTransformedObservablePddlDomain(
        domain_path=domain_path,
        problem_path=problem_path,
        obs_encoding=ObservationEncoding.GYM_GRAPH_OBJECTS,
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
    return (
        s1.atoms == s2.atoms and s1.fluents == s2.fluents
        #     (
        #     [f for i, f in enumerate(s1.fluents) if i not in cost_functions]
        #     == [f for i, f in enumerate(s2.fluents) if i not in cost_functions]
        # )
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
            if (
                domain._llg_encoder.encode_actions
                and not domain._llg_encoder.simplify_encoding
            ):
                # check decode_llg
                assert are_pladostates_equal(
                    decode_llg(obs, domain.cost_functions),
                    domain._translate_state_to_plado(obs),
                )
                assert are_pladostates_equal(
                    decode_llg(outcome.observation, domain.cost_functions),
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
        # store initial weigths to see if they update during training
        solver._init_algo()
        value_init_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        action_init_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]
        # solve
        solver.solve()
        # check policy params updated
        value_final_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        action_final_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]
        assert not (
            all(
                np.allclose(value_init_params[name], value_final_params[name])
                for name in value_init_params
            )
        ), f"value net params not updated"
        for i in range(len(action_init_params)):
            assert not (
                all(
                    np.allclose(
                        action_init_params[i][name], action_final_params[i][name]
                    )
                    for name in action_init_params[i]
                )
            ), f"action net #{i} params not updated"
        # rollout
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
    plado_graph_object_domain_factory,
):
    domain_factory = plado_graph_object_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=AutoregressiveGraphPPO,
        baselines_policy="GraphInputPolicy",
        autoregressive_action=True,
        learn_config={"total_timesteps": 300},
        n_steps=100,
    ) as solver:
        # pre-init algo (done normally during solve()) to extract init weights
        solver._init_algo()
        value_init_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_init_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_init_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]
        # solve
        solver.solve()
        # compare final parameters to see if actually updated during training
        value_final_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_final_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_final_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]

        assert not (
            all(
                np.allclose(value_init_params[name], value_final_params[name])
                for name in value_init_params
            )
        ), f"value net params not updated"
        assert not (
            all(
                np.allclose(
                    gnn_features_init_params[name], gnn_features_final_params[name]
                )
                for name in gnn_features_final_params
            )
        ), f"value net params not updated"
        for i in range(len(action_init_params)):
            assert not (
                all(
                    np.allclose(
                        action_init_params[i][name], action_final_params[i][name]
                    )
                    for name in action_init_params[i]
                )
            ), f"action net #{i} params not updated"

        # rollout
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


def test_plado_domain_blocksworld_autoregressive_advancedgnn_sb3(
    plado_graph_object_domain_factory,
):
    domain_factory = plado_graph_object_domain_factory
    gnn_hidden_channels = 16
    with (
        StableBaseline(
            domain_factory=domain_factory,
            algo_class=AutoregressiveGraphPPO,
            baselines_policy="GraphInputPolicy",
            policy_kwargs=dict(
                features_extractor_kwargs=dict(
                    # kwargs for GraphFeaturesExtractor
                    gnn_class=AdvancedGNN,
                    gnn_kwargs=dict(
                        # in_channels automatically filled by GraphFeaturesExtractor
                        hidden_channels=gnn_hidden_channels,  # output_dim as out_channels=None
                        num_layers=3,
                        dropout=0.2,
                        message_passing_cls=thg.nn.GCNConv,
                        supports_edge_weight=True,
                        supports_edge_attr=False,
                        using_encoder=True,
                    ),
                    gnn_out_dim=gnn_hidden_channels,  # correspond to GNN ouput dim
                )
            ),
            autoregressive_action=True,
            learn_config={"total_timesteps": 300},
            n_steps=100,
        ) as solver
    ):
        # pre-init algo (done normally during solve()) to extract init weights
        solver._init_algo()
        value_init_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_init_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_init_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]
        # solve
        solver.solve()
        # compare final parameters to see if actually updated during training
        value_final_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_final_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_final_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]

        assert not (
            all(
                np.allclose(value_init_params[name], value_final_params[name])
                for name in value_init_params
            )
        ), f"value net params not updated"
        assert not (
            all(
                np.allclose(
                    gnn_features_init_params[name], gnn_features_final_params[name]
                )
                for name in gnn_features_final_params
            )
        ), f"value net params not updated"
        for i in range(len(action_init_params)):
            assert not (
                all(
                    np.allclose(
                        action_init_params[i][name], action_final_params[i][name]
                    )
                    for name in action_init_params[i]
                )
            ), f"action net #{i} params not updated"

        # rollout
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
    plado_graph_object_domain_factory,
):
    domain_factory = plado_graph_object_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=AutoregressiveGraphPPO,
        baselines_policy="Graph2NodePolicy",
        autoregressive_action=True,
        learn_config={"total_timesteps": 300},
        n_steps=100,
    ) as solver:
        # pre-init algo (done normally during solve()) to extract init weights
        solver._init_algo()
        for i in range(len(domain_factory().get_action_space()._gym_space.nvec)):
            if i == 0:
                assert isinstance(solver._algo.policy.action_nets[i], th.nn.Linear)
            else:
                assert isinstance(solver._algo.policy.action_nets[i], Graph2NodeLayer)
        value_init_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_init_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_init_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]

        # solve
        solver.solve()

        # compare final parameters to see if actually updated during training
        value_final_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_final_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_final_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]

        assert not (
            all(
                np.allclose(value_init_params[name], value_final_params[name])
                for name in value_init_params
            )
        ), f"value net params not updated"
        assert not (
            all(
                np.allclose(
                    gnn_features_init_params[name], gnn_features_final_params[name]
                )
                for name in gnn_features_final_params
            )
        ), f"value net params not updated"
        for i in range(len(action_init_params)):
            assert not (
                all(
                    np.allclose(
                        action_init_params[i][name], action_final_params[i][name]
                    )
                    for name in action_init_params[i]
                )
            ), f"action net #{i} params not updated"

        # rollout
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


def test_plado_domain_blocksworld_autoregressive_heterograph2node_sb3(
    plado_llg_domain_factory,
):
    domain_factory = plado_llg_domain_factory
    domain = domain_factory()
    action_components_node_flag_indices = (
        domain.get_action_components_node_flag_indices()
    )
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=AutoregressiveGraphPPO,
        baselines_policy="HeteroGraph2NodePolicy",
        policy_kwargs=dict(
            action_components_node_flag_indices=action_components_node_flag_indices
        ),
        autoregressive_action=True,
        learn_config={"total_timesteps": 100},
        n_steps=100,
    ) as solver:
        # pre-init algo (done normally during solve()) to extract init weights
        solver._init_algo()
        for i_component, action_net in enumerate(solver._algo.policy.action_nets):
            if i_component > 0 or domain._llg_encoder.encode_actions:
                assert isinstance(action_net, Graph2NodeLayer)
            else:
                assert isinstance(action_net, th.nn.Linear)
        value_init_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_init_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_init_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]

        # solve
        solver.solve()

        # compare final parameters to see if actually updated during training
        value_final_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_final_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_final_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]

        assert not (
            all(
                np.allclose(value_init_params[name], value_final_params[name])
                for name in value_init_params
            )
        ), f"value net params not updated"
        assert not (
            all(
                np.allclose(
                    gnn_features_init_params[name], gnn_features_final_params[name]
                )
                for name in gnn_features_final_params
            )
        ), f"value net params not updated"
        for i in range(len(action_init_params)):
            assert not (
                all(
                    np.allclose(
                        action_init_params[i][name], action_final_params[i][name]
                    )
                    for name in action_init_params[i]
                )
            ), f"action net #{i} params not updated"

        # rollout
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


def test_plado_domain_blocksworld_autoregressive_heterograph2node_advancedgnn_sb3(
    plado_llg_domain_factory,
):
    domain_factory = plado_llg_domain_factory
    domain = domain_factory()
    action_components_node_flag_indices = (
        domain.get_action_components_node_flag_indices()
    )
    gnn_hidden_channels = 16
    with (
        StableBaseline(
            domain_factory=domain_factory,
            algo_class=AutoregressiveGraphPPO,
            baselines_policy="HeteroGraph2NodePolicy",
            policy_kwargs=dict(
                action_components_node_flag_indices=action_components_node_flag_indices,
                action_gnn_class=AdvancedGNN,  # Graph2NodeLayer's gnn_class
                action_gnn_kwargs=dict(  # Graph2NodeLayer's gnn_kwargs
                    # in_channels automatically filled by Graph2NodeLayer
                    # out_channels automatically filled by Graph2NodeLayer
                    hidden_channels=gnn_hidden_channels,
                    num_layers=3,
                    dropout=0.2,
                    message_passing_cls=thg.nn.GCNConv,
                    supports_edge_weight=True,
                    supports_edge_attr=False,
                    using_encoder=True,
                    using_decoder=True,
                ),
                features_extractor_kwargs=dict(
                    # kwargs for GraphFeaturesExtractor
                    gnn_class=AdvancedGNN,
                    gnn_kwargs=dict(
                        # in_channels automatically filled by GraphFeaturesExtractor
                        hidden_channels=gnn_hidden_channels,  # output_dim as out_channels=None
                        num_layers=3,
                        dropout=0.2,
                        message_passing_cls=thg.nn.GCNConv,
                        supports_edge_weight=True,
                        supports_edge_attr=False,
                        using_encoder=True,
                    ),
                    gnn_out_dim=gnn_hidden_channels,  # correspond to GNN ouput dim
                ),
            ),
            autoregressive_action=True,
            learn_config={"total_timesteps": 100},
            n_steps=100,
        ) as solver
    ):
        # pre-init algo (done normally during solve()) to extract init weights
        solver._init_algo()
        for i_component, action_net in enumerate(solver._algo.policy.action_nets):
            if i_component > 0 or domain._llg_encoder.encode_actions:
                assert isinstance(action_net, Graph2NodeLayer)
            else:
                assert isinstance(action_net, th.nn.Linear)
        value_init_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_init_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_init_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]

        # solve
        solver.solve()

        # compare final parameters to see if actually updated during training
        value_final_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        gnn_features_final_params = extract_module_parameters_values(
            solver._algo.policy.features_extractor
        )
        action_final_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]

        assert not (
            all(
                np.allclose(value_init_params[name], value_final_params[name])
                for name in value_init_params
            )
        ), f"value net params not updated"
        assert not (
            all(
                np.allclose(
                    gnn_features_init_params[name], gnn_features_final_params[name]
                )
                for name in gnn_features_final_params
            )
        ), f"value net params not updated"
        for i in range(len(action_init_params)):
            assert not (
                all(
                    np.allclose(
                        action_init_params[i][name], action_final_params[i][name]
                    )
                    for name in action_init_params[i]
                )
            ), f"action net #{i} params not updated"

        # rollout
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
