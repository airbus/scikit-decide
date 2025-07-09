from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch as th

from skdecide import rollout
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.autoregressive.ppo.autoregressive_ppo import (
    AutoregressiveGraphPPO,
    AutoregressivePPO,
)
from skdecide.hub.solver.utils.gnn.torch_layers import Graph2NodeLayer
from skdecide.hub.solver.utils.torch.utils import extract_module_parameters_values


def test_autoregressive_ppo_w_gym_env(graph_walk_env):
    env = graph_walk_env

    algo = AutoregressivePPO(
        "MlpPolicy",
        env,
        n_steps=100,
    )

    value_init_params = extract_module_parameters_values(algo.policy.value_net)
    action_init_params = [
        extract_module_parameters_values(action_net)
        for action_net in algo.policy.action_nets
    ]
    algo.learn(total_timesteps=500)
    value_final_params = extract_module_parameters_values(algo.policy.value_net)
    action_final_params = [
        extract_module_parameters_values(action_net)
        for action_net in algo.policy.action_nets
    ]

    # check policy params updated
    assert not (
        all(
            np.allclose(value_init_params[name], value_final_params[name])
            for name in value_init_params
        )
    ), f"value net params not updated"
    for i in range(len(action_init_params)):
        assert not (
            all(
                np.allclose(action_init_params[i][name], action_final_params[i][name])
                for name in action_init_params[i]
            )
        ), f"action net #{i} params not updated"

    # rollout
    obs, info = env.reset()
    terminal = False
    i_step = 0
    print(f"#{i_step}: obs={obs}, terminal={terminal}")

    max_steps = 20

    while i_step < max_steps and not terminal:
        i_step += 1
        action, _ = algo.predict(obs, action_masks=env.action_masks())
        obs, reward, terminal, truncated, info = env.step(action)
        print(f"#{i_step}: action={action}, obs={obs}, terminal={terminal}")

    assert i_step < max_steps  # optimal would be 2, but not always found...


def test_autoregressive_ppo_w_skdecide_domain(graph_walk_domain_factory):
    domain_factory = graph_walk_domain_factory
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
        solver.solve()
        value_final_params = extract_module_parameters_values(
            solver._algo.policy.value_net
        )
        action_final_params = [
            extract_module_parameters_values(action_net)
            for action_net in solver._algo.policy.action_nets
        ]
        # check policy params updated
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
        assert (
            len(actions) < max_steps - 1
        )  # optimal would be 2, but not always found...


def test_autoregressive_graph_ppo(
    graph_walk_with_graph_obs_domain_factory,
):
    domain_factory = graph_walk_with_graph_obs_domain_factory
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
        assert (
            len(actions) < max_steps - 1
        )  # optimal would be 2, but not always found...


def test_autoregressive_graph2node_ppo(
    graph_walk_with_graph_obs_domain_factory,
):
    domain_factory = graph_walk_with_graph_obs_domain_factory
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
        for i in range(len(domain_factory()._gym_env.action_space.nvec)):
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
        assert (
            len(actions) < max_steps - 1
        )  # optimal would be 2, but not always found...


def test_autoregressive_heterograph2node_ppo(
    graph_walk_with_heterograph_obs_domain_factory,
    action_components_node_flag_indices,
):
    domain_factory = graph_walk_with_heterograph_obs_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=AutoregressiveGraphPPO,
        baselines_policy="HeteroGraph2NodePolicy",
        autoregressive_action=True,
        learn_config={"total_timesteps": 300},
        n_steps=100,
        policy_kwargs=dict(
            action_components_node_flag_indices=action_components_node_flag_indices,
            n_graph2node_components=1,
        ),
    ) as solver:
        # pre-init algo (done normally during solve()) to extract init weights
        solver._init_algo()
        for i in range(len(domain_factory()._gym_env.action_space.nvec)):
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
        assert (
            len(actions) < max_steps - 1
        )  # optimal would be 2, but not always found...


def test_autoregressive_heterograph2node_ppo_save_load(
    graph_walk_with_heterograph_obs_domain_factory,
    graph_walk_with_heterograph_obs_domain_factory_2,
    action_components_node_flag_indices,
):
    domain_factory = graph_walk_with_heterograph_obs_domain_factory
    domain_factory_2 = graph_walk_with_heterograph_obs_domain_factory_2

    with TemporaryDirectory() as directory:
        path = f"{directory}/sb3"
        with StableBaseline(
            domain_factory=domain_factory,
            algo_class=AutoregressiveGraphPPO,
            baselines_policy="HeteroGraph2NodePolicy",
            autoregressive_action=True,
            learn_config={"total_timesteps": 30},
            n_steps=10,
            policy_kwargs=dict(
                action_components_node_flag_indices=action_components_node_flag_indices,
                n_graph2node_components=1,
            ),
        ) as solver:
            # solve
            solver.solve()
            # store
            solver.save(path)

        # load for another domain (more objects) + solve + rollout
        with StableBaseline(
            domain_factory=domain_factory_2,
            algo_class=AutoregressiveGraphPPO,
            baselines_policy="HeteroGraph2NodePolicy",
            autoregressive_action=True,
            learn_config={"total_timesteps": 30},
            n_steps=15,
            policy_kwargs=dict(
                action_components_node_flag_indices=action_components_node_flag_indices,
                n_graph2node_components=1,
            ),
        ) as solver:
            # load
            solver.load(path)
            assert solver._algo.n_steps == 15

            # solve
            solver.solve()

            # rollout on domain 2
            max_steps = 20
            episodes = rollout(
                domain=graph_walk_with_heterograph_obs_domain_factory_2(),
                solver=solver,
                max_steps=max_steps,
                num_episodes=1,
                render=False,
                return_episodes=True,
            )

        # load for another domain (more objects) + solve + rollout
        with StableBaseline(
            domain_factory=domain_factory_2,
            algo_class=AutoregressiveGraphPPO,
            baselines_policy="HeteroGraph2NodePolicy",
            autoregressive_action=True,
            learn_config={"total_timesteps": 30},
            n_steps=15,
            policy_kwargs=dict(
                action_components_node_flag_indices=action_components_node_flag_indices,
                n_graph2node_components=1,
            ),
        ) as solver:
            # load
            solver.load(path)
            assert solver._algo.n_steps == 15

            # rollout on domain 2
            max_steps = 20
            episodes = rollout(
                domain=graph_walk_with_heterograph_obs_domain_factory_2(),
                solver=solver,
                max_steps=max_steps,
                num_episodes=1,
                render=False,
                return_episodes=True,
            )


def test_autoregressive_heterograph2node_ppo_save_load_nok(
    graph_walk_with_heterograph_obs_domain_factory,
    graph_walk_with_heterograph_obs_domain_factory_3,
    action_components_node_flag_indices,
):
    domain_factory = graph_walk_with_heterograph_obs_domain_factory
    domain_factory_2 = graph_walk_with_heterograph_obs_domain_factory_3

    with TemporaryDirectory() as directory:
        path = f"{directory}/sb3"
        with StableBaseline(
            domain_factory=domain_factory,
            algo_class=AutoregressiveGraphPPO,
            baselines_policy="HeteroGraph2NodePolicy",
            autoregressive_action=True,
            learn_config={"total_timesteps": 30},
            n_steps=10,
            policy_kwargs=dict(
                action_components_node_flag_indices=action_components_node_flag_indices,
                n_graph2node_components=1,
            ),
        ) as solver:
            # solve
            solver.solve()
            # store
            solver.save(path)

        # load for another domain (more objects) + solve + rollout
        with StableBaseline(
            domain_factory=domain_factory_2,
            algo_class=AutoregressiveGraphPPO,
            baselines_policy="HeteroGraph2NodePolicy",
            autoregressive_action=True,
            learn_config={"total_timesteps": 30},
            n_steps=15,
            policy_kwargs=dict(
                action_components_node_flag_indices=action_components_node_flag_indices,
                n_graph2node_components=1,
            ),
        ) as solver:
            # load nok
            with pytest.raises(AssertionError):
                solver.load(path)
