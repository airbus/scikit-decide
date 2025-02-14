import logging
import os

import numpy as np
import pytest
import ray
from pytest_cases import fixture

from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.hub.solver.ray_rllib.gnn.algorithms import GraphPPO
from skdecide.utils import rollout


@fixture
def ray_init():
    # add module test_gnn_ray_rllib and thus GraphMaze to ray runtimeenv
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"working_dir": os.path.dirname(__file__)},
        # local_mode=True,  # uncomment this line and comment the one above to debug more easily
    )


@fixture
def graphppo_config():
    return (
        GraphPPO.get_default_config()
        # set num of CPU<1 to avoid hanging for ever in github actions on macos 11
        .resources(
            num_cpus_per_worker=0.5,
        )
        # small number to increase speed of the unit test
        .training(train_batch_size=256)
    )


def test_ppo(unmasked_graph_maze_domain_factory, graphppo_config, ray_init):
    domain_factory = unmasked_graph_maze_domain_factory
    solver_kwargs = dict(
        algo_class=GraphPPO, train_iterations=1  # , gamma=0.95, train_batch_size_log2=8
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert not solver._action_masking and solver._is_graph_obs
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )


def test_ppo_user_gnn(
    unmasked_graph_maze_domain_factory,
    discrete_features,
    my_gnn_class,
    my_gnn_kwargs,
    graphppo_config,
    ray_init,
    caplog,
):
    if discrete_features:
        pytest.skip("Test only for one domain.")
    domain_factory = unmasked_graph_maze_domain_factory
    domain = domain_factory()
    node_features_dim = int(
        np.prod(domain.get_observation_space().unwrapped().node_space.shape)
    )
    gnn_out_dim = 64
    gnn_class = my_gnn_class
    gnn_kwargs = my_gnn_kwargs(
        node_features_dim=node_features_dim, gnn_out_dim=gnn_out_dim
    )

    solver_kwargs = dict(
        algo_class=GraphPPO,
        train_iterations=1,
        graph_feature_extractors_kwargs=dict(
            gnn_class=gnn_class,
            gnn_kwargs=gnn_kwargs,
            gnn_out_dim=gnn_out_dim,
            features_dim=64,
        ),
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        with caplog.at_level(logging.WARNING):
            solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )

    assert gnn_class(**gnn_kwargs).warning() in caplog.text


def test_ppo_user_reduction_layer(
    unmasked_graph_maze_domain_factory,
    discrete_features,
    my_reduction_layer_class,
    my_reduction_layer_kwargs,
    graphppo_config,
    ray_init,
    caplog,
):
    if discrete_features:
        pytest.skip("Test only for one domain.")
    domain_factory = unmasked_graph_maze_domain_factory
    gnn_out_dim = 128
    features_dim = 64
    reduction_layer_class = my_reduction_layer_class
    reduction_layer_kwargs = my_reduction_layer_kwargs(
        gnn_out_dim=gnn_out_dim,
        features_dim=features_dim,
    )
    solver_kwargs = dict(
        algo_class=GraphPPO,
        train_iterations=1,
        graph_feature_extractors_kwargs=dict(
            gnn_out_dim=gnn_out_dim,
            features_dim=features_dim,
            reduction_layer_class=reduction_layer_class,
            reduction_layer_kwargs=reduction_layer_kwargs,
        ),
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        with caplog.at_level(logging.WARNING):
            solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )

    assert reduction_layer_class(**reduction_layer_kwargs).warning() in caplog.text


@pytest.mark.skip(
    "The ray.rllib wrapper does not yet manage graphs with changing structure"
)
def test_dict_ppo(unmasked_jsp_dict_domain_factory, graphppo_config, ray_init):
    domain_factory = unmasked_jsp_dict_domain_factory
    solver_kwargs = dict(
        algo_class=GraphPPO, train_iterations=1  # , gamma=0.95, train_batch_size_log2=8
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert not solver._action_masking and solver._is_graph_multiinput_obs
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )


def test_ppo_masked(graph_maze_domain_factory, graphppo_config, ray_init):
    domain_factory = graph_maze_domain_factory
    solver_kwargs = dict(
        algo_class=GraphPPO, train_iterations=1  # , gamma=0.95, train_batch_size_log2=8
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert solver._action_masking and solver._is_graph_obs
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )


@pytest.mark.skip(
    "The ray.rllib wrapper does not yet manage graphs with changing structure"
)
def test_dict_ppo_masked(unmasked_jsp_dict_domain_factory, graphppo_config, ray_init):
    domain_factory = unmasked_jsp_dict_domain_factory
    solver_kwargs = dict(
        algo_class=GraphPPO, train_iterations=1  # , gamma=0.95, train_batch_size_log2=8
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert solver._action_masking and solver._is_graph_multiinput_obs
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )


def test_ppo_masked_user_gnn(
    graph_maze_domain_factory,
    discrete_features,
    my_gnn_class,
    my_gnn_kwargs,
    graphppo_config,
    ray_init,
    caplog,
):
    if discrete_features:
        pytest.skip("Test only for one domain.")
    domain_factory = graph_maze_domain_factory
    node_features_dim = int(
        np.prod(domain_factory().get_observation_space().unwrapped().node_space.shape)
    )
    gnn_out_dim = 64
    gnn_class = my_gnn_class
    gnn_kwargs = my_gnn_kwargs(
        node_features_dim=node_features_dim, gnn_out_dim=gnn_out_dim
    )

    solver_kwargs = dict(
        algo_class=GraphPPO,
        train_iterations=1,
        graph_feature_extractors_kwargs=dict(
            gnn_class=gnn_class,
            gnn_kwargs=gnn_kwargs,
            gnn_out_dim=gnn_out_dim,
            features_dim=64,
        ),
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert solver._action_masking and solver._is_graph_obs
        with caplog.at_level(logging.WARNING):
            solver.solve()
    assert gnn_class(**gnn_kwargs).warning() in caplog.text


@pytest.mark.skip(
    "The ray.rllib wrapper does not yet manage graphs with changing structure"
)
def test_dict_ppo_masked_user_gnn(
    jsp_dict_domain_factory,
    my_gnn_class,
    my_gnn_kwargs,
    graphppo_config,
    ray_init,
    caplog,
):
    domain_factory = jsp_dict_domain_factory
    node_features_dim = int(
        np.prod(
            domain_factory()
            .get_observation_space()
            .unwrapped()["graph"]
            .node_space.shape
        )
    )
    gnn_out_dim = 64
    gnn_class = my_gnn_class
    gnn_kwargs = my_gnn_kwargs(
        node_features_dim=node_features_dim, gnn_out_dim=gnn_out_dim
    )

    solver_kwargs = dict(
        algo_class=GraphPPO,
        train_iterations=1,
        graph_feature_extractors_kwargs=dict(
            gnn_class=gnn_class,
            gnn_kwargs=gnn_kwargs,
            gnn_out_dim=gnn_out_dim,
            features_dim=64,
        ),
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert solver._action_masking and solver._is_graph_multiinput_obs
        with caplog.at_level(logging.WARNING):
            solver.solve()
    assert gnn_class(**gnn_kwargs).warning() in caplog.text
