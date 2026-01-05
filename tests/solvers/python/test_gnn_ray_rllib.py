import logging
import os

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


def test_ppo(unmasked_graph_domain_factory, graphppo_config, ray_init):
    domain_factory = unmasked_graph_domain_factory
    solver_kwargs = dict(algo_class=GraphPPO, train_iterations=1)
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert not solver._action_masking and solver._is_graph_obs
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
        )


def test_ppo_user_gnn(
    unmasked_jsp_domain_factory,
    my_gnn_class,
    my_gnn_kwargs,
    graphppo_config,
    ray_init,
    caplog,
):
    domain_factory = unmasked_jsp_domain_factory
    gnn_out_dim = 64
    gnn_class = my_gnn_class
    gnn_kwargs = my_gnn_kwargs(gnn_out_dim=gnn_out_dim)

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
            max_steps=30,
            num_episodes=1,
            render=False,
        )

    assert gnn_class(in_channels=1, **gnn_kwargs).warning() in caplog.text


def test_ppo_user_reduction_layer(
    unmasked_jsp_domain_factory,
    my_reduction_layer_class,
    my_reduction_layer_kwargs,
    graphppo_config,
    ray_init,
    caplog,
):
    domain_factory = unmasked_jsp_domain_factory
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
            max_steps=30,
            num_episodes=1,
            render=False,
        )

    assert reduction_layer_class(**reduction_layer_kwargs).warning() in caplog.text


def test_dict_ppo(unmasked_jsp_dict_domain_factory, graphppo_config, ray_init):
    domain_factory = unmasked_jsp_dict_domain_factory
    solver_kwargs = dict(algo_class=GraphPPO, train_iterations=1)
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert not solver._action_masking and solver._is_graph_multiinput_obs
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
        )


def test_ppo_masked(graph_domain_factory, graphppo_config, ray_init):
    domain_factory = graph_domain_factory
    solver_kwargs = dict(algo_class=GraphPPO, train_iterations=1)
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert solver._action_masking and solver._is_graph_obs
        solver.solve()
        episodes = rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
            return_episodes=True,
        )
    if "Jsp" in domain_factory().__class__.__name__:
        # with masking only 9 steps necessary since only 9 tasks to perform
        observations, actions, values = episodes[0]
        assert len(actions) == 9


def test_dict_ppo_masked(jsp_dict_domain_factory, graphppo_config, ray_init):
    domain_factory = jsp_dict_domain_factory
    solver_kwargs = dict(algo_class=GraphPPO, train_iterations=1)
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert solver._action_masking and solver._is_graph_multiinput_obs
        solver.solve()
        episodes = rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
            return_episodes=True,
        )
    # with masking only 9 steps necessary since only 9 tasks to perform
    observations, actions, values = episodes[0]
    assert len(actions) == 9


def test_ppo_masked_user_gnn(
    jsp_domain_factory,
    my_gnn_class,
    my_gnn_kwargs,
    graphppo_config,
    ray_init,
    caplog,
):
    domain_factory = jsp_domain_factory
    gnn_out_dim = 64
    gnn_class = my_gnn_class
    gnn_kwargs = my_gnn_kwargs(gnn_out_dim=gnn_out_dim)

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
    assert gnn_class(in_channels=1, **gnn_kwargs).warning() in caplog.text


def test_dict_ppo_masked_user_gnn(
    jsp_dict_domain_factory,
    my_gnn_class,
    my_gnn_kwargs,
    graphppo_config,
    ray_init,
    caplog,
):
    domain_factory = jsp_dict_domain_factory
    gnn_out_dim = 64
    gnn_class = my_gnn_class
    gnn_kwargs = my_gnn_kwargs(gnn_out_dim=gnn_out_dim)

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
    assert gnn_class(in_channels=1, **gnn_kwargs).warning() in caplog.text


def test_graph2node_ppo(
    unmasked_jsp_domain_factory,
    graphppo_config,
    ray_init,
):
    domain_factory = unmasked_jsp_domain_factory
    solver_kwargs = dict(
        algo_class=GraphPPO,
        train_iterations=1,
        graph_node_action=True,
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert (
            not solver._action_masking and solver._is_graph_obs and solver._graph2node
        )
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
        )


def test_maskable_graph2node_ppo(
    jsp_graph2node_domain_factory,
    graphppo_config,
    ray_init,
):
    domain_factory = jsp_graph2node_domain_factory
    solver_kwargs = dict(
        algo_class=GraphPPO,
        train_iterations=1,
        graph_node_action=True,
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert solver._action_masking and solver._is_graph_obs and solver._graph2node
        solver.solve()
        episodes = rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
            return_episodes=True,
        )
    # with masking only 9 steps necessary since only 9 tasks to perform
    observations, actions, values = episodes[0]
    assert len(actions) == 9
