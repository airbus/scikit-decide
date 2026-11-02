import os
from tempfile import TemporaryDirectory

import numpy as np
import ray
from pytest_cases import fixture
from ray.rllib.algorithms import PPO
from ray.rllib.core.models.torch.encoder import TorchActorCriticEncoder
from ray.rllib.core.models.torch.heads import TorchMLPHead

from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.hub.solver.ray_rllib.autoregressive.algorithms.ppo.torch.autoregressive_ppo_torch_rl_module import (
    AutoregressivePPOTorchRLModule,
)
from skdecide.hub.solver.ray_rllib.gnn.models.torch.encoder import TorchGnnEncoder
from skdecide.hub.solver.ray_rllib.gnn.models.torch.graph2node import TorchGraph2Node
from skdecide.hub.solver.ray_rllib.ray_rllib import SK_DEFAULT_MODULE_ID
from skdecide.hub.solver.utils.torch.utils import extract_module_parameters_values
from skdecide.utils import rollout


@fixture
def ray_init():
    # add module test_gnn_ray_rllib and thus GraphMaze to ray runtimeenv
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"working_dir": os.path.abspath(f"{os.path.dirname(__file__)}/..")},
    )


@fixture
def ppo_config():
    return (
        PPO.get_default_config()
        .env_runners(
            # set num of CPU<1 to avoid hanging for ever in github actions on macos 11)
            num_cpus_per_env_runner=0.5
        )
        # small number to increase speed of the unit test
        .training(minibatch_size=16, train_batch_size_per_learner=32)
        # uncomment next lines to run in local mode and debug more easily
        # .env_runners(num_env_runners=0)
        # .learners(num_learners=0)
    )


def test_autoregressive_ppo(graph_walk_domain_factory, ppo_config, ray_init):
    domain_factory = graph_walk_domain_factory
    with RayRLlib(
        domain_factory=domain_factory,
        algo_class=PPO,
        config=ppo_config,
        train_iterations=1,
        autoregressive_action=True,
    ) as solver:
        # store initial weigths to see if they update during training
        solver._init_algo()
        rl_module = solver.get_policy()[SK_DEFAULT_MODULE_ID]
        assert isinstance(rl_module, AutoregressivePPOTorchRLModule)
        value_init_params = extract_module_parameters_values(rl_module.vf)
        action_init_params = [
            extract_module_parameters_values(pi_component)
            for pi_component in rl_module.pi_per_component
        ]
        solver.solve()
        # rl_module = solver.get_policy()[SK_DEFAULT_MODULE_ID]
        # assert isinstance(rl_module, AutoregressivePPOTorchRLModule)
        value_final_params = extract_module_parameters_values(rl_module.vf)
        action_final_params = [
            extract_module_parameters_values(pi_component)
            for pi_component in rl_module.pi_per_component
        ]
        max_steps = 20
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=max_steps,
            num_episodes=1,
            render=False,
        )

        # check policy params updated
        assert not all(
            all(
                np.allclose(action_init_params[i][name], action_final_params[i][name])
                for name in action_init_params[i]
            )
            for i in range(len(action_init_params))
        ), f"action net params not updated"
        # assert not (
        #     all(
        #         np.allclose(value_init_params[name], value_final_params[name])
        #         for name in value_init_params
        #     )
        # ), f"value net params not updated"


def test_autoregressive_graph_ppo(
    graph_walk_with_graph_obs_domain_factory, ppo_config, ray_init
):
    domain_factory = graph_walk_with_graph_obs_domain_factory
    with RayRLlib(
        domain_factory=domain_factory,
        algo_class=PPO,
        config=ppo_config,
        train_iterations=1,
        autoregressive_action=True,
    ) as solver:
        # pre-init algo (done normally during solve()) to extract init weights
        solver._init_algo()
        # store weigths to see if they update during training
        rl_module = solver.get_policy()
        agent_rl_module = rl_module.get(SK_DEFAULT_MODULE_ID)
        assert isinstance(agent_rl_module, AutoregressivePPOTorchRLModule)
        assert isinstance(agent_rl_module.encoder, TorchActorCriticEncoder)
        assert isinstance(agent_rl_module.encoder.actor_encoder, TorchGnnEncoder)
        action_init_params = [
            extract_module_parameters_values(pi_component)
            for pi_component in agent_rl_module.pi_per_component
        ]
        gnn_encoder_init_params = extract_module_parameters_values(
            agent_rl_module.encoder.actor_encoder
        )
        # solve
        solver.solve()
        # compare final parameters to see if actually updated during training
        action_final_params = [
            extract_module_parameters_values(pi_component)
            for pi_component in agent_rl_module.pi_per_component
        ]
        gnn_encoder_final_params = extract_module_parameters_values(
            agent_rl_module.encoder.actor_encoder
        )
        assert not (
            all(
                np.allclose(
                    gnn_encoder_init_params[name], gnn_encoder_final_params[name]
                )
                for name in gnn_encoder_final_params
            )
        ), f"graph feature extractor params not updated"
        assert not all(
            all(
                np.allclose(action_init_params[i][name], action_final_params[i][name])
                for name in action_init_params[i]
            )
            for i in range(len(action_init_params))
        ), f"action net params not updated"

        # rollout
        max_steps = 20
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=max_steps,
            num_episodes=1,
            render=False,
            verbose=False,
        )


def test_autoregressive_graph2node_ppo(
    graph_walk_with_graph_obs_domain_factory, ppo_config, ray_init
):
    domain_factory = graph_walk_with_graph_obs_domain_factory
    with RayRLlib(
        domain_factory=domain_factory,
        algo_class=PPO,
        config=ppo_config,
        train_iterations=1,
        autoregressive_action=True,
        autoregressive_action_nb_graph_independent_components=1,
    ) as solver:
        # pre-init algo (done normally during solve()) to extract init weights
        solver._init_algo()
        # store weigths to see if they update during training
        rl_module = solver.get_policy()
        agent_rl_module = rl_module.get(SK_DEFAULT_MODULE_ID)
        assert isinstance(agent_rl_module, AutoregressivePPOTorchRLModule)
        assert isinstance(agent_rl_module.encoder, TorchActorCriticEncoder)
        assert isinstance(agent_rl_module.encoder.actor_encoder, TorchGnnEncoder)
        for i in range(len(domain_factory()._gym_env.action_space.nvec)):
            if i == 0:
                assert isinstance(agent_rl_module.pi_per_component[i], TorchMLPHead)
            else:
                assert isinstance(agent_rl_module.pi_per_component[i], TorchGraph2Node)
        action_init_params = [
            extract_module_parameters_values(pi_component)
            for pi_component in agent_rl_module.pi_per_component
        ]
        gnn_encoder_init_params = extract_module_parameters_values(
            agent_rl_module.encoder.actor_encoder
        )
        # solve
        solver.solve()
        # compare final parameters to see if actually updated during training
        action_final_params = [
            extract_module_parameters_values(pi_component)
            for pi_component in agent_rl_module.pi_per_component
        ]
        gnn_encoder_final_params = extract_module_parameters_values(
            agent_rl_module.encoder.actor_encoder
        )
        assert not (
            all(
                np.allclose(
                    gnn_encoder_init_params[name], gnn_encoder_final_params[name]
                )
                for name in gnn_encoder_final_params
            )
        ), f"value net params not updated"
        assert not all(
            all(
                np.allclose(action_init_params[i][name], action_final_params[i][name])
                for name in action_init_params[i]
            )
            for i in range(len(action_init_params))
        ), f"action net params not updated"

        # rollout
        max_steps = 20
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=max_steps,
            num_episodes=1,
            render=False,
        )


def test_autoregressive_heterograph2node_ppo(
    graph_walk_with_heterograph_obs_domain_factory_3, ppo_config, ray_init
):
    domain_factory = graph_walk_with_heterograph_obs_domain_factory_3
    action_components_node_flag_indices = (
        domain_factory().action_components_node_flag_indices
    )
    with RayRLlib(
        domain_factory=domain_factory,
        algo_class=PPO,
        config=ppo_config,
        train_iterations=1,
        autoregressive_action=True,
        autoregressive_action_components_node_flag_indices=action_components_node_flag_indices,
    ) as solver:
        # pre-init algo (done normally during solve()) to extract init weights
        solver._init_algo()

        # store weigths to see if they update during training
        rl_module = solver.get_policy()
        agent_rl_module = rl_module.get(SK_DEFAULT_MODULE_ID)
        assert isinstance(agent_rl_module, AutoregressivePPOTorchRLModule)
        assert isinstance(agent_rl_module.encoder, TorchActorCriticEncoder)
        assert isinstance(agent_rl_module.encoder.actor_encoder, TorchGnnEncoder)
        for i in range(len(domain_factory()._gym_env.action_space.nvec)):
            if i == 0:
                assert isinstance(agent_rl_module.pi_per_component[i], TorchMLPHead)
            else:
                assert isinstance(agent_rl_module.pi_per_component[i], TorchGraph2Node)
        action_init_params = [
            extract_module_parameters_values(pi_component)
            for pi_component in agent_rl_module.pi_per_component
        ]
        gnn_encoder_init_params = extract_module_parameters_values(
            agent_rl_module.encoder.actor_encoder
        )

        # solve
        solver.solve()

        # compare final parameters to see if actually updated during training
        action_final_params = [
            extract_module_parameters_values(pi_component)
            for pi_component in agent_rl_module.pi_per_component
        ]
        gnn_encoder_final_params = extract_module_parameters_values(
            agent_rl_module.encoder.actor_encoder
        )
        assert not (
            all(
                np.allclose(
                    gnn_encoder_init_params[name], gnn_encoder_final_params[name]
                )
                for name in gnn_encoder_final_params
            )
        ), f"graph feature extractor params not updated"
        assert not all(
            all(
                np.allclose(action_init_params[i][name], action_final_params[i][name])
                for name in action_init_params[i]
            )
            for i in range(len(action_init_params))
        ), f"action net params not updated"

        # rollout
        max_steps = 20
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=max_steps,
            num_episodes=1,
            render=False,
        )


def test_autoregressive_heterograph2node_ppo_save_load(
    graph_walk_with_heterograph_obs_domain_factory,
    graph_walk_with_heterograph_obs_domain_factory_2,
    ppo_config,
    ray_init,
):
    domain_factory = graph_walk_with_heterograph_obs_domain_factory
    domain_factory_2 = graph_walk_with_heterograph_obs_domain_factory_2
    action_components_node_flag_indices = (
        domain_factory().action_components_node_flag_indices
    )

    with TemporaryDirectory() as directory:
        path = f"{directory}/sb3"
        with RayRLlib(
            domain_factory=domain_factory,
            algo_class=PPO,
            config=ppo_config,
            train_iterations=1,
            autoregressive_action=True,
            autoregressive_action_components_node_flag_indices=action_components_node_flag_indices,
        ) as solver:
            # solve
            solver.solve()
            # store
            solver.save(path)

        # load for another domain (more objects) + solve + rollout
        with RayRLlib(
            domain_factory=domain_factory_2,
            algo_class=PPO,
            config=ppo_config,
            train_iterations=1,
            autoregressive_action=True,
            autoregressive_action_components_node_flag_indices=action_components_node_flag_indices,
        ) as solver:
            # load
            solver.load(path)

            # rollout on domain 2
            max_steps = 20
            rollout(
                domain=graph_walk_with_heterograph_obs_domain_factory_2(),
                solver=solver,
                max_steps=max_steps,
                num_episodes=1,
                render=False,
            )
