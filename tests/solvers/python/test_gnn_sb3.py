import logging

import numpy as np
import torch as th
import torch_geometric as thg

from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.gnn import GraphPPO
from skdecide.hub.solver.stable_baselines.gnn.a2c import GraphA2C
from skdecide.hub.solver.stable_baselines.gnn.common.torch_layers import (
    GraphFeaturesExtractor,
)
from skdecide.hub.solver.stable_baselines.gnn.dqn.dqn import GraphDQN
from skdecide.hub.solver.stable_baselines.gnn.ppo.ppo import Graph2NodePPO
from skdecide.hub.solver.stable_baselines.gnn.ppo_mask import MaskableGraphPPO
from skdecide.hub.solver.stable_baselines.gnn.ppo_mask.ppo_mask import (
    MaskableGraph2NodePPO,
)
from skdecide.utils import rollout


def test_observation_space(graph_domain_factory):
    domain = graph_domain_factory()
    assert domain.reset() in domain.get_observation_space()
    rollout(domain=domain, num_episodes=1, max_steps=3, render=False, verbose=False)


def test_dict_observation_space(jsp_dict_domain_factory):
    domain = jsp_dict_domain_factory()
    assert domain.reset() in domain.get_observation_space()
    rollout(domain=domain, num_episodes=1, max_steps=3, render=False, verbose=False)


def test_ppo(unmasked_graph_domain_factory):
    domain_factory = unmasked_graph_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=GraphPPO,
        baselines_policy="GraphInputPolicy",
        learn_config={"total_timesteps": 100},
    ) as solver:

        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
        )


def test_dqn(unmasked_graph_domain_factory):
    domain_factory = unmasked_graph_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=GraphDQN,
        baselines_policy="GraphInputPolicy",
        learn_config={"total_timesteps": 100},
    ) as solver:

        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
        )


def test_a2c(unmasked_jsp_domain_factory):
    domain_factory = unmasked_jsp_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=GraphA2C,
        baselines_policy="GraphInputPolicy",
        learn_config={"total_timesteps": 100},
    ) as solver:

        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
        )


def test_ppo_user_gnn(caplog, unmasked_jsp_domain_factory, my_gnn_class, my_gnn_kwargs):
    domain_factory = unmasked_jsp_domain_factory
    domain = domain_factory()
    node_features_dim = int(
        np.prod(domain.get_observation_space().unwrapped().node_space.shape)
    )
    gnn_out_dim = 64
    gnn_class = my_gnn_class
    gnn_kwargs = my_gnn_kwargs(
        node_features_dim=node_features_dim, gnn_out_dim=gnn_out_dim
    )
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=GraphPPO,
        baselines_policy="GraphInputPolicy",
        learn_config={"total_timesteps": 100},
        policy_kwargs=dict(
            features_extractor_class=GraphFeaturesExtractor,
            features_extractor_kwargs=dict(
                gnn_class=gnn_class,
                gnn_kwargs=gnn_kwargs,
                gnn_out_dim=gnn_out_dim,
                features_dim=64,
            ),
        ),
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
    assert gnn_class(**gnn_kwargs).warning() in caplog.text


def test_ppo_user_reduction_layer(
    caplog,
    unmasked_jsp_domain_factory,
    my_reduction_layer_class,
    my_reduction_layer_kwargs,
):
    domain_factory = unmasked_jsp_domain_factory
    gnn_out_dim = 128
    features_dim = 64
    reduction_layer_class = my_reduction_layer_class
    reduction_layer_kwargs = my_reduction_layer_kwargs(
        gnn_out_dim=gnn_out_dim,
        features_dim=features_dim,
    )
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=GraphPPO,
        baselines_policy="GraphInputPolicy",
        learn_config={"total_timesteps": 100},
        policy_kwargs=dict(
            features_extractor_class=GraphFeaturesExtractor,
            features_extractor_kwargs=dict(
                gnn_out_dim=gnn_out_dim,
                features_dim=features_dim,
                reduction_layer_class=reduction_layer_class,
                reduction_layer_kwargs=reduction_layer_kwargs,
            ),
        ),
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


def test_maskable_ppo(graph_domain_factory):
    domain_factory = graph_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=MaskableGraphPPO,
        baselines_policy="GraphInputPolicy",
        learn_config={"total_timesteps": 100},
        use_action_masking=True,
    ) as solver:

        solver.solve()
        episodes = rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
            use_applicable_actions=True,
            return_episodes=True,
        )

    if "Jsp" in domain_factory().__class__.__name__:
        # with masking only 9 steps necessary since only 9 tasks to perform
        observations, actions, values = episodes[0]
        assert len(actions) == 9


def test_dict_ppo(unmasked_jsp_dict_domain_factory):
    domain_factory = unmasked_jsp_dict_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=GraphPPO,
        baselines_policy="MultiInputPolicy",
        learn_config={"total_timesteps": 100},
    ) as solver:

        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
        )


def test_dict_maskable_ppo(jsp_dict_domain_factory):
    domain_factory = jsp_dict_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=MaskableGraphPPO,
        baselines_policy="MultiInputPolicy",
        learn_config={"total_timesteps": 100},
        use_action_masking=True,
    ) as solver:

        solver.solve()
        episodes = rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
            use_applicable_actions=True,
            return_episodes=True,
        )
    # with masking only 9 steps necessary since only 9 tasks to perform
    observations, actions, values = episodes[0]
    assert len(actions) == 9


def test_dict_a2c(unmasked_jsp_dict_domain_factory):
    domain_factory = unmasked_jsp_dict_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=GraphA2C,
        baselines_policy="MultiInputPolicy",
        learn_config={"total_timesteps": 100},
    ) as solver:
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
        )


def test_dict_dqn(unmasked_jsp_dict_domain_factory):
    domain_factory = unmasked_jsp_dict_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=GraphDQN,
        baselines_policy="MultiInputPolicy",
        learn_config={"total_timesteps": 100},
    ) as solver:

        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
        )


def test_graph2node_ppo(unmasked_jsp_domain_factory, variable_n_nodes):

    domain_factory = unmasked_jsp_domain_factory

    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=Graph2NodePPO,
        baselines_policy="GraphInputPolicy",
        learn_config={
            "total_timesteps": 200,
        },
        n_steps=100,
    ) as solver:
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
        )

        # check batch handling by policy  (only once)
        if not variable_n_nodes:
            domain = domain_factory()
            obs = domain.reset()
            policy = solver._algo.policy
            obs_thg_data, _ = policy.obs_to_tensor(obs)
            # set training=False to be deterministic
            policy.train(False)
            # batch with same obs => same logits
            obs_batch = thg.data.Batch.from_data_list([obs_thg_data, obs_thg_data])
            batched_logits = policy.get_distribution(obs_batch).distribution.logits
            assert th.allclose(batched_logits[0], batched_logits[1])
            # batch with an obs with less node => last logit ~= -inf
            x = obs_thg_data.x[:-1, :]
            edge_index, edge_attr = thg.utils.subgraph(
                subset=list(range(len(obs_thg_data.x) - 1)),
                edge_index=obs_thg_data.edge_index,
                edge_attr=obs_thg_data.edge_attr,
            )
            obs_thg_data2 = thg.data.Data(
                x=x, edge_attr=edge_attr, edge_index=edge_index
            )
            obs_batch = thg.data.Batch.from_data_list([obs_thg_data, obs_thg_data2])
            last_prob = th.exp(policy.get_distribution(obs_batch).distribution.logits)[
                -1, -1
            ]
            assert last_prob == 0.0


def test_maskable_graph2node_ppo(jsp_graph2node_domain_factory):

    domain_factory = jsp_graph2node_domain_factory

    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=MaskableGraph2NodePPO,
        baselines_policy="GraphInputPolicy",
        learn_config={
            "total_timesteps": 200,
        },
        n_steps=100,
        use_action_masking=True,
    ) as solver:
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
