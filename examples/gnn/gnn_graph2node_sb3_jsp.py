from typing import Any

import numpy as np
from domains import GraphJspDomain
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from gymnasium.spaces import Box, Graph, GraphInstance

from skdecide.builders.domain import (
    FullyObservable,
    Initializable,
    Markovian,
    Renderable,
    Rewards,
    Sequential,
    SingleAgent,
)
from skdecide.core import Space, TransitionOutcome, Value
from skdecide.domains import Domain
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.gnn.ppo.ppo import Graph2NodePPO
from skdecide.hub.solver.utils.gnn.torch_utils import extract_module_parameters_values
from skdecide.hub.space.gym import GymSpace, ListSpace
from skdecide.utils import rollout

jsp = np.array(
    [
        [
            [0, 1, 2],  # machines for job 0
            [0, 2, 1],  # machines for job 1
            [0, 1, 2],  # machines for job 2
        ],
        [
            [3, 2, 2],  # task durations of job 0
            [2, 1, 4],  # task durations of job 1
            [0, 4, 3],  # task durations of job 2
        ],
    ]
)


domain_factory = lambda: GraphJspDomain(
    gym_env=DisjunctiveGraphJspEnv(
        jps_instance=jsp,
        perform_left_shift_if_possible=True,
        normalize_observation_space=False,
        flat_observation_space=False,
        action_mode="task",
    )
)


with StableBaseline(
    domain_factory=domain_factory,
    algo_class=Graph2NodePPO,
    baselines_policy="GraphInputPolicy",
    policy_kwargs=dict(debug=True),
    learn_config={
        "total_timesteps": 10_000,
    },
) as solver:
    solver.solve()
    rollout(
        domain=domain_factory(),
        solver=solver,
        max_steps=30,
        num_episodes=1,
        render=True,
    )


# action gnn parameters
initial_parameters = solver._algo.policy.action_net.initial_parameters
final_parameters = extract_module_parameters_values(solver._algo.policy.action_net)
same_parameters: dict[str, bool] = {
    name: (initial_parameters[name] == final_parameters[name]).all()
    for name in final_parameters
}

if all(same_parameters.values()):
    print("Action full GNN parameters have not changed during training!")
else:
    unchanging_parameters = [name for name, same in same_parameters.items() if same]
    print(
        f"Action full GNN parameter unchanged after training: {unchanging_parameters}"
    )
    changing_parameters = [name for name, same in same_parameters.items() if not same]
    print(
        f"Action full GNN parameters having changed during training: {changing_parameters}"
    )
    diff_parameters = {
        name: abs(initial_parameters[name] - final_parameters[name]).max()
        for name in changing_parameters
    }
    print(diff_parameters)

# value gnn parameters
initial_parameters = solver._algo.policy.features_extractor.extractor.initial_parameters
final_parameters = extract_module_parameters_values(
    solver._algo.policy.features_extractor.extractor
)
same_parameters: dict[str, bool] = {
    name: (initial_parameters[name] == final_parameters[name]).all()
    for name in final_parameters
}

if all(same_parameters.values()):
    print("Value GNN feature extractor parameters have not changed during training!")
else:
    unchanging_parameters = [name for name, same in same_parameters.items() if same]
    print(
        f"Value GNN feature extracto parameter unchanged after training: {unchanging_parameters}"
    )
    changing_parameters = [name for name, same in same_parameters.items() if not same]
    print(
        f"Value GNN feature extractor parameters having changed during training: {changing_parameters}"
    )
    diff_parameters = {
        name: abs(initial_parameters[name] - final_parameters[name]).max()
        for name in changing_parameters
    }
    print(diff_parameters)
