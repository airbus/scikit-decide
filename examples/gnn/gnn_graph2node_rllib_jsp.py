import numpy as np
from domains import UnmaskedGraphJspDomain
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.hub.solver.ray_rllib.gnn.algorithms.ppo import GraphPPO
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


domain_factory = lambda: UnmaskedGraphJspDomain(
    gym_env=DisjunctiveGraphJspEnv(
        jps_instance=jsp,
        perform_left_shift_if_possible=True,
        normalize_observation_space=False,
        flat_observation_space=False,
        action_mode="task",
    )
)

# Uncomment line below to run locally and be able to debug
# ray.init(local_mode=True)

graphppo_config = GraphPPO.get_default_config()
with RayRLlib(
    domain_factory=domain_factory,
    config=graphppo_config,
    algo_class=GraphPPO,
    train_iterations=1,
    graph_node_action=True,
) as solver:
    solver.solve()
    rollout(
        domain=domain_factory(),
        solver=solver,
        max_steps=100,
        num_episodes=1,
        render=True,
    )
