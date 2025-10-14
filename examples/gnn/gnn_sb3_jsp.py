import numpy as np
from domains import GraphJspDomain
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.gnn import GraphPPO
from skdecide.hub.solver.stable_baselines.gnn.ppo_mask import MaskableGraphPPO
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


jsp_env = DisjunctiveGraphJspEnv(
    jps_instance=jsp,
    perform_left_shift_if_possible=True,
    normalize_observation_space=False,
    flat_observation_space=False,
    action_mode="task",
)

# random rollout
domain = GraphJspDomain(gym_env=jsp_env)
rollout(domain=domain, max_steps=jsp_env.total_tasks_without_dummies, num_episodes=1)

# solve with sb3-GraphPPO
domain_factory = lambda: GraphJspDomain(gym_env=jsp_env)
with StableBaseline(
    domain_factory=domain_factory,
    algo_class=GraphPPO,
    baselines_policy="GraphInputPolicy",
    learn_config={"total_timesteps": 100},
) as solver:
    solver.solve()
    rollout(domain=domain_factory(), solver=solver, max_steps=100, num_episodes=1)

# solver with sb3-MaskableGraphPPO
domain_factory = lambda: GraphJspDomain(gym_env=jsp_env)
with StableBaseline(
    domain_factory=domain_factory,
    algo_class=MaskableGraphPPO,
    baselines_policy="GraphInputPolicy",
    learn_config={"total_timesteps": 100},
    use_action_masking=True,
) as solver:
    solver.solve()
    rollout(
        domain=domain_factory(),
        solver=solver,
        max_steps=100,
        num_episodes=1,
        use_applicable_actions=True,
    )
