import numpy as np
from domains import GraphJspDomain
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.gnn.ppo_mask.ppo_mask import (
    MaskableGraph2NodePPO,
)
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


# Uncomment the block below to use PPO without action masking
# with StableBaseline(
#     domain_factory=domain_factory,
#     algo_class=Graph2NodePPO,
#     baselines_policy="GraphInputPolicy",
#     policy_kwargs=dict(debug=True),
#     learn_config={
#         "total_timesteps": 10_000,
#     },
# ) as solver:
#     solver.solve()
#     rollout(
#         domain=domain_factory(),
#         solver=solver,
#         max_steps=30,
#         num_episodes=1,
#         render=True,
#     )

# PPO graph -> node + action masking
with StableBaseline(
    domain_factory=domain_factory,
    algo_class=MaskableGraph2NodePPO,
    baselines_policy="GraphInputPolicy",
    policy_kwargs=dict(debug=True),
    learn_config={
        "total_timesteps": 10_000,
    },
    use_action_masking=True,
) as solver:
    solver.solve()
    rollout(
        domain=domain_factory(),
        solver=solver,
        max_steps=30,
        num_episodes=1,
        render=True,
    )
