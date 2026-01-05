from domains import GraphMaze

from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.gnn import GraphPPO
from skdecide.hub.solver.stable_baselines.gnn.ppo_mask import MaskableGraphPPO
from skdecide.utils import rollout

MAZE = """
+-+-+-+-+o+-+-+--+-+-+
|   |              | |
+ + + +-+-+-+ +--+ + +
| | |   |   | |  |   |
+ +-+-+ +-+ + + -+ +-+
| |   |   | |    |   |
+ + + + + + + +--+ +-+
|   |   |   | |      |
+-+-+-+-+-+-+-+ -+-+ +
|             |    | |
+ +-+-+-+-+ + +--+-+ +
|   |       |        |
+ + + +-+ +-+ +--+-+-+
| | |   |     |      |
+ +-+-+ + +-+ + -+-+ +
| |     | | | |    | |
+-+ +-+ + + + +--+ + +
|   |   |   |    | | |
+ +-+ +-+-+-+-+ -+ + +
|   |       |      | |
+-+-+-+-+-+x+-+--+-+-+
"""

domain = GraphMaze(maze_str=MAZE, discrete_features=True)
assert domain.reset() in domain.get_observation_space()

# random rollout
rollout(domain=domain, max_steps=50, num_episodes=1)

# solve with sb3-PPO-GNN
domain_factory = lambda: GraphMaze(maze_str=MAZE)
max_steps = domain.maze_domain._num_cols * domain.maze_domain._num_rows
with StableBaseline(
    domain_factory=domain_factory,
    algo_class=GraphPPO,
    baselines_policy="GraphInputPolicy",
    learn_config={"total_timesteps": 100},
) as solver:
    solver.solve()
    rollout(domain=domain_factory(), solver=solver, max_steps=max_steps, num_episodes=1)

# solver with sb3-MaskableGraphPPO
domain_factory = lambda: GraphMaze(maze_str=MAZE)
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
