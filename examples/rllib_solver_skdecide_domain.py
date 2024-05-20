# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ray.rllib.algorithms.ppo import PPO

from skdecide.hub.domain.simple_grid_world import SimpleGridWorld
from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.utils import rollout

# This example shows how to solve the simple grid world domain using RLlib's PPO

domain_factory = lambda: SimpleGridWorld(num_cols=10, num_rows=10)
domain = domain_factory()

# Check domain compatibility
if RayRLlib.check_domain(domain):
    solver_factory = lambda: RayRLlib(
        domain_factory=domain_factory, algo_class=PPO, train_iterations=5
    )

    # Start solving
    with solver_factory() as solver:
        solver.solve()

        # Test solution
        rollout(
            domain,
            solver,
            num_episodes=1,
            max_steps=100,
            max_framerate=30,
            outcome_formatter=None,
        )
