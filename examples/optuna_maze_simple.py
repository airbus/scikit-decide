#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example using OPTUNA to choose a solving method and tune its hyperparameters for maze.

Simplest version, without using any fancy options.

Results can be viewed on optuna-dashboard with:

    optuna-dashboard optuna-journal.log

"""

from __future__ import annotations

import logging

from skdecide.hub.domain.maze.maze import Maze
from skdecide.hub.solver.astar import Astar
from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.optuna_utils import generic_optuna_experiment_monoproblem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


# Params of the study
n_trials = 100  # number of trials to launch
study_basename = "maze-simple"  # base name for the optuna study
domain_reset_is_deterministic = (
    True  # set this to True when the domain reset() is known to be deterministic
)
# => avoid repeating rollout when everything (domain + solver) is deterministic

# Domain to test
domain_factory = Maze

# Solvers to test
solver_classes = [Astar, StableBaseline, RayRLlib]

# Parameters for solvers initialization
kwargs_fixed_by_solver = {
    # only needed by StableBaseline
    StableBaseline: dict(baselines_policy="MlpPolicy", total_timesteps=1000),
}


# Create and launch the optuna study
generic_optuna_experiment_monoproblem(
    domain_factory=domain_factory,
    solver_classes=solver_classes,
    n_trials=n_trials,
    domain_reset_is_deterministic=domain_reset_is_deterministic,
    study_basename=f"maze-simple",
    kwargs_fixed_by_solver=kwargs_fixed_by_solver,
)
