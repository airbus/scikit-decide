#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example using OPTUNA to choose a solving method and tune its hyperparameters for maze.

Here we choose to optimize the number of nodes explored. (No rollout needed.)

Results can be viewed on optuna-dashboard with:

    optuna-dashboard optuna-journal.log

"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple, Type

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    Hyperparameter,
)
from stable_baselines3 import A2C, PPO

from skdecide import Solver
from skdecide.core import D, Value
from skdecide.hub.domain.maze.maze import Maze, State
from skdecide.hub.solver.astar import Astar
from skdecide.hub.solver.iw import IW
from skdecide.hub.solver.mcts import MCTS, UCT
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.optuna_utils import generic_optuna_experiment_monoproblem
from skdecide.utils import match_solvers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


# Params of the study
rollout_num_episodes = 0  # no rollout needed
rollout_max_steps_by_episode = 500
seed = 42  # set this to an integer to get reproducible results, else to None
n_trials = 10  # number of trials to launch
create_another_study = True
overwrite_study = False
study_basename = f"maze-explored-nodes"
domain_reset_is_deterministic = (
    True  # set this to True when the domain reset() is known to be deterministic
)
# => avoid repeating rollout when everything (domain + solver) is deterministic


# Domain to test
domain_factory = Maze


# Solvers to test
# solver_classes = match_solvers(domain_factory())
solver_classes = [Astar, IW, UCT]
solver_classes = match_solvers(domain=domain_factory(), candidates=solver_classes)


# heuristics and state_features (needed by some solvers
def euclidean_heuristic(domain: Maze, state: State):
    return Value(
        cost=math.sqrt(
            (domain._goal.x - state.x) ** 2 + (domain._goal.y - state.y) ** 2
        )
    )


def manhattan_heuristic(domain: Maze, state: State):
    return Value(
        cost=math.fabs(domain._goal.x - state.x) + math.fabs(domain._goal.y - state.y)
    )


def state_features(domain: Maze, state: State):
    return state.x, state.y


# Fixed kwargs per solver: either hyperparameters we do not want to search, or other parameters like time limits
kwargs_fixed_by_solver: Dict[Type[Solver], Dict[str, Any]] = {
    Astar: dict(
        parallel=False,
        verbose=False,
    ),
    UCT: dict(
        time_budget=200,
        rollout_budget=100000,
        online_node_garbage=True,
        max_depth=500,
        residual_moving_average_window=100,
        epsilon=0.0,
        discount=1.0,
        ucb_constant=1.0 / math.sqrt(2.0),
        parallel=False,
        verbose=False,
        rollout_policy=MCTS.RolloutPolicy.RANDOM,
        transition_mode=MCTS.TransitionMode.DISTRIBUTION,
        continuous_planning=True,
    ),
    IW: dict(
        state_features=state_features,
        node_ordering=lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: a_novelty
        > b_novelty,
        parallel=False,
        verbose=False,
    ),
}

# Add new hyperparameters to some solvers
additional_hyperparameters_by_solver: Dict[Type[Solver], List[Hyperparameter]] = {
    # ex: heuristic for A* and co
    Astar: [
        CategoricalHyperparameter(
            name="heuristic",
            choices={  # associate a label to an actual function (for optuna)
                "euclidean": euclidean_heuristic,
                "manhattan": manhattan_heuristic,
            },
        )
    ],
    UCT: [
        CategoricalHyperparameter(
            name="heuristic",
            choices={  # associate a label to an actual function (for optuna)
                "euclidean": lambda d, s: (euclidean_heuristic(d, s), 10000),
                "manhattan": lambda d, s: (manhattan_heuristic(d, s), 10000),
            },
        )
    ],
}


# Restrict some hyperparameters choices, for some solvers (making use of `kwargs_by_name` of `suggest_with_optuna`)
suggest_optuna_kwargs_by_name_by_solver: Dict[
    Type[Solver], Dict[str, Dict[str, Any]]
] = {
    StableBaseline: {
        # restrict the choices of algo classes
        "algo_class": dict(
            choices={
                "A2C": A2C,
                "PPO": PPO,
            }
        )
    }
}


def objective(
    solver: Solver,
    episodes: List[
        Tuple[
            List[D.T_agent[D.T_observation]],
            List[D.T_agent[D.T_concurrency[D.T_event]]],
            List[D.T_agent[Value[D.T_value]]],
        ]
    ],
) -> float:
    return solver.get_nb_explored_states()


optuna_tuning_direction = "minimize"


# Create and launch the optuna study
generic_optuna_experiment_monoproblem(
    domain_factory=domain_factory,
    solver_classes=solver_classes,
    kwargs_fixed_by_solver=kwargs_fixed_by_solver,
    suggest_optuna_kwargs_by_name_by_solver=suggest_optuna_kwargs_by_name_by_solver,
    additional_hyperparameters_by_solver=additional_hyperparameters_by_solver,
    n_trials=n_trials,
    rollout_num_episodes=rollout_num_episodes,
    rollout_max_steps_by_episode=rollout_max_steps_by_episode,
    domain_reset_is_deterministic=domain_reset_is_deterministic,
    study_basename=study_basename,
    create_another_study=create_another_study,
    overwrite_study=overwrite_study,
    seed=seed,
    objective=objective,
    optuna_tuning_direction=optuna_tuning_direction,
)
