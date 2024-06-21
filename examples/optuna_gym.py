#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example using OPTUNA to choose a solving method and tune its hyperparameters for a gym domain.

We demonstrate some options that allow:
- adding new hyperparameters for the study (e.g. heuristic for A*)
- fixing some hyperparameters
- restricting hyperparameters choices (e.g. algo_class for StableBaseline)

Results can be viewed on optuna-dashboard with:

    optuna-dashboard optuna-journal.log

"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Type, Union

import gymnasium as gym
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    Hyperparameter,
)
from optuna_gym_utils import (
    GymDomainForWidthSolvers,
    get_state_continuous_mountain_car,
    set_state_continuous_mountain_car,
)
from stable_baselines3 import A2C, PPO

from skdecide import Solver
from skdecide.core import Value
from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.solver.astar import Astar
from skdecide.hub.solver.bfws import BFWS
from skdecide.hub.solver.iw import IW
from skdecide.hub.solver.mcts import UCT
from skdecide.hub.solver.riw import RIW
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.optuna_utils import generic_optuna_experiment_monoproblem
from skdecide.utils import match_solvers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


# Params of the study
rollout_num_episodes = 2
rollout_max_steps_by_episode = 500
seed = 42  # set this to an integer to get reproducible results, else to None
n_trials = 25  # number of trials to launch
create_another_study = True
overwrite_study = False
domain_reset_is_deterministic = (
    True  # set this to True when the domain reset() is known to be deterministic
)
# => avoid repeating rollout when everything (domain + solver) is deterministic


# Domain to test
ENV_NAME = "MountainCarContinuous-v0"
domain_factory = lambda: GymDomain(gym_env=gym.make(ENV_NAME))
domain4width_factory = lambda: GymDomainForWidthSolvers(
    gym_env=gym.make(ENV_NAME),
    get_state=get_state_continuous_mountain_car,
    set_state=set_state_continuous_mountain_car,
)
width_based_solver_classes = [IW, UCT, BFWS, RIW]
alternative_domain_factory = {
    s: domain4width_factory for s in width_based_solver_classes
}

study_basename = f"gym-{ENV_NAME}-{rollout_num_episodes}-{rollout_max_steps_by_episode}"


# Solvers to test
solver_classes = [Astar, StableBaseline, IW]
solver_classes = match_solvers(
    domain=domain_factory(), candidates=solver_classes
) + match_solvers(
    domain=domain4width_factory(),
    candidates=[s for s in solver_classes if s in width_based_solver_classes],
)
print(f"Selected solver classes to test: {solver_classes}")


# heuristics and state_features (needed by some solvers)
def heuristic(domain, state):
    return Value(150)


def bee1_features(
    domain: Union[GymDomainForWidthSolvers, GymDomain], state: "State"
) -> Tuple[int, Any]:
    return domain.bee1_features(state)


def bee2_features(
    domain: Union[GymDomainForWidthSolvers, GymDomain], state: "State"
) -> Tuple[int, Any]:
    return domain.bee2_features(state)


# Fixed kwargs per solver: either hyperparameters we do not want to search, or other parameters like time limits
kwargs_fixed_by_solver: Dict[Type[Solver], Dict[str, Any]] = {
    Astar: dict(
        parallel=False,
        verbose=False,
        heuristic=heuristic,
    ),
    IW: dict(
        node_ordering=lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: a_novelty
        > b_novelty,
        parallel=False,
        verbose=False,
    ),
    StableBaseline: dict(baselines_policy="MlpPolicy", total_timesteps=1000),
}


# Add new hyperparameters to some solvers
additional_hyperparameters_by_solver: Dict[Type[Solver], List[Hyperparameter]] = {
    StableBaseline: [
        # defined only if $algo_class \in [PPO]$
        FloatHyperparameter(
            name="ent_coef",
            low=0.0,
            high=1.0,
            depends_on=("algo_class", [PPO]),
            suggest_high=True,
            suggest_low=True,
        )
    ],
    IW: [
        CategoricalHyperparameter(
            name="state_features",
            choices={  # associate a label to an actual function (for optuna)
                "bee1": bee1_features,
                "bee2": bee2_features,
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
    alternative_domain_factory=alternative_domain_factory,
)
