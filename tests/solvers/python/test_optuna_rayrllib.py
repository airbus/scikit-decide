from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Type

import ray

from skdecide.hub.domain.maze.maze import Maze
from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.optuna_utils import generic_optuna_experiment_monoproblem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


class FakeRayRLlib(RayRLlib):
    generated_configs: List[Dict[str, Any]]

    def _solve(self) -> None:
        FakeRayRLlib.generated_configs.append(self._config.to_dict())


def test_generic_optuna_experiment_monoproblem_with_ray_rllib():
    ray.init(
        runtime_env={"working_dir": os.path.dirname(__file__)}
    )  # add FakeRayRLlib to ray runtimeenv

    # Params of the study
    rollout_num_episodes = 0
    rollout_max_steps_by_episode = 500
    domain_reset_is_deterministic = True
    seed = 42  # set this to an integer to get reproducible results, else to None
    n_trials = 5  # number of trials to launch
    create_another_study = False
    overwrite_study = True
    study_basename = (
        f"maze-rayrllib-{rollout_num_episodes}-{rollout_max_steps_by_episode}"
    )

    # Domain to test
    domain_factory = Maze

    # Init list of configs generated
    FakeRayRLlib.generated_configs = []

    # Solvers to test
    solver_classes = [FakeRayRLlib]

    # Fixed kwargs per solver: either hyperparameters we do not want to search, or other parameters like time limits
    kwargs_fixed_by_solver = {
        RayRLlib: dict(train_iterations=1, num_cpus_per_worker=0.5),
    }

    study = generic_optuna_experiment_monoproblem(
        domain_factory=domain_factory,
        solver_classes=solver_classes,
        kwargs_fixed_by_solver=kwargs_fixed_by_solver,
        n_trials=n_trials,
        rollout_num_episodes=rollout_num_episodes,
        rollout_max_steps_by_episode=rollout_max_steps_by_episode,
        domain_reset_is_deterministic=domain_reset_is_deterministic,
        study_basename=study_basename,
        create_another_study=create_another_study,
        overwrite_study=overwrite_study,
        seed=seed,
    )

    assert len(FakeRayRLlib.generated_configs) == n_trials

    for trial, config in zip(study.trials, FakeRayRLlib.generated_configs):
        algo_class_name = trial.params["FakeRayRLlib.algo_class"]
        lr_log = trial.params[f"FakeRayRLlib.lr_log"]
        lr = 10**lr_log
        gamma_complement_log = trial.params["FakeRayRLlib.gamma_complement_log"]
        gamma = 1 - 10**gamma_complement_log
        assert config["lr"] == lr
        assert config["gamma"] == gamma
        if algo_class_name in ["PPO"]:
            entropy_coeff_log = trial.params["FakeRayRLlib.entropy_coeff_log"]
            entropy_coeff = 10**entropy_coeff_log
            assert config["entropy_coeff"] == entropy_coeff
            assert "FakeRayRLlib.train_batch_size_log2" not in trial.params
            sgd_minibatch_size_log2 = trial.params[
                "FakeRayRLlib.sgd_minibatch_size_log2"
            ]
            sgd_minibatch_size = 2**sgd_minibatch_size_log2
            assert config["sgd_minibatch_size"] == sgd_minibatch_size

        else:
            assert "FakeRayRLlib.entropy_coeff_log" not in trial.params
            assert "FakeRayRLlib.sgd_minibatch_size_log2" not in trial.params
            train_batch_size_log2 = trial.params["FakeRayRLlib.train_batch_size_log2"]
            train_batch_size = 2**train_batch_size_log2
            assert config["train_batch_size"] == train_batch_size
