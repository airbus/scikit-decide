#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Utilities to create optuna studies for scikit-decide."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    Hyperparameter,
)

from skdecide import Domain, Solver
from skdecide.builders.domain import (
    DeterministicTransitions,
    SingleAgent,
    TransformedObservable,
)
from skdecide.builders.solver import DeterministicPolicies
from skdecide.core import D, Value
from skdecide.utils import rollout

logger = logging.getLogger(__name__)


try:
    import optuna
except ImportError:
    logger.warning("You should install optuna to use this module.")
else:
    from optuna.pruners import BasePruner, MedianPruner
    from optuna.samplers import BaseSampler
    from optuna.storages import JournalFileStorage, JournalStorage
    from optuna.trial import Trial, TrialState


def generic_optuna_experiment_monoproblem(
    domain_factory: Callable[[], Domain],
    solver_classes: List[Type[Solver]],
    kwargs_fixed_by_solver: Optional[Dict[Type[Solver], Dict[str, Any]]] = None,
    suggest_optuna_kwargs_by_name_by_solver: Optional[
        Dict[Type[Solver], Dict[str, Dict[str, Any]]]
    ] = None,
    additional_hyperparameters_by_solver: Optional[
        Dict[Type[Solver], List[Hyperparameter]]
    ] = None,
    n_trials: int = 150,
    allow_retry_same_trial: bool = False,
    rollout_num_episodes: int = 3,
    rollout_max_steps_by_episode: int = 1000,
    rollout_from_memory: Optional[D.T_memory[D.T_state]] = None,
    domain_reset_is_deterministic: bool = False,
    study_basename: str = "study",
    create_another_study: bool = True,
    overwrite_study=False,
    storage_path: str = "./optuna-journal.log",
    sampler: Optional[BaseSampler] = None,
    pruner: Optional[BasePruner] = None,
    seed: Optional[int] = None,
    objective: Optional[
        Callable[
            [
                Solver,
                List[
                    Tuple[
                        List[D.T_agent[D.T_observation]],
                        List[D.T_agent[D.T_concurrency[D.T_event]]],
                        List[D.T_agent[Value[D.T_value]]],
                    ]
                ],
            ],
            float,
        ]
    ] = None,
    optuna_tuning_direction: str = "maximize",
    alternative_domain_factory: Optional[
        Dict[Type[Solver], Callable[[], Domain]]
    ] = None,
) -> optuna.Study:
    """Create and run an optuna study to tune solvers hyperparameters for a given domain factory.

    The optuna study will choose a solver and its hyperparameters in order to optimize
    the cumulated reward during a rollout.

    When
    - solver policy is deterministic,
    - domain treansitions are deterministic,
    - domain state to observation is deterministic,
    - and rollout starts from a specified memory or domain.reset() is deterministic,
    we avoid repeating episode are they will be all the same.

    One can
    - freeze some hyperparameters via kwargs_fixed_by_solver
    - tune solvers.__init__ via kwargs_fixed_by_solver
    - restrict the choices/ranges for some hyperparameters via suggest_optuna_kwargs_by_name_by_solver
    - add other hyperparameters to some solvers via additional_hyperparameters_by_solver

    The optuna study can be monitored with optuna-dashboard with

        optuna-dashboard optuna-journal.log

    (or the relevant path set by `storage_path`)

    # Parameters
    domain_factory: a callable with no argument returning the domain to solve (can be a mere domain class).
    solver_classes: list of solvers to consider.
    kwargs_fixed_by_solver: fixed hyperparameters by solver. Can also be other parameters needed by solvers' __init__().
    suggest_optuna_kwargs_by_name_by_solver: kwargs_by_name passed to solvers' suggest_with_optuna().
        Useful to restrict or specify choices, step, high, ...
    additional_hyperparameters_by_solver: additional user-defined hyperparameters by solver, to be suggested by optuna
    n_trials: number of trials to be run in the optuna study
    allow_retry_same_trial: if True, allow trial with same parameters as before to be retried (useful if solve process is random for instance)
    rollout_num_episodes: nb of episodes used in rollout to compute the value associated to a set of hyperparameters
    rollout_max_steps_by_episode: max steps by episode used in rollout to compute the value associated to a set of hyperparameters
    rollout_from_memory: if specified, rollout episode will start from this memory
    domain_reset_is_deterministic: specified whether the domain reset() method (when existing) is deterministic.
        This information is used when rollout_from_memory is None (and thus domain.reset() is used) ,
        to decide if several episodes are needed or not, depending on whether everything is deterministic or not.
    study_basename: base name of the study generated.
        If `create_another_study` is True, a timestamp will be added to this base name.
    create_another_study: if `True` a timestamp prefix will be added to the study base name in order to avoid
        overwritting or continuing a previously created study.
        Should be False, if one wants to add trials to an existing study.
    overwrite_study: if True, any study with the same name as the one generated here will be deleted before starting the optuna study.
        Should be False, if one wants to add trials to an existing study.
    storage_path: path to the journal used by optuna used to log the study. Can be a NFS path to allow parallelized optuna studies.
    sampler: sampler used by the optuna study. If None, a TPESampler is used with the provided `seed`.
    pruner: pruner used by the optuna study. if None, a MedianPruner is used.
    seed: used to create the sampler if `sampler` is None. Should be set to an integer if one wants to ensure
        reproducible results.
    aggreg_outcome_rewards: function used to aggregate outcome.value into a scalar.
        Default to taking `float(outcome.value.reward)` for single agent solver,
        and to taking `sum(float(v.reward) for v in outcome.value.values())` for multi agents solver.
    objective: function used to compute the scalar optimized by optuna.
        Takes solver and episodes obtain by solve + rollout as arguments and should return a float.
        Episodes being a list of episode represented by a tuple of observations, actions, values.
        Default to the cumulated reward other all episodes (and all agents when on a multiagent domain).
    optuna_tuning_direction: direction of optuna optimization ("maximize" or "minimize")
    alternative_domain_factory: mapping solver_class -> domain_factory when some solvers need a different domain_factory
        (e.g. width-based solvers need GymDomainForWidthSolvers instead of simple GymDomain)

    # Returns
    the launched optuna study.

    """
    domain_cls = domain_factory().__class__

    # default parameters
    if kwargs_fixed_by_solver is None:
        kwargs_fixed_by_solver = defaultdict(dict)
    if suggest_optuna_kwargs_by_name_by_solver is None:
        suggest_optuna_kwargs_by_name_by_solver = defaultdict(dict)
    if additional_hyperparameters_by_solver is None:
        additional_hyperparameters_by_solver = defaultdict(list)
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=seed)
    if pruner is None:
        pruner = MedianPruner()
    if alternative_domain_factory is None:
        alternative_domain_factory = {}
    if objective is None:
        if issubclass(domain_cls, SingleAgent):

            def aggreg_outcome_rewards(
                value: domain_cls.T_agent[Value[domain_cls.T_value]],
            ) -> float:
                return float(value.reward)

        else:

            def aggreg_outcome_rewards(
                value: domain_cls.T_agent[Value[domain_cls.T_value]],
            ) -> float:
                return sum(float(v.reward) for v in value.values())

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
            """Compute cumulated reward over all episodes and all agents."""
            return sum(
                [
                    sum(aggreg_outcome_rewards(v) for v in values)
                    for observations, actions, values in episodes
                ]
            )

    # study name
    suffix = f"-{time.time()}" if create_another_study else ""
    study_name = f"{study_basename}{suffix}"

    # we need to map the classes to a unique string, to be seen as a categorical hyperparameter by optuna
    # by default, we use the class name, but if there are identical names, f"{cls.__module__}.{cls.__name__}" could be used.
    solvers_by_name: Dict[str, Type[Solver]] = {
        cls.__name__: cls for cls in solver_classes
    }

    # add new user-defined hyperparameters to the solvers
    for (
        solver_cls,
        additional_hyperparameters,
    ) in additional_hyperparameters_by_solver.items():
        solver_cls.hyperparameters = (
            list(solver_cls.hyperparameters) + additional_hyperparameters
        )

    # objective definition
    def complete_optuna_objective(trial: Trial):
        # hyperparameters to test

        # first parameter: solver choice
        solver_name: str = trial.suggest_categorical("solver", choices=solvers_by_name)
        solver_class = solvers_by_name[solver_name]

        # hyperparameters for the chosen solver
        suggested_hyperparameters_kwargs = (
            solver_class.suggest_hyperparameters_with_optuna(
                trial=trial,
                prefix=solver_name + ".",
                kwargs_by_name=suggest_optuna_kwargs_by_name_by_solver.get(
                    solver_class, None
                ),
                fixed_hyperparameters=kwargs_fixed_by_solver.get(solver_class, None),
            )
        )

        # get actual domain_factory corresponding to the solver
        actual_domain_factory = alternative_domain_factory.get(
            solver_class, domain_factory
        )

        if not allow_retry_same_trial:
            # use existing value if corresponding to a previous complete trial (it may happen that the sampler repropose same params)
            states_to_consider = (TrialState.COMPLETE,)
            trials_to_consider = trial.study.get_trials(
                deepcopy=False, states=states_to_consider
            )
            for t in reversed(trials_to_consider):
                if trial.params == t.params:
                    msg = "Trial with same hyperparameters as a previous complete trial: returning previous fit."
                    logger.warning(msg)
                    trial.set_user_attr("Error", msg)
                    return t.value

            # prune if corresponding to a previous failed trial
            states_to_consider = (TrialState.FAIL, TrialState.PRUNED)
            trials_to_consider = trial.study.get_trials(
                deepcopy=False, states=states_to_consider
            )
            for t in reversed(trials_to_consider):
                if trial.params == t.params:
                    msg = "Pruning trial identical to a previous failed trial."
                    trial.set_user_attr("Error", msg)
                    raise optuna.TrialPruned(msg)

        logger.info(f"Launching trial {trial.number} with parameters: {trial.params}")

        # construct kwargs for __init__, init_model, and solve
        kwargs = dict(suggested_hyperparameters_kwargs)  # copy
        if solver_class in kwargs_fixed_by_solver:
            kwargs.update(kwargs_fixed_by_solver[solver_class])

        try:
            with solver_class(domain_factory=actual_domain_factory, **kwargs) as solver:
                solver.solve()
                domain = actual_domain_factory()

                _is_rollout_deterministic = (
                    isinstance(solver, DeterministicPolicies)
                    and isinstance(domain, DeterministicTransitions)
                    and isinstance(domain, TransformedObservable)
                    and (
                        domain_reset_is_deterministic or rollout_from_memory is not None
                    )
                )

                if _is_rollout_deterministic:
                    num_episodes = 1
                    episodes_multiplier = rollout_num_episodes
                else:
                    num_episodes = rollout_num_episodes
                    episodes_multiplier = 1

                episodes = rollout(
                    domain=domain,
                    solver=solver,
                    num_episodes=num_episodes,
                    max_steps=rollout_max_steps_by_episode,
                    verbose=False,
                    outcome_formatter=None,
                    action_formatter=None,
                    render=False,
                    return_episodes=True,
                )
                # in case of deterministic rollout, repeat episodes
                episodes = episodes_multiplier * episodes

            return objective(solver, episodes)

        except Exception as e:
            if isinstance(e, optuna.TrialPruned):
                raise e  # pruning error managed directly by optuna
            else:
                # Store exception message as trial user attribute
                msg = f"{e.__class__}: {e}"
                trial.set_user_attr("Error", msg)
                raise optuna.TrialPruned(msg)  # show failed

    # create study + database to store it
    storage = JournalStorage(JournalFileStorage(storage_path))
    if overwrite_study:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except:
            pass
    study = optuna.create_study(
        study_name=study_name,
        direction=optuna_tuning_direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=not overwrite_study,
    )
    study.optimize(complete_optuna_objective, n_trials=n_trials)
    return study
