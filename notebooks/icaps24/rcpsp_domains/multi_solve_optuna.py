"""Example using OPTUNA to choose a solving method and tune its hyperparameters for coloring.

This example show different features of optuna integration with discrete-optimization:
- use of `suggest_hyperparameters_with_optuna()` to get hyperparameters values
- use of a dedicated callback to report intermediate results with corresponding time to optuna
  and potentially prune the trial
- time-based pruner
- how to fix some parameters/hyperparameters

Results can be viewed on optuna-dashboard with:

    optuna-dashboard optuna-journal.log

"""

import logging

from skdecide import rollout
from skdecide.builders.domain.scheduling.scheduling_domains import SingleModeRCPSP
from skdecide.hub.solver.do_solver import BasePolicyMethod, DOSolver, PolicyMethodParams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")
import time
from collections import defaultdict
from typing import Any, Dict, List, Type

import optuna
from discrete_optimization.generic_rcpsp_tools.solvers.ls import LsGenericRcpspSolver
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.callbacks.optuna import OptunaCallback
from discrete_optimization.generic_tools.cp_tools import CpSolverName, ParametersCp
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.optuna.timed_percentile_pruner import (
    TimedPercentilePruner,
)
from discrete_optimization.rcpsp.solvers.cp_mzn import CpRcpspSolver
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.solvers.ga import GaRcpspSolver
from discrete_optimization.rcpsp.solvers.pile import PileRcpspSolver
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.trial import Trial, TrialState


def run_optuna_multisolve(
    rcpsp_domain: SingleModeRCPSP, name_file_log: str = "optuna_journal.log"
):
    seed = 42  # set this to an integer to get reproducible results, else to None
    optuna_nb_trials = 100  # number of trials to launch
    # gurobi_full_license_available = False  # is the installed gurobi having a full license?
    # (contrary to the license installed by `pip install gurobipy`)
    create_another_study = True  # True: generate a study name with timestamp to avoid overwriting previous study,
    # False: keep same study name
    overwrite = False  # True: delete previous studies with same name (in particular, if create_another_study=False),
    # False: keep the study and add trials to the existing ones
    max_time_per_solver = 10  # max duration per solver (seconds)
    min_time_per_solver = 5  # min duration before pruning a solver (seconds)

    suffix = f"-{time.time()}" if create_another_study else ""
    study_name = f"rcpsp_all_solvers-auto-pruning-{suffix}"
    storage_path = f"./{name_file_log}"  # NFS path for distributed optimization
    elapsed_time_attr = "elapsed_time"
    # name of the user attribute used to store duration of trials
    # (updated during intermediate reports)

    # solvers to test
    solvers_to_test: List[Type[SolverDO]] = [
        CpRcpspSolver,
        GaRcpspSolver,
        LsGenericRcpspSolver,
        CpRcpspSolver,
        PileRcpspSolver,
        CpSatRcpspSolver,
    ]
    # fixed kwargs per solver: either hyperparameters we do not want to search, or other parameters like time limits
    p = ParametersCp.default_cpsat()
    p.nb_process = 6
    p.time_limit = max_time_per_solver
    kwargs_fixed_by_solver: Dict[Type[SolverDO], Dict[str, Any]] = defaultdict(
        dict,  # default kwargs for unspecified solvers
        {
            CpSatRcpspSolver: dict(parameters_cp=p, warmstart=True),
            CpRcpspSolver: dict(parameters_cp=p),
            GaRcpspSolver: dict(max_evals=10000),
        },
    )
    # restrict some hyperparameters choices, for some solvers (making use of `kwargs_by_name` of `suggest_with_optuna`)
    suggest_optuna_kwargs_by_name_by_solver: Dict[
        Type[SolverDO], Dict[str, Dict[str, Any]]
    ] = defaultdict(
        dict,
        {
            CpRcpspSolver: {
                "cp_solver_name": {
                    "choices": [
                        CpSolverName.CHUFFED,
                        CpSolverName.ORTOOLS,
                        CpSolverName.GECODE,
                    ]
                }
            }
        },
    )

    # we need to map the classes to a unique string, to be seen as a categorical hyperparameter by optuna
    # by default, we use the class name, but if there are identical names, f"{cls.__module__}.{cls.__name__}"
    # could be used.
    solvers_by_name: Dict[str, Type[SolverDO]] = {
        cls.__name__: cls for cls in solvers_to_test
    }

    direction = "maximize"

    # objective definition
    def objective(trial: Trial):
        # hyperparameters to test

        # first parameter: solver choice
        solver_name: str = trial.suggest_categorical("solver", choices=solvers_by_name)
        solver_class = solvers_by_name[solver_name]

        # hyperparameters for the chosen solver  (only those not already fixed)
        hyperparameters_names = [
            h
            for h in solver_class.get_hyperparameters_names()
            if h not in kwargs_fixed_by_solver[solver_class]
        ]
        suggested_hyperparameters_kwargs = (
            solver_class.suggest_hyperparameters_with_optuna(
                names=hyperparameters_names,
                trial=trial,
                prefix=solver_name + ".",
                kwargs_by_name=suggest_optuna_kwargs_by_name_by_solver[
                    solver_class
                ],  # options to restrict the choices of some hyperparameter
            )
        )
        # use existing value if corresponding to a previous complete trial
        # (it may happen that the sampler repropose same params)
        states_to_consider = (TrialState.COMPLETE,)
        trials_to_consider = trial.study.get_trials(
            deepcopy=False, states=states_to_consider
        )
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                logger.warning(
                    "Trial with same hyperparameters as a previous complete trial: returning previous fit."
                )
                return t.value

        # prune if corresponding to a previous failed trial
        states_to_consider = (TrialState.FAIL,)
        trials_to_consider = trial.study.get_trials(
            deepcopy=False, states=states_to_consider
        )
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                raise optuna.TrialPruned(
                    "Pruning trial identical to a previous failed trial."
                )

        logger.info(f"Launching trial {trial.number} with parameters: {trial.params}")

        # construct kwargs for __init__, init_model, and solve
        kwargs = dict(
            kwargs_fixed_by_solver[solver_class]
        )  # copy the frozen kwargs dict
        kwargs.update(suggested_hyperparameters_kwargs)
        starting_time = time.perf_counter()
        # solver init
        kwargs["callbacks"] = [
            OptunaCallback(
                trial=trial,
                starting_time=starting_time,
                elapsed_time_attr=elapsed_time_attr,
                report_time=True,
                # report intermediate values according to elapsed time instead of iteration number
            ),
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            ),
        ]
        solver = DOSolver(
            domain_factory=lambda: rcpsp_domain,
            policy_method_params=PolicyMethodParams(
                base_policy_method=BasePolicyMethod.FOLLOW_GANTT
            ),
            method=None,
            do_solver_type=solver_class,
            dict_params=kwargs,
        )
        try:
            solver.solve()
            # store elapsed time
            elapsed_time = time.perf_counter() - starting_time
            trial.set_user_attr(elapsed_time_attr, elapsed_time)
            episodes = rollout(
                domain=rcpsp_domain,
                solver=solver,
                from_memory=rcpsp_domain.get_initial_state(),
                verbose=False,
                return_episodes=True,
                num_episodes=1,
            )
            values = episodes[0][-1]
            fit = sum([v.cost for v in values])
            return -fit
        except:
            raise optuna.TrialPruned("failed")

    storage = JournalStorage(JournalFileStorage(storage_path))
    if overwrite:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except:
            pass
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=TimedPercentilePruner(  # intermediate values interpolated at same "step"
            percentile=60,  # median pruner
            n_warmup_steps=min_time_per_solver,  # no pruning during first seconds
        ),
        storage=storage,
        load_if_exists=not overwrite,
    )
    study.set_metric_names(["makespan"])
    study.optimize(objective, n_trials=optuna_nb_trials)


def run_example_study():
    from discrete_optimization.rcpsp.parser import get_data_available

    from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain

    file = [f for f in get_data_available() if "j1201_1.sm" in f][0]
    rcpsp_domain = load_domain(file)
    run_optuna_multisolve(
        rcpsp_domain=rcpsp_domain, name_file_log="optuna_journal_offline.log"
    )


if __name__ == "__main__":
    run_example_study()
