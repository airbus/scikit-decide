# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPSolution,
    MS_RCPSPSolution_Variant,
)

from skdecide import Domain
from skdecide.builders.domain.scheduling.scheduling_domains import SchedulingDomain
from skdecide.hub.solver.do_solver.sgs_policies import PolicyMethodParams, PolicyRCPSP
from skdecide.hub.solver.do_solver.sk_to_do_binding import build_do_domain
from skdecide.solvers import DeterministicPolicies, Solver


class D(SchedulingDomain):
    pass


class SolvingMethod(Enum):
    """
    - PILE : solve scheduling problem with greedy queue method
    - GA : solve scheduling problem with genetic algorithm
    - LS : solve scheduling problem with local search algorithm (hill climber or simulated annealing)
    - LP : solve scheduling problem with constraint programming solver
    - CP : solve scheduling problem with constraint programming solver
    - LNS_LP : solve scheduling problem with large neighborhood search + LP solver
    - LNS_CP : solve scheduling problem with large neighborhood search + CP solver
    """

    PILE = "greedy"
    GA = "ga"
    LS = "ls"
    LP = "lp"
    CP = "cp"
    LNS_LP = "lns-lp"
    LNS_CP = "lns-scheduling"


def build_solver(
    solving_method: SolvingMethod, do_domain: Problem
) -> Tuple[SolverDO, Dict[str, Any]]:
    """Build the discrete-optimization solver for a given solving method

    # Parameters
    solving_method: method of the solver
    do_domain: discrete-opt problem to solve.
    """
    if isinstance(do_domain, RCPSPModel):
        from discrete_optimization.rcpsp.rcpsp_solvers import (
            look_for_solver,
            solvers_map,
        )

        do_domain_cls = RCPSPModel
    elif isinstance(do_domain, MS_RCPSPModel):
        from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_solvers import (
            look_for_solver,
            solvers_map,
        )

        do_domain_cls = MS_RCPSPModel
    else:
        raise ValueError("do_domain should be either a RCPSPModel or a MS_RCPSPModel.")
    available = look_for_solver(do_domain)
    smap = [
        (av, solvers_map[av])
        for av in available
        if solvers_map[av][0] == solving_method.value
    ]
    if len(smap) > 0:
        return smap[0]
    else:
        raise ValueError(
            f"solving_method {solving_method} not available for {do_domain_cls}."
        )


def from_solution_to_policy(
    solution: Union[RCPSPSolution, MS_RCPSPSolution, MS_RCPSPSolution_Variant],
    domain: SchedulingDomain,
    policy_method_params: PolicyMethodParams,
) -> PolicyRCPSP:
    """Create a PolicyRCPSP object (a skdecide policy) from a scheduling solution
    from the discrete-optimization library."""
    permutation_task = None
    modes_dictionnary = None
    schedule = None
    resource_allocation = None
    resource_allocation_priority = None
    if isinstance(solution, RCPSPSolution):
        permutation_task = sorted(
            solution.rcpsp_schedule,
            key=lambda x: (solution.rcpsp_schedule[x]["start_time"], x),
        )
        schedule = solution.rcpsp_schedule
        modes_dictionnary = {}
        # set modes for start and end (dummy) jobs
        modes_dictionnary[1] = 1
        modes_dictionnary[solution.problem.n_jobs_non_dummy + 2] = 1
        for i in range(len(solution.rcpsp_modes)):
            modes_dictionnary[i + 2] = solution.rcpsp_modes[i]
    elif isinstance(solution, MS_RCPSPSolution):
        permutation_task = sorted(
            solution.schedule, key=lambda x: (solution.schedule[x]["start_time"], x)
        )
        schedule = solution.schedule
        employees = sorted(domain.get_resource_units_names())
        resource_allocation = {
            task: [
                employees[i] for i in solution.employee_usage[task].keys()
            ]  # warning here...
            for task in solution.employee_usage
        }
        if isinstance(solution, MS_RCPSPSolution_Variant):
            resource_allocation_priority = solution.priority_worker_per_task
            modes_dictionnary = {}
            # set modes for start and end (dummy) jobs
            modes_dictionnary[1] = 1
            modes_dictionnary[solution.problem.n_jobs_non_dummy + 2] = 1
            for i in range(len(solution.modes_vector)):
                modes_dictionnary[i + 2] = solution.modes_vector[i]
        else:
            modes_dictionnary = solution.modes

    return PolicyRCPSP(
        domain=domain,
        policy_method_params=policy_method_params,
        permutation_task=permutation_task,
        modes_dictionnary=modes_dictionnary,
        schedule=schedule,
        resource_allocation=resource_allocation,
        resource_allocation_priority=resource_allocation_priority,
    )


class DOSolver(Solver, DeterministicPolicies):
    """Wrapper of discrete-optimization solvers for scheduling problems

    # Attributes
    - policy_method_params:  params for the returned policy.
    - method: method of the discrete-optim solver used
    - dict_params: specific params passed to the do-solver
    - callback: scikit-decide callback to be called inside do-solver when relevant.
    """

    T_domain = D

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        policy_method_params: PolicyMethodParams,
        method: SolvingMethod = SolvingMethod.PILE,
        dict_params: Optional[Dict[Any, Any]] = None,
        callback: Callable[[DOSolver], bool] = lambda solver: False,
    ):
        Solver.__init__(self, domain_factory=domain_factory)
        self.callback = callback
        self.method = method
        self.policy_method_params = policy_method_params
        self.dict_params = dict_params
        if self.dict_params is None:
            self.dict_params = {}

    def get_available_methods(self, domain: SchedulingDomain):
        do_domain = build_do_domain(domain)
        if isinstance(do_domain, (MS_RCPSPModel)):
            from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_solvers import (
                look_for_solver,
                solvers_map,
            )

            available = look_for_solver(do_domain)
        elif isinstance(do_domain, RCPSPModel):
            from discrete_optimization.rcpsp.rcpsp_solvers import (
                look_for_solver,
                solvers_map,
            )

            available = look_for_solver(do_domain)
        smap = [(av, solvers_map[av]) for av in available]

        return smap

    def _solve(self) -> None:
        self.domain = self._domain_factory()
        self.do_domain = build_do_domain(self.domain)
        solvers = build_solver(solving_method=self.method, do_domain=self.do_domain)
        solver_class = solvers[0]
        key, params = solvers[1]
        for k in params:
            if k not in self.dict_params:
                self.dict_params[k] = params[k]

        # callbacks
        callbacks = [_DOCallback(callback=self.callback, solver=self)]
        copy_dict_params = deepcopy(self.dict_params)
        if "callbacks" in copy_dict_params:
            callbacks = callbacks + copy_dict_params.pop("callbacks")

        self.solver = solver_class(self.do_domain, **copy_dict_params)

        if hasattr(self.solver, "init_model") and callable(self.solver.init_model):
            self.solver.init_model(**copy_dict_params)

        result_storage = self.solver.solve(callbacks=callbacks, **copy_dict_params)
        best_solution: RCPSPSolution = result_storage.get_best_solution()

        assert best_solution is not None

        fits = self.do_domain.evaluate(best_solution)

        self.best_solution = best_solution

        self.policy_object = from_solution_to_policy(
            solution=best_solution,
            domain=self.domain,
            policy_method_params=self.policy_method_params,
        )

    def get_external_policy(self) -> PolicyRCPSP:
        return self.policy_object

    def compute_external_policy(self, policy_method_params: PolicyMethodParams):
        return from_solution_to_policy(
            solution=self.best_solution,
            domain=self.domain,
            policy_method_params=policy_method_params,
        )

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        return self.policy_object.get_next_action(observation=observation)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return self.policy_object.is_policy_defined_for(observation=observation)


class _DOCallback(Callback):
    def __init__(
        self,
        callback: Callable[[DOSolver], bool],
        solver: DOSolver,
    ):
        self.solver = solver
        self.callback = callback

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        """Called at the end of an optimization step.

        # Parameters
            step: index of step
            res: current result storage
            solver: solvers using the callback

        # Returns
            If `True`, the optimization process is stopped, else it goes on.

        """
        stopping = self.callback(self.solver)
        return stopping
