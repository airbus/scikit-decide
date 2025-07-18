import logging
from typing import Optional

import networkx as nx
import numpy as np
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_utils import compute_graph_rcpsp
from discrete_optimization.rcpsp.solver.cpm import (
    CPM,
    CPMObject,
    run_partial_classic_cpm,
)

import skdecide
from skdecide import Space, TransitionOutcome, Value
from skdecide.builders.domain import FullyObservable, Goals
from skdecide.domains import DeterministicPlanningDomain, RLDomain
from skdecide.hub.space.gym import (
    BoxSpace,
    DiscreteSpace,
    GymSpace,
    ListSpace,
    MultiDiscreteSpace,
)

logger = logging.getLogger(__name__)
records = []


class D(RLDomain, FullyObservable):
    T_state = np.ndarray  # Type of states
    T_observation = T_state  # Type of observations
    T_event = int  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information in environment outcome


class ParamsDomainEncodingLS:
    def __init__(
        self,
        use_multidiscrete: bool = True,
        use_boxspace: bool = False,
        depth_terminal: int = 5,
    ):
        self.use_multidiscrete = use_multidiscrete
        self.use_boxspace = use_boxspace
        self.depth_terminal = depth_terminal
        assert self.use_multidiscrete ^ self.use_boxspace


class RCPSP_LS_Domain(D):
    def __init__(
        self,
        problem: Optional[RCPSPModel] = None,
        params_domain_encoding: Optional[ParamsDomainEncodingLS] = None,
    ):
        self.params_domain_encoding = params_domain_encoding
        if self.params_domain_encoding is None:
            self.params_domain_encoding = ParamsDomainEncodingLS()
        if problem is None:
            return
        self.problem = problem
        self.nb_tasks = self.problem.n_jobs
        self.nb_resources = len(self.problem.resources_list)
        self.dur = np.array(
            [
                self.problem.mode_details[t][1]["duration"]
                for t in self.problem.tasks_list
            ],
            dtype=int,
        )
        self.resource_consumption = np.array(
            [
                [
                    self.problem.mode_details[t][1].get(r, 0)
                    for r in self.problem.resources_list
                ]
                for t in self.problem.tasks_list
            ],
            dtype=int,
        )
        self.rcpsp_successors = {t: set(s) for t, s in self.problem.successors.items()}
        self.graph = compute_graph_rcpsp(self.problem)
        self.graph_nx = self.graph.to_networkx()
        self.topological_order = list(nx.topological_sort(self.graph_nx))
        self.all_ancestors = self.graph.full_predecessors
        self.all_ancestors_order = {
            t: sorted(
                self.all_ancestors[t], key=lambda x: self.topological_order.index(x)
            )
            for t in self.all_ancestors
        }
        self.task_to_index = {
            self.problem.tasks_list[i]: i for i in range(self.problem.n_jobs)
        }
        self.index_to_task = {
            i: self.problem.tasks_list[i] for i in range(self.problem.n_jobs)
        }
        self.initial_state = np.zeros([self.nb_tasks, 2], dtype=np.int64)
        self.initial_resource_availability = np.array(
            [
                self.problem.get_resource_availability_array(r)
                for r in self.problem.resources_list
            ],
            dtype=int,
        )
        self.state = np.copy(self.initial_state)
        self.current_permutation = np.array(
            [i for i in range(self.problem.n_jobs_non_dummy)]
        )[::-1]
        self.resource_availability = np.copy(self.initial_resource_availability)
        self.scheduled_tasks = set()
        self.cur_makespan = 0
        self.records = []
        self.cpm = CPM(problem=self.problem)
        self.cur_sol = RCPSPSolution(
            problem=self.problem, rcpsp_permutation=self.current_permutation
        )
        self.state = np.array(
            [
                self.cur_sol.rcpsp_schedule[t]["start_time"]
                for t in self.problem.tasks_list
            ]
        )
        self.cur_depth = 0
        self.cur_makespan = self.cur_sol.rcpsp_schedule[self.problem.sink_task][
            "start_time"
        ]

    def shallow_copy(self) -> "RCPSP_LS_Domain":
        d = RCPSP_LS_Domain()
        for attr in self.__dict__.keys():
            setattr(d, attr, getattr(self, attr))
        d.reset()
        return d

    def _state_step(
        self, action: D.T_event
    ) -> TransitionOutcome[D.T_state, Value[D.T_value], D.T_predicate, D.T_info]:
        i0, i1 = action
        i0 = int(i0)
        i1 = int(i1)
        max_i1 = max(i1, i0)
        min_i0 = min(i1, i0)
        pre = self.current_permutation[i0]
        self.current_permutation[min_i0:max_i1] = self.current_permutation[
            min_i0:max_i1
        ][::-1]
        # self.current_permutation[i0] = self.current_permutation[i1]
        # self.current_permutation[i1] = pre
        pre_m = self.cur_makespan
        self.cur_sol = RCPSPSolution(
            problem=self.problem, rcpsp_permutation=self.current_permutation
        )
        self.cur_makespan = self.cur_sol.rcpsp_schedule[self.problem.sink_task][
            "end_time"
        ]
        self.cur_depth += 1
        # self.current_permutation = np.array(sol.standardised_permutation)
        self.state = np.array(
            [
                self.cur_sol.rcpsp_schedule[t]["start_time"]
                for t in self.problem.tasks_list
            ]
        )
        return TransitionOutcome(
            self.state,
            Value(cost=self.cur_makespan - pre_m),
            termination=self.cur_depth >= self.params_domain_encoding.depth_terminal,
            info=None,
        )

    def _get_action_space_(self) -> Space[D.T_event]:
        if self.params_domain_encoding.use_multidiscrete:
            return MultiDiscreteSpace([self.problem.n_jobs_non_dummy for _ in range(2)])
        if self.params_domain_encoding.use_boxspace:
            return BoxSpace(
                low=0, high=self.problem.n_jobs_non_dummy - 1, dtype=int, shape=(2,)
            )

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        if self.params_domain_encoding.use_multidiscrete:
            return MultiDiscreteSpace([self.problem.n_jobs_non_dummy for _ in range(2)])
        if self.params_domain_encoding.use_boxspace:
            return BoxSpace(
                low=0, high=self.problem.n_jobs_non_dummy - 1, dtype=int, shape=(2,)
            )

    def _state_reset(self) -> D.T_state:
        global records
        records.append(self.cur_makespan)
        if len(records) >= 30:
            logger.info(f"{sum(records[-30:])/30}")
        self.state = np.copy(self.initial_state)
        self.resource_availability = np.copy(self.initial_resource_availability)
        self.scheduled_tasks = set()
        self.cur_makespan = 0
        self.cur_depth = 0
        self.current_permutation = np.array(
            [i for i in range(self.problem.n_jobs_non_dummy)]
        )[::-1]
        self.cur_sol = RCPSPSolution(
            problem=self.problem, rcpsp_permutation=self.current_permutation
        )
        self.state = np.array(
            [
                self.cur_sol.rcpsp_schedule[t]["start_time"]
                for t in self.problem.tasks_list
            ]
        )
        self.cur_makespan = self.cur_sol.rcpsp_schedule[self.problem.sink_task][
            "start_time"
        ]
        return self.state

    def _get_observation(
        self, state: D.T_state, action: Optional[D.T_event] = None
    ) -> D.T_observation:
        return state

    def _get_observation_space_(self) -> Space[D.T_observation]:
        return BoxSpace(low=0, high=self.problem.horizon, shape=(self.problem.n_jobs,))
        # return MultiDiscreteSpace(nvec=np.array([[1, self.problem.horizon]
        #                                          for t in self.problem.tasks_list]))


class ParamsDomainEncodingLSOneStep:
    def __init__(self, action_as_float: bool = True, action_as_int: bool = False):
        self.action_as_float = action_as_float
        self.action_as_int = action_as_int
        assert self.action_as_int ^ self.action_as_float


class RCPSP_LS_Domain_OneStep(D):
    def __init__(
        self,
        problem: Optional[RCPSPModel] = None,
        params_domain_encoding: Optional[ParamsDomainEncodingLSOneStep] = None,
    ):
        self.params_domain_encoding = params_domain_encoding
        if params_domain_encoding is None:
            self.params_domain_encoding = ParamsDomainEncodingLSOneStep()
        if problem is None:
            return
        self.problem = problem
        self.nb_tasks = self.problem.n_jobs
        self.nb_resources = len(self.problem.resources_list)
        self.dur = np.array(
            [
                self.problem.mode_details[t][1]["duration"]
                for t in self.problem.tasks_list
            ],
            dtype=int,
        )
        self.resource_consumption = np.array(
            [
                [
                    self.problem.mode_details[t][1].get(r, 0)
                    for r in self.problem.resources_list
                ]
                for t in self.problem.tasks_list
            ],
            dtype=int,
        )
        self.rcpsp_successors = {t: set(s) for t, s in self.problem.successors.items()}
        self.graph = compute_graph_rcpsp(self.problem)
        self.graph_nx = self.graph.to_networkx()
        self.topological_order = list(nx.topological_sort(self.graph_nx))
        self.all_ancestors = self.graph.full_predecessors
        self.all_ancestors_order = {
            t: sorted(
                self.all_ancestors[t], key=lambda x: self.topological_order.index(x)
            )
            for t in self.all_ancestors
        }
        self.task_to_index = {
            self.problem.tasks_list[i]: i for i in range(self.problem.n_jobs)
        }
        self.index_to_task = {
            i: self.problem.tasks_list[i] for i in range(self.problem.n_jobs)
        }
        self.initial_state = np.zeros([self.nb_tasks, 2], dtype=np.int64)
        self.initial_resource_availability = np.array(
            [
                self.problem.get_resource_availability_array(r)
                for r in self.problem.resources_list
            ],
            dtype=int,
        )
        self.state = np.copy(self.initial_state)
        self.current_permutation = np.array(
            [i for i in range(self.problem.n_jobs_non_dummy)]
        )[::-1]
        self.resource_availability = np.copy(self.initial_resource_availability)
        self.scheduled_tasks = set()
        self.cur_makespan = 0
        self.records = []
        self.cpm = CPM(problem=self.problem)
        self.cur_sol = RCPSPSolution(
            problem=self.problem, rcpsp_permutation=self.current_permutation
        )
        self.state = np.array(
            [
                self.cur_sol.rcpsp_schedule[t]["start_time"]
                for t in self.problem.tasks_list
            ]
        )
        self.cur_depth = 0
        self.cur_makespan = self.cur_sol.rcpsp_schedule[self.problem.sink_task][
            "start_time"
        ]

    def shallow_copy(self) -> "RCPSP_LS_Domain_OneStep":
        d = RCPSP_LS_Domain_OneStep()
        for attr in self.__dict__.keys():
            setattr(d, attr, getattr(self, attr))
        d.reset()
        return d

    def _state_step(
        self, action: D.T_event
    ) -> TransitionOutcome[D.T_state, Value[D.T_value], D.T_predicate, D.T_info]:
        self.current_permutation = np.argsort(action)
        # self.current_permutation[i0] = self.current_permutation[i1]
        # self.current_permutation[i1] = pre
        pre_m = self.cur_makespan
        self.cur_sol = RCPSPSolution(
            problem=self.problem, rcpsp_permutation=self.current_permutation
        )
        self.cur_makespan = self.cur_sol.rcpsp_schedule[self.problem.sink_task][
            "end_time"
        ]
        self.current_permutation = np.array(self.cur_sol.standardised_permutation)
        self.state = np.array(
            [
                self.cur_sol.rcpsp_schedule[t]["start_time"]
                for t in self.problem.tasks_list
            ]
        )
        return TransitionOutcome(
            self.state, Value(cost=self.cur_makespan), termination=True, info=None
        )

    def _get_action_space_(self) -> Space[D.T_event]:
        if self.params_domain_encoding.action_as_float:
            return BoxSpace(
                low=0, high=1, dtype=float, shape=(self.problem.n_jobs_non_dummy,)
            )
        else:
            return MultiDiscreteSpace(
                nvec=[
                    self.problem.n_jobs_non_dummy
                    for _ in range(self.problem.n_jobs_non_dummy)
                ]
            )
            # return BoxSpace(low=0, high=self.problem.n_jobs_non_dummy,
            #                 dtype=int, shape=(self.problem.n_jobs_non_dummy,))

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        if self.params_domain_encoding.action_as_float:
            return BoxSpace(
                low=0, high=1, dtype=float, shape=(self.problem.n_jobs_non_dummy,)
            )
        else:
            # return BoxSpace(low=0, high=self.problem.n_jobs_non_dummy,
            #                 dtype=int,
            #                 shape=(self.problem.n_jobs_non_dummy,))
            return MultiDiscreteSpace(
                nvec=[
                    self.problem.n_jobs_non_dummy
                    for _ in range(self.problem.n_jobs_non_dummy)
                ]
            )

    def _state_reset(self) -> D.T_state:
        global records
        records.append(self.cur_makespan)
        if len(records) >= 30:
            logger.info(f"{sum(records[-30:])/30}")
        self.current_permutation = np.array(
            [i for i in range(self.problem.n_jobs_non_dummy)]
        )
        self.cur_sol = RCPSPSolution(
            problem=self.problem, rcpsp_permutation=self.current_permutation
        )
        self.state = np.array(
            [
                self.cur_sol.rcpsp_schedule[t]["start_time"]
                for t in self.problem.tasks_list
            ]
        )
        self.state = np.array([0 for t in self.problem.tasks_list])
        self.cur_makespan = self.cur_sol.rcpsp_schedule[self.problem.sink_task][
            "start_time"
        ]
        return self.state

    def _get_observation_space_(self) -> Space[D.T_observation]:
        return BoxSpace(low=0, high=self.problem.horizon, shape=(self.problem.n_jobs,))
