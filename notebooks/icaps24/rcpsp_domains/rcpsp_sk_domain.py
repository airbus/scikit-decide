import logging
from typing import Optional

import networkx as nx
import numpy as np
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solvers.cpm import (
    CpmRcpspSolver,
    run_partial_classic_cpm,
)
from discrete_optimization.rcpsp.utils import compute_graph_rcpsp

from skdecide import Space, TransitionOutcome, Value
from skdecide.builders.domain import FullyObservable
from skdecide.domains import RLDomain
from skdecide.hub.space.gym import BoxSpace, DiscreteSpace, SetSpace

logger = logging.getLogger(__name__)
records = []


class D(RLDomain, FullyObservable):
    T_state = np.ndarray  # Type of states
    T_observation = T_state  # Type of observations
    T_event = int  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information in environment outcome


class ParamsDomainEncoding:
    def __init__(
        self,
        return_times_in_state: bool = True,
        return_scheduled_in_state: bool = True,
        use_cpm_for_cost: bool = False,
        use_additive_makespan_for_cost: bool = True,
        terminate_when_already_schedule: bool = False,
        dummy_cost_when_already_schedule: float = 1,
        nb_min_task_inserted: Optional[int] = None,
        nb_max_task_inserted: Optional[int] = None,
        filter_tasks: bool = True,
        only_available_tasks: bool = False,
    ):
        self.return_times_in_state = return_times_in_state
        self.return_scheduled_in_state = return_scheduled_in_state
        self.use_cpm_for_cost = use_cpm_for_cost
        self.use_additive_makespan_for_cost = use_additive_makespan_for_cost
        self.terminate_when_already_schedule = terminate_when_already_schedule
        self.dummy_cost_when_already_schedule = dummy_cost_when_already_schedule
        self.nb_min_task_inserted = nb_min_task_inserted
        self.nb_max_task_inserted = nb_max_task_inserted
        self.filter_tasks = filter_tasks
        self.only_available_tasks = only_available_tasks


class RCPSPSGSDomain(D):
    def __init__(
        self,
        problem: Optional[RcpspProblem] = None,
        params_domain_encoding: Optional[ParamsDomainEncoding] = None,
    ):
        self.params_domain_encoding = params_domain_encoding
        if params_domain_encoding is None:
            self.params_domain_encoding = ParamsDomainEncoding()
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
        self.resource_availability = np.copy(self.initial_resource_availability)
        self.scheduled_tasks = set()
        self.cur_makespan = 0
        self.records = []
        self.cpm = CpmRcpspSolver(problem=self.problem)

    def shallow_copy(self) -> "RCPSPSGSDomain":
        d = RCPSPSGSDomain()
        for attr in self.__dict__.keys():
            setattr(d, attr, getattr(self, attr))
        d.reset()
        return d

    def return_state(self):
        if (
            self.params_domain_encoding.return_scheduled_in_state
            and self.params_domain_encoding.return_times_in_state
        ):
            return self.state
        if self.params_domain_encoding.return_times_in_state:
            return self.state[:, 1]
        if self.params_domain_encoding.return_scheduled_in_state:
            return self.state[:, 0]

    def _state_step(
        self, action: D.T_event
    ) -> TransitionOutcome[D.T_state, Value[D.T_value], D.T_predicate, D.T_info]:
        action = int(action)
        if self.state[action, 0]:
            # action = np.nonzero(self.state[:, 0] == 0)[0][-1]
            # return self._state_step(action)
            return TransitionOutcome(
                self.return_state(),
                Value(
                    cost=self.params_domain_encoding.dummy_cost_when_already_schedule
                ),
                termination=self.params_domain_encoding.terminate_when_already_schedule,
                info=None,
            )
        else:
            tasks = [
                self.task_to_index[j]
                for j in self.all_ancestors_order[self.index_to_task[action]]
                if self.state[self.task_to_index[j], 0] == 0
            ] + [action]
            pre = self.cur_makespan
            if (
                self.params_domain_encoding.nb_max_task_inserted is None
                or len(tasks) - 1 <= self.params_domain_encoding.nb_max_task_inserted
            ) and (
                self.params_domain_encoding.nb_min_task_inserted is None
                or len(tasks) >= self.params_domain_encoding.nb_min_task_inserted
                or action == self.task_to_index[self.problem.sink_task]
            ):
                if self.params_domain_encoding.use_cpm_for_cost:
                    _, m_pre = run_partial_classic_cpm(
                        partial_schedule={
                            self.problem.tasks_list[t]: (
                                self.state[t, 1],
                                self.state[t, 1] + self.dur[t],
                            )
                            for t in range(self.problem.n_jobs)
                            if self.state[t, 0] == 1
                        },
                        cpm_solver=self.cpm,
                    )
                for k in tasks:
                    self.insert_task(k)
                if self.params_domain_encoding.use_cpm_for_cost:
                    _, m = run_partial_classic_cpm(
                        partial_schedule={
                            self.problem.tasks_list[t]: (
                                self.state[t, 1],
                                self.state[t, 1] + self.dur[t],
                            )
                            for t in range(self.problem.n_jobs)
                            if self.state[t, 0] == 1
                        },
                        cpm_solver=self.cpm,
                    )
                term = np.all(self.state[:, 0] == 1)
                return TransitionOutcome(
                    state=self.return_state(),
                    value=Value(
                        cost=self.cur_makespan - pre
                        if self.params_domain_encoding.use_additive_makespan_for_cost
                        else m[self.problem.sink_task]._LSD
                        - m_pre[self.problem.sink_task]._LSD
                    ),
                    termination=term,
                    info=None,
                )
            else:
                return TransitionOutcome(
                    state=self.return_state(),
                    value=Value(cost=50),  # len(tasks)),
                    termination=False,
                    info=None,
                )

    def insert_task(self, k: int):
        res_consumption = self.resource_consumption[k, :]
        min_date = self.state[k, 1]
        if self.dur[k] == 0:
            next_date = min_date
        else:
            next_date = next(
                (
                    t
                    for t in range(min_date, 2 * self.problem.horizon)
                    if all(
                        np.min(self.resource_availability[p, t : t + self.dur[k]])
                        >= res_consumption[p]
                        for p in range(self.resource_availability.shape[0])
                    )
                ),
                None,
            )
        self.state[k, 1] = next_date
        self.state[k, 0] = 1
        for t in range(next_date, next_date + self.dur[k]):
            self.resource_availability[:, t] -= res_consumption
        for succ in self.rcpsp_successors[self.index_to_task[k]]:
            self.state[self.task_to_index[succ], 1] = max(
                self.state[self.task_to_index[succ], 1], next_date + self.dur[k]
            )
        self.cur_makespan = max(self.cur_makespan, next_date + self.dur[k])

    def _get_action_space_(self) -> Space[D.T_event]:
        return DiscreteSpace(self.nb_tasks)

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        if not self.params_domain_encoding.filter_tasks:
            return DiscreteSpace(self.nb_tasks)
        if (
            self.params_domain_encoding.return_scheduled_in_state
            and self.params_domain_encoding.return_times_in_state
        ):
            if self.params_domain_encoding.only_available_tasks:
                s = {
                    i
                    for i in range(self.nb_tasks)
                    if memory[i, 0] == 0
                    and all(
                        memory[self.task_to_index[x], 0] == 1
                        for x in self.all_ancestors_order[self.index_to_task[i]]
                    )
                }
            else:
                s = {i for i in range(self.nb_tasks) if memory[i, 0] == 0}
            if len(s) == 0:
                return SetSpace({self.nb_tasks - 1})
            return SetSpace(s)
        if self.params_domain_encoding.return_scheduled_in_state:
            if self.params_domain_encoding.only_available_tasks:
                s = {
                    i
                    for i in range(self.nb_tasks)
                    if memory[i] == 0
                    and all(
                        memory[self.task_to_index[x]] == 1
                        for x in self.all_ancestors_order[self.index_to_task[i]]
                    )
                }
            else:
                s = {i for i in range(self.nb_tasks) if memory[i] == 0}
            if len(s) == 0:
                return SetSpace({self.nb_tasks - 1})
            return SetSpace(s)
        return DiscreteSpace(self.nb_tasks)

    def _state_reset(self) -> D.T_state:
        global records
        if self.state[-1, 0]:
            records.append(self.state[-1, 1])
            if len(records) >= 30:
                logger.info(f"{sum(records[-30:]) / 30}")
        self.state = np.copy(self.initial_state)
        self.resource_availability = np.copy(self.initial_resource_availability)
        self.scheduled_tasks = set()
        self.cur_makespan = 0
        return self.return_state()

    # def _get_observation(self, state: D.T_state, action: Optional[D.T_event] = None) -> D.T_observation:
    #     return state

    def _get_observation_space_(self) -> Space[D.T_observation]:
        if (
            self.params_domain_encoding.return_times_in_state
            and self.params_domain_encoding.return_scheduled_in_state
        ):
            return BoxSpace(
                low=0,
                high=self.problem.horizon,
                dtype=int,
                shape=(self.problem.n_jobs, 2),
            )
        if self.params_domain_encoding.return_times_in_state:
            return BoxSpace(
                low=0,
                high=self.problem.horizon,
                dtype=int,
                shape=(self.problem.n_jobs,),
            )
        if self.params_domain_encoding.return_scheduled_in_state:
            return BoxSpace(low=0, high=1, dtype=int, shape=(self.problem.n_jobs,))
