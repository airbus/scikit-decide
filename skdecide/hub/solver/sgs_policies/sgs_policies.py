# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Union, List, Dict, Optional
from skdecide.builders.solver.policy import DeterministicPolicies
from skdecide.builders.domain.scheduling.scheduling_domains_modelling import State, SchedulingAction, SchedulingActionEnum
from skdecide.builders.domain.scheduling.scheduling_domains import SchedulingDomain, D, MultiModeRCPSP, SingleModeRCPSP
from enum import Enum
from functools import partial


class BasePolicyMethod(Enum):
    FOLLOW_GANTT = 0
    SGS_PRECEDENCE = 1
    SGS_READY = 2
    SGS_STRICT = 3
    SGS_TIME_FREEDOM = 4
    SGS_INDEX_FREEDOM = 5
    PILE = 6


class PolicyMethodParams:
    def __init__(self, base_policy_method: BasePolicyMethod,
                 delta_time_freedom=10, delta_index_freedom=10):
        self.base_policy_method = base_policy_method
        self.delta_time_freedom = delta_time_freedom
        self.delta_index_freedom = delta_index_freedom


class PolicyRCPSP(DeterministicPolicies):
    T_domain = D

    def __init__(self,
                 domain: SchedulingDomain,
                 policy_method_params: PolicyMethodParams,
                 permutation_task: List[int],
                 modes_dictionnary: Dict[int, int],
                 schedule: Optional[Dict[int, Dict[str, int]]]=None,  # {id: {"start_time":, "end_time"}}
                 resource_allocation: Optional[Dict[int, List[str]]]=None,
                 resource_allocation_priority: Optional[Dict[int, List[str]]]=None):
        self.domain = domain
        self.policy_method_params = policy_method_params
        self.permutation_task = permutation_task
        self.modes_dictionnary = modes_dictionnary
        self.schedule = schedule
        self.store_start_date = {}
        if self.schedule is not None:
            for task_id in self.schedule:
                start_date = self.schedule[task_id]["start_time"]
                if start_date not in self.store_start_date:
                    self.store_start_date[start_date] = set()
                self.store_start_date[start_date].add(task_id)
        self.resource_allocation = resource_allocation
        self.resource_allocation_priority = resource_allocation_priority
        self.build_function()

    def reset(self):
        pass

    def build_function(self):
        func = partial(map_method_to_function[self.policy_method_params.base_policy_method],
                       policy_rcpsp=self,
                       check_if_applicable=False,
                       domain_sk_decide=self.domain,
                       delta_time_freedom=self.policy_method_params.delta_time_freedom,
                       delta_index_freedom=self.policy_method_params.delta_index_freedom)
        self.func = func

    def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        return self.func(state=observation)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True


def action_in_applicable_actions(domain_sk_decide,
                                 observation: D.T_agent[D.T_observation],
                                 the_action: SchedulingAction):
    return domain_sk_decide.check_if_action_can_be_started(observation, the_action)


def next_action_follow_static_gantt(policy_rcpsp: PolicyRCPSP,
                                    state: State,
                                    check_if_applicable: bool=False,
                                    **kwargs):
    obs: State = state
    t = obs.t
    ongoing_task = obs.tasks_ongoing
    complete_task = obs.tasks_complete
    possible_task_to_launch = policy_rcpsp.domain.task_possible_to_launch_precedence(state=state)
    the_action = SchedulingAction(task=None,
                                  action=SchedulingActionEnum.TIME_PR,
                                  mode=None,
                                  time_progress=True,
                                  resource_unit_names=None)
    if t in policy_rcpsp.store_start_date:
        tasks = [task for task in policy_rcpsp.store_start_date[t]
                 if task not in ongoing_task and task not in complete_task
                 and task in possible_task_to_launch]
        if len(tasks) > 0:
            the_action = SchedulingAction(task=tasks[0],
                                          action=SchedulingActionEnum.START,
                                          mode=policy_rcpsp.modes_dictionnary[tasks[0]],
                                          time_progress=False,
                                          resource_unit_names=None)
            if policy_rcpsp.resource_allocation is not None:
                if tasks[0] in policy_rcpsp.resource_allocation:
                    the_action.resource_unit_names = policy_rcpsp.resource_allocation[tasks[0]]
    if True:
        action_available = action_in_applicable_actions(policy_rcpsp.domain,
                                                        state,
                                                        the_action)
        if not action_available[0]:
            the_action = SchedulingAction(task=None,
                                          action=SchedulingActionEnum.TIME_PR,
                                          mode=None,
                                          time_progress=True,
                                          resource_unit_names=None)
    return the_action


def next_action_sgs_first_task_precedence_ready(policy_rcpsp: PolicyRCPSP,
                                                state: State,
                                                check_if_applicable: bool=False,
                                                **kwargs):
    obs: State = state

    next_task_to_launch = None
    possible_task_to_launch = policy_rcpsp.domain.task_possible_to_launch_precedence(state=state)
    sorted_task_not_done = sorted([(index, policy_rcpsp.permutation_task[index])
                                   for index in range(len(policy_rcpsp.permutation_task))
                                   if policy_rcpsp.permutation_task[index] in state.tasks_remaining],
                                  key=lambda x: x[0])
    for i in range(len(sorted_task_not_done)):
        task = sorted_task_not_done[i][1]
        if task in possible_task_to_launch:
            next_task_to_launch = task
            break
    if next_task_to_launch is not None:
        if policy_rcpsp.schedule is not None:
            original_time_start_task = policy_rcpsp.schedule[next_task_to_launch]["start_time"]
            other_tasks_same_time = [task_id for ind, task_id in sorted_task_not_done
                                     if task_id != next_task_to_launch
                                     and policy_rcpsp.schedule[task_id]["start_time"] == original_time_start_task
                                     and task_id in possible_task_to_launch]
        else:
            other_tasks_same_time = []
        tasks_of_interest = [next_task_to_launch]+other_tasks_same_time
        the_action = None
        for tinterest in tasks_of_interest:
            the_action = SchedulingAction(task=tinterest,
                                          action=SchedulingActionEnum.START,
                                          mode=policy_rcpsp.modes_dictionnary[tinterest],
                                          time_progress=False, resource_unit_names=None)
            if policy_rcpsp.resource_allocation is not None:
                if tinterest in policy_rcpsp.resource_allocation:
                    the_action.resource_unit_names = policy_rcpsp.resource_allocation[tinterest]
            applicable = action_in_applicable_actions(domain_sk_decide=policy_rcpsp.domain,
                                                      observation=state,
                                                      the_action=the_action)
            if applicable[0]:
                break
            else:
                the_action = None
        if the_action is None:
            the_action = SchedulingAction(task=None,
                                          action=SchedulingActionEnum.TIME_PR,
                                          mode=None,
                                          time_progress=True,
                                          resource_unit_names=None)
        return the_action
    else:
        return SchedulingAction(task=None,
                                action=SchedulingActionEnum.TIME_PR,
                                mode=None,
                                time_progress=True,
                                resource_unit_names=None)


def next_action_sgs_first_task_ready(policy_rcpsp: PolicyRCPSP,
                                     state: State,
                                     check_if_applicable: bool=False,
                                     domain_sk_decide: Union[MultiModeRCPSP, SingleModeRCPSP]=None,
                                     **kwargs):
    obs: State = state
    t = obs.t
    sorted_task_not_done = sorted([(index, policy_rcpsp.permutation_task[index])
                                   for index in range(len(policy_rcpsp.permutation_task))
                                   if policy_rcpsp.permutation_task[index] in state.tasks_remaining],
                                  key=lambda x: x[0])
    next_task_to_launch = None
    possible_task_to_launch = policy_rcpsp.domain.task_possible_to_launch_precedence(state=state)
    for i in range(len(sorted_task_not_done)):
        task = sorted_task_not_done[i][1]
        if task not in possible_task_to_launch:
            continue
        the_action = SchedulingAction(task=task,
                                      action=SchedulingActionEnum.START,
                                      mode=policy_rcpsp.modes_dictionnary[task],
                                      time_progress=False, resource_unit_names=None)
        if policy_rcpsp.resource_allocation is not None:
            if task in policy_rcpsp.resource_allocation:
                the_action.resource_unit_names = policy_rcpsp.resource_allocation[task]
        action_available = action_in_applicable_actions(policy_rcpsp.domain, state, the_action)
        if action_available[0]:
            return the_action

    the_action = SchedulingAction(task=None,
                                  action=SchedulingActionEnum.TIME_PR,
                                  mode=None,
                                  time_progress=True,
                                  resource_unit_names=None)
    return the_action


def next_action_sgs_strict(policy_rcpsp: PolicyRCPSP,
                           state: State,
                           check_if_applicable: bool=False,
                           domain_sk_decide: Union[MultiModeRCPSP, SingleModeRCPSP]=None,
                           **kwargs):
    obs: State = state
    t = obs.t
    possible_task_to_launch = policy_rcpsp.domain.task_possible_to_launch_precedence(state=state)
    sorted_task_not_done = sorted([(index, policy_rcpsp.permutation_task[index])
                                   for index in range(len(policy_rcpsp.permutation_task))
                                   if policy_rcpsp.permutation_task[index] in state.tasks_remaining
                                   and policy_rcpsp.permutation_task[index] in possible_task_to_launch],
                                  key=lambda x: x[0])
    the_action = None
    if len(sorted_task_not_done) > 0:
        other_tasks_same_time = [sorted_task_not_done[0][1]]
        if policy_rcpsp.schedule is not None:
            scheduled_time = policy_rcpsp.schedule[sorted_task_not_done[0][1]]["start_time"]
            other_tasks_same_time = [task_id for ind, task_id in sorted_task_not_done
                                     if policy_rcpsp.schedule[task_id]["start_time"] == scheduled_time]
        for tinterest in other_tasks_same_time:
            the_action = SchedulingAction(task=tinterest,
                                          action=SchedulingActionEnum.START,
                                          mode=policy_rcpsp.modes_dictionnary[tinterest],
                                          time_progress=False,
                                          resource_unit_names=None) #start_tasks=[tinterest], advance_time=False)
            if policy_rcpsp.resource_allocation is not None:
                if tinterest in policy_rcpsp.resource_allocation:
                    the_action.resource_unit_names = policy_rcpsp.resource_allocation[tinterest]
            applicable = action_in_applicable_actions(policy_rcpsp.domain,
                                                      observation=state,
                                                      the_action=the_action)
            if applicable[0]:
                break
            else:
                the_action = None
    if the_action is None:
        the_action = SchedulingAction(task=None,
                                      action=SchedulingActionEnum.TIME_PR,
                                      mode=None,
                                      time_progress=True,
                                      resource_unit_names=None)
    return the_action


def next_action_sgs_time_freedom(policy_rcpsp: PolicyRCPSP,
                                 state: State,
                                 check_if_applicable: bool=False,
                                 domain_sk_decide: Union[MultiModeRCPSP, SingleModeRCPSP]=None,
                                 delta_time_freedom: int=10,
                                 **kwargs):
    obs: State = state
    possible_task_to_launch = policy_rcpsp.domain.task_possible_to_launch_precedence(state=state)
    sorted_task_not_done = sorted([(index, policy_rcpsp.permutation_task[index])
                                   for index in range(len(policy_rcpsp.permutation_task))
                                   if policy_rcpsp.permutation_task[index] in state.tasks_remaining
                                   and policy_rcpsp.permutation_task[index] in possible_task_to_launch],
                                  key=lambda x: x[0])
    the_action = None
    if len(sorted_task_not_done) > 0:
        other_tasks_same_time = [sorted_task_not_done[0][1]]
        if policy_rcpsp.schedule is not None:
            scheduled_time = policy_rcpsp.schedule[sorted_task_not_done[0][1]]["start_time"]
            other_tasks_same_time = [task_id for ind, task_id in sorted_task_not_done
                                     if scheduled_time <= policy_rcpsp.schedule[task_id]["start_time"]
                                     <= scheduled_time + delta_time_freedom]
        for tinterest in other_tasks_same_time:
            the_action = SchedulingAction(task=tinterest,
                                          action=SchedulingActionEnum.START,
                                          mode=policy_rcpsp.modes_dictionnary[tinterest],
                                          time_progress=False,
                                          resource_unit_names=None) # start_tasks=[tinterest], advance_time=False)
            if policy_rcpsp.resource_allocation is not None:
                if tinterest in policy_rcpsp.resource_allocation:
                    the_action.resource_unit_names = policy_rcpsp.resource_allocation[tinterest]
            applicable = action_in_applicable_actions(policy_rcpsp.domain,
                                                      observation=state,
                                                      the_action=the_action)
            if applicable[0]:
                break
            else:
                the_action = None
    if the_action is None:
        the_action = SchedulingAction(task=None,
                                      action=SchedulingActionEnum.TIME_PR,
                                      mode=None,
                                      time_progress=True,
                                      resource_unit_names=None)
    return the_action


def next_action_sgs_index_freedom(policy_rcpsp: PolicyRCPSP,
                                  state: State,
                                  check_if_applicable: bool=False,
                                  domain_sk_decide: Union[MultiModeRCPSP, SingleModeRCPSP]=None,
                                  delta_index_freedom: int=10,
                                  **kwargs):
    obs: State = state
    possible_task_to_launch = policy_rcpsp.domain.task_possible_to_launch_precedence(state=state)
    sorted_task_not_done = sorted([(index, policy_rcpsp.permutation_task[index])
                                   for index in range(len(policy_rcpsp.permutation_task))
                                   if policy_rcpsp.permutation_task[index] in state.tasks_remaining
                                   and policy_rcpsp.permutation_task[index] in possible_task_to_launch],
                                  key=lambda x: x[0])
    the_action = None
    if len(sorted_task_not_done) > 0:
        index_t = sorted_task_not_done[0][0]
        other_tasks_same_time = [task_id for ind, task_id in sorted_task_not_done
                                 if ind <= index_t + delta_index_freedom]
        for tinterest in other_tasks_same_time:
            the_action = SchedulingAction(task=tinterest,
                                          action=SchedulingActionEnum.START,
                                          mode=policy_rcpsp.modes_dictionnary[tinterest],
                                          time_progress=False,
                                          resource_unit_names=None)  # start_tasks=[tinterest], advance_time=False)
            if policy_rcpsp.resource_allocation is not None:
                if tinterest in policy_rcpsp.resource_allocation:
                    the_action.resource_unit_names = policy_rcpsp.resource_allocation[tinterest]
            applicable = action_in_applicable_actions(policy_rcpsp.domain,
                                                      observation=state,
                                                      the_action=the_action)
            if applicable[0]:
                break
            else:
                the_action = None
    if the_action is None:
        the_action = SchedulingAction(task=None,
                                      action=SchedulingActionEnum.TIME_PR,
                                      mode=None,
                                      time_progress=True,
                                      resource_unit_names=None)
    return the_action


map_method_to_function = {BasePolicyMethod.FOLLOW_GANTT: next_action_follow_static_gantt,
                          BasePolicyMethod.SGS_PRECEDENCE: next_action_sgs_first_task_precedence_ready,
                          BasePolicyMethod.SGS_STRICT: next_action_sgs_strict,
                          BasePolicyMethod.SGS_READY: next_action_sgs_first_task_ready,
                          BasePolicyMethod.SGS_TIME_FREEDOM: next_action_sgs_time_freedom,
                          BasePolicyMethod.SGS_INDEX_FREEDOM: next_action_sgs_index_freedom}
