# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from enum import Enum
from itertools import product
from typing import Dict, Iterable, List, Optional, Set, Tuple

from skdecide import (
    DiscreteDistribution,
    Distribution,
    Domain,
    EnumerableSpace,
    ImplicitSpace,
    SamplableSpace,
    Space,
    T,
    TransitionOutcome,
    Value,
)
from skdecide.builders.domain import (
    Actions,
    DeterministicInitialized,
    DeterministicTransitions,
    FullyObservable,
    Goals,
    Markovian,
    Sequential,
    Simulation,
    SingleAgent,
    UncertainTransitions,
)
from skdecide.builders.domain.scheduling.conditional_tasks import (
    WithConditionalTasks,
    WithoutConditionalTasks,
)
from skdecide.builders.domain.scheduling.graph_toolbox import Graph
from skdecide.builders.domain.scheduling.modes import MultiMode, SingleMode
from skdecide.builders.domain.scheduling.preallocations import (
    WithoutPreallocations,
    WithPreallocations,
)
from skdecide.builders.domain.scheduling.precedence import WithPrecedence
from skdecide.builders.domain.scheduling.preemptivity import (
    WithoutPreemptivity,
    WithPreemptivity,
)
from skdecide.builders.domain.scheduling.resource_availability import (
    DeterministicResourceAvailabilityChanges,
    UncertainResourceAvailabilityChanges,
    WithoutResourceAvailabilityChange,
)
from skdecide.builders.domain.scheduling.resource_consumption import (
    ConstantResourceConsumption,
    VariableResourceConsumption,
)
from skdecide.builders.domain.scheduling.resource_costs import (
    WithModeCosts,
    WithoutModeCosts,
    WithoutResourceCosts,
    WithResourceCosts,
)
from skdecide.builders.domain.scheduling.resource_renewability import (
    MixedRenewable,
    RenewableOnly,
)
from skdecide.builders.domain.scheduling.resource_type import (
    WithoutResourceUnit,
    WithResourceTypes,
    WithResourceUnits,
)
from skdecide.builders.domain.scheduling.scheduling_domains_modelling import (
    SchedulingAction,
    SchedulingActionEnum,
    State,
)
from skdecide.builders.domain.scheduling.skills import (
    WithoutResourceSkills,
    WithResourceSkills,
)
from skdecide.builders.domain.scheduling.task import Task
from skdecide.builders.domain.scheduling.task_duration import (
    DeterministicTaskDuration,
    SimulatedTaskDuration,
    UncertainUnivariateTaskDuration,
)
from skdecide.builders.domain.scheduling.task_progress import (
    CustomTaskProgress,
    DeterministicTaskProgress,
)
from skdecide.builders.domain.scheduling.time_lag import WithoutTimeLag, WithTimeLag
from skdecide.builders.domain.scheduling.time_windows import (
    WithoutTimeWindow,
    WithTimeWindow,
)


class SchedulingObjectiveEnum(Enum):
    """Enum defining the different scheduling objectives"""

    MAKESPAN = 0
    COST = 1


SchedulingObjectiveEnum.MAKESPAN.__doc__ = "makespan (to be minimized)"
SchedulingObjectiveEnum.COST.__doc__ = "cost of resources (to be minimized)"


class D(
    Domain,
    SingleAgent,
    Sequential,
    Simulation,
    DeterministicInitialized,
    Actions,
    FullyObservable,
    Markovian,
    Goals,
):
    """
    Base class for any scheduling statefull domain
    """

    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = SchedulingAction  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class SchedulingDomain(
    WithPrecedence,
    MultiMode,
    VariableResourceConsumption,
    WithPreemptivity,
    WithResourceTypes,
    WithResourceUnits,
    MixedRenewable,
    SimulatedTaskDuration,
    CustomTaskProgress,
    WithResourceSkills,
    WithTimeLag,
    WithTimeWindow,
    WithPreallocations,
    WithConditionalTasks,
    UncertainResourceAvailabilityChanges,
    WithModeCosts,
    WithResourceCosts,
    D,
):
    """
    This is the highest level scheduling domain class (inheriting top-level class for each mandatory
    domain characteristic).
    This is where the implementation of the statefull scheduling domain is implemented,
    letting to the user the possibility
    to the user to define the scheduling problem without having to think of a statefull version.
    """

    def _state_sample(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> TransitionOutcome[D.T_state, D.T_agent[Value[D.T_value]], D.T_agent[D.T_info]]:
        """This function will be used if the domain is defined as a Simulation (i.e. transitions are defined by call to
        a simulation). This function may also be used by simulation-based solvers on non-Simulation domains."""
        current_state: State = memory
        next_state = current_state if self.inplace_environment else current_state.copy()
        next_state = self.update_pause_tasks_simulation(next_state, action)
        next_state = self.update_resume_tasks_simulation(next_state, action)
        next_state = self.update_start_tasks_simulation(next_state, action)
        next_state = self.update_complete_dummy_tasks_simulation(next_state, action)
        next_state = self.update_conditional_tasks_simulation(next_state, action)
        next_state = self.update_time_simulation(next_state, action)
        next_state = self.update_resource_availability_simulation(next_state, action)
        is_terminal = self._is_terminal(next_state)
        is_goal = self._is_goal(next_state)
        return TransitionOutcome(
            state=next_state,
            value=self._get_transition_value(memory, action, next_state),
            termination=is_goal or is_terminal,
            info=None,
        )

    def _get_next_state_distribution(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> Distribution[D.T_state]:
        """This function will be used if the domain is defined with UncertainTransitions. This function will be ignored
        if the domain is defined as a Simulation. This function may also be used by uncertainty-specialised solvers
         on deterministic domains."""
        current_state: State = memory
        next_state = current_state if self.inplace_environment else current_state.copy()
        next_state = self.update_pause_tasks_uncertain(next_state, action)
        next_state = self.update_resume_tasks_uncertain(next_state, action)
        next_state_distrib = self.update_start_tasks_uncertain(next_state, action)
        next_state_distrib = self.update_complete_dummy_tasks_uncertain(
            next_state_distrib, action
        )
        next_state_distrib = self.update_conditional_tasks_uncertain(
            next_state_distrib, action
        )
        next_state_distrib = self.update_time_uncertain(next_state_distrib, action)
        next_state_distrib = self.update_resource_availability_uncertain(
            next_state_distrib, action
        )  # maybe this should be done after self.update_time too ??
        return next_state_distrib

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        """
        This function will be used if the domain is defined with DeterministicTransitions. This function will be ignored
        if the domain is defined as having UncertainTransitions or Simulation."""
        current_state: State = memory
        next_state = current_state if self.inplace_environment else current_state.copy()
        next_state = self.update_pause_tasks(next_state, action)
        next_state = self.update_resume_tasks(next_state, action)
        next_state = self.update_start_tasks(next_state, action)
        next_state = self.update_complete_dummy_tasks(next_state, action)
        next_state = self.update_conditional_tasks(next_state, action)
        next_state = self.update_time(next_state, action)
        next_state = self.update_resource_availability(
            next_state, action
        )  # maybe this should be done after self.update_time too ??

        return next_state

    def _get_initial_state_(self) -> D.T_state:
        """
        Create and return an empty initial state
        """
        s = State(
            task_ids=self.get_tasks_ids(),
            tasks_available=self.get_all_unconditional_tasks(),
        )
        s.t = 0
        resource_availability = {
            r: self.sample_quantity_resource(resource=r, time=s.t)
            for r in self.get_resource_types_names()
        }
        resource_availability.update(
            {
                runit: self.sample_quantity_resource(resource=runit, time=s.t)
                for runit in self.get_resource_units_names()
            }
        )
        s.resource_availability = resource_availability
        s.resource_used = {r: 0 for r in s.resource_availability}
        return s

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        """
        To be implemented if needed one day.
        """
        pass

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        """
        Returns the action space from a state.
        TODO : think about a way to avoid the instaceof usage.
        """
        if isinstance(self, WithoutResourceSkills) and isinstance(
            self, WithoutResourceUnit
        ):
            return SchedulingActionSpace(domain=self, state=memory)
        else:
            # return SchedulingActionSpaceWithResourceUnit(domain=self, state=memory)
            return SchedulingActionSpaceWithResourceUnitSamplable(
                domain=self, state=memory
            )

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        """
        To be implemented if needed one day.
        """
        pass

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return ImplicitSpace(
            lambda state: (
                len(state.task_ids)
                == len(state.tasks_complete) + len(state.tasks_unsatisfiable)
                and (len(state.tasks_ongoing) == 0)
                and (len(state.tasks_paused) == 0)
            )
        )

    def get_max_horizon(self) -> int:
        """Return the maximum time horizon (int)"""
        return self._get_max_horizon()

    def _get_max_horizon(self) -> int:
        """Return the maximum time horizon (int)"""
        raise NotImplementedError

    def initialize_domain(self):
        """Initialize a scheduling domain. This function needs to be called when instantiating a scheduling domain."""
        self.sampled_durations = {}  # TODO : remove ?
        self.graph = (
            self.compute_graph()
        )  # TODO : this is very specific to the precendence caracteristic,
        # should it be done there ?
        self.ancestors = self.graph.predecessors_map()
        self.successors = self.graph.successors_map()
        self.full_predecessors = self.graph.ancestors_map()
        self.full_successors = self.graph.descendants_map()
        self.inplace_environment = False

    def set_inplace_environment(self, inplace_environment: bool):
        """
        Activate or not the fact that the simulator modifies the given state inplace or create a copy before.
        The inplace version is several times faster but will lead to bugs in graph search solver.
        """
        self.inplace_environment = inplace_environment

    # Build the precedence graph.
    # TODO : maybe this function should be in the precedence module.
    def compute_graph(self):
        task_ids = self.get_tasks_ids()
        successors = self.get_successors()
        mode_details = self.get_tasks_modes()
        nodes = [
            (
                n,
                {
                    mode: self.sample_task_duration(task=n, mode=mode)
                    for mode in mode_details[n]
                },
            )
            for n in task_ids
        ]
        edges = []
        for n in successors:
            for succ in successors[n]:
                edges += [(n, succ, {})]
        return Graph(nodes, edges, False)

    def update_time(self, state: State, action: SchedulingAction):
        """Update the time of the state if the time_progress attribute of the given EnumerableAction is True."""
        next_state = state  # .copy()
        if action.time_progress:
            next_state = self.update_progress(next_state)
            next_state = self.update_res_consumption(next_state)
            next_state = self.update_complete_tasks(next_state)
            next_state.t = state.t + 1
        return next_state

    def update_time_uncertain(
        self, states: DiscreteDistribution[State], action: SchedulingAction
    ):
        """Update the time of the state if the time_progress attribute of the given EnumerableAction is True."""
        next_states = DiscreteDistribution(
            [(state, prob) for (state, prob) in states.get_values()]
        )
        if action.time_progress:
            next_states = self.update_progress_uncertain(next_states)
            next_states = self.update_res_consumption_uncertain(next_states)
            next_states = self.update_complete_tasks_uncertain(next_states)
            for next_state, _ in next_states.get_values():
                next_state.t = next_state.t + 1
        return next_states

    def update_time_simulation(self, state: State, action: SchedulingAction):
        """In a simulated scheduling environment, update the time of the state if the time_progress attribute of the
        given EnumerableAction is True."""
        next_state = state  # .copy()
        if action.time_progress:
            next_state = self.update_progress_simulation(next_state)
            next_state = self.update_res_consumption_simulation(next_state)
            next_state = self.update_complete_tasks_simulation(next_state)
            next_state.t = state.t + 1
        return next_state

    def update_res_consumption(self, state: State):
        # TODO : test this.
        next_state = state  # .copy()
        for task_id in state.tasks_ongoing:
            resource_to_use = self.get_resource_used(
                task=task_id,
                mode=state.tasks_mode[task_id],
                resource_unit_names=set(),
                time_since_start=state.tasks_details[task_id].get_task_active_time(
                    state.t + 1
                ),
            )
            for r in resource_to_use:
                prev = next_state.resource_used_for_task[task_id][r]
                new = resource_to_use[r]
                next_state.resource_used_for_task[task_id][r] = resource_to_use[r]
                next_state.resource_used[r] += new - prev
        return next_state

    def update_res_consumption_uncertain(self, states: DiscreteDistribution[State]):
        next_states = DiscreteDistribution(
            [(state, prob) for (state, prob) in states.get_values()]
        )

        for next_state, _ in next_states.get_values():
            for task_id in next_state.tasks_ongoing:
                resource_to_use = self.get_resource_used(
                    task=task_id,
                    mode=next_state.tasks_mode[task_id],
                    resource_unit_names=set(),
                    time_since_start=next_state.tasks_details[
                        task_id
                    ].get_task_active_time(next_state.t + 1),
                )
                for r in resource_to_use:
                    prev = next_state.resource_used_for_task[task_id][r]
                    new = resource_to_use[r]
                    next_state.resource_used_for_task[task_id][r] = resource_to_use[r]
                    next_state.resource_used[r] += new - prev

        return next_states

    def update_res_consumption_simulation(self, state: State):
        return self.update_res_consumption(state)

    def update_progress(self, state: State):
        """Update the progress of all ongoing tasks in the state."""
        next_state = state  # .copy()
        for task_id in next_state.tasks_ongoing:
            # TODO : update the resource used dictionnary also, for the task that consumes a varying number
            # of ressource
            next_state.tasks_progress[task_id] += self.get_task_progress(
                task_id,
                t_from=next_state.t,
                t_to=next_state.t + 1,
                mode=next_state.tasks_mode[task_id],
                sampled_duration=next_state.tasks_details[task_id].sampled_duration,
            )
        return next_state

    def update_progress_uncertain(self, states: DiscreteDistribution[State]):
        """In an uncertain scheduling environment, update the progress of all ongoing tasks in the state."""
        next_states = DiscreteDistribution(
            [(state, prob) for (state, prob) in states.get_values()]
        )

        for next_state, _ in next_states.get_values():
            for task_id in next_state.tasks_ongoing:
                # TODO : update the resource used dictionnary also, for the task that consumes a varying number
                # of ressource
                next_state.tasks_progress[task_id] += self.get_task_progress(
                    task_id,
                    t_from=next_state.t,
                    t_to=next_state.t + 1,
                    mode=next_state.tasks_mode[task_id],
                    sampled_duration=next_state.tasks_details[task_id].sampled_duration,
                )
        return next_states

    def update_progress_simulation(self, state: State):
        """In a simulation scheduling environment, update the progress of all ongoing tasks in the state."""
        return self.update_progress(state)

    def update_complete_tasks(self, state: State):
        """Update the status of newly completed tasks in the state from ongoing to complete
        and update resource availability. This function will also log in task_details the time it was complete"""
        next_state = state  # .copy()
        completed_tmp = []
        for task_id in next_state.tasks_ongoing:
            if next_state.tasks_progress[task_id] >= 0.9999:
                next_state.tasks_progress[task_id] = 1
                completed_tmp.append(task_id)
        for completed_task in completed_tmp:
            next_state.tasks_complete.add(completed_task)
            next_state.tasks_ongoing.remove(completed_task)
            for res in next_state.resource_used_for_task[completed_task]:
                res_consumption = next_state.resource_used_for_task[completed_task][res]
                if self.is_renewable(res):
                    next_state.resource_used[res] -= res_consumption
            next_state.resource_used_for_task.pop(completed_task)
            next_state.tasks_details[completed_task].end = next_state.t + 1
            # WARNING : considering how it's coded, we should put +1 here. could be ccleaner if it was done in the update_progress.
            next_state.tasks_complete_details.push_front(
                next_state.tasks_details[completed_task]
            )
            del next_state.tasks_details[completed_task]
            next_state.tasks_complete_progress.push_front(
                next_state.tasks_progress[completed_task]
            )
            del next_state.tasks_progress[completed_task]
            next_state.tasks_complete_mode.push_front(
                (completed_task, next_state.tasks_mode[completed_task])
            )
            del next_state.tasks_mode[completed_task]

        return next_state

    def update_complete_tasks_uncertain(self, states: DiscreteDistribution[State]):
        """In an uncertain scheduling environment, update the status of newly completed tasks in the state from ongoing
        to complete, update resource availability and update on-completion conditions.
        This function will also log in task_details the time it was complete."""
        next_states = DiscreteDistribution(
            [(state, prob) for (state, prob) in states.get_values()]
        )

        states_and_proba = []
        for next_state, _ in next_states.get_values():
            completed_tmp = []
            for task_id in next_state.tasks_ongoing:
                if next_state.tasks_progress[task_id] >= 0.9999:
                    next_state.tasks_progress[task_id] = 1
                    completed_tmp.append(task_id)

            all_values = []
            all_models = {}
            for completed_task in completed_tmp:
                next_state.tasks_complete.add(completed_task)
                next_state.tasks_ongoing.remove(completed_task)
                for res in next_state.resource_used_for_task[completed_task]:
                    res_consumption = next_state.resource_used_for_task[completed_task][
                        res
                    ]
                    if self.is_renewable(res):
                        next_state.resource_used[res] -= res_consumption
                next_state.resource_used_for_task.pop(completed_task)
                next_state.tasks_details[completed_task].end = (
                    next_state.t + 1
                )  # WARNING : considering how it's coded, we should put +1 here.
                next_state.tasks_complete_details.push_front(
                    next_state.tasks_details[completed_task]
                )
                del next_state.tasks_details[completed_task]
                next_state.tasks_complete_progress.push_front(
                    next_state.tasks_progress[completed_task]
                )
                del next_state.tasks_progress[completed_task]
                next_state.tasks_complete_mode.push_front(
                    (completed_task, next_state.tasks_mode[completed_task])
                )
                del next_state.tasks_mode[completed_task]
                if completed_task in self.get_task_on_completion_added_conditions():
                    all_models[completed_task] = []
                    for i in range(
                        len(
                            self.get_task_on_completion_added_conditions()[
                                completed_task
                            ]
                        )
                    ):
                        for j in range(
                            len(
                                self.get_task_on_completion_added_conditions()[
                                    completed_task
                                ][i].get_values()
                            )
                        ):
                            all_values.append(
                                {
                                    "task": completed_task,
                                    "cond": self.get_task_on_completion_added_conditions()[
                                        completed_task
                                    ][
                                        i
                                    ].get_values()[
                                        j
                                    ][
                                        0
                                    ],
                                    "prob": self.get_task_on_completion_added_conditions()[
                                        completed_task
                                    ][
                                        i
                                    ].get_values()[
                                        j
                                    ][
                                        1
                                    ],
                                }
                            )
                            all_models[completed_task].append(len(all_values) - 1)

            combinations = list(product(*all_models.values()))
            # for comb in combinations:
            #     next_state_2 = next_state.copy()
            #     proba = 1.
            #     for i in comb:
            #         next_state_2._current_conditions.add(all_values[i]['cond'])
            #         proba *= all_values[i]['prob']
            #     states_and_proba += [(next_state_2, proba)]
            for comb in combinations:
                do_copy = len(comb) > 0
                next_state_2 = next_state.copy() if do_copy else next_state
                proba = 1.0
                for i in comb:
                    next_state_2._current_conditions.add(all_values[i]["cond"])
                    proba *= all_values[i]["prob"]
                states_and_proba += [(next_state_2, proba)]
        return DiscreteDistribution(states_and_proba)

    def update_complete_tasks_simulation(self, state: State):
        """In a simulated scheduling environment, update the status of newly completed tasks in the state from ongoing to complete
        and update resource availability. This function will also log in task_details the time it was complete"""
        next_state: State = state
        next_state = self.update_complete_tasks_uncertain(
            states=DiscreteDistribution([(next_state, 1.0)])
        ).sample()
        return next_state

    def update_complete_dummy_tasks(self, state: State, action: SchedulingAction):
        """Update the status of newly started tasks whose duration is 0 from ongoing to complete."""
        next_state = state  # .copy()
        if action.action == SchedulingActionEnum.START:
            task = action.task
            if next_state.tasks_details[task].sampled_duration == 0:
                next_state.tasks_complete.add(task)
                next_state.tasks_progress[task] = 1
                next_state.tasks_ongoing.remove(task)
                next_state.tasks_details[task].end = next_state.t
                next_state.tasks_complete_details.push_front(
                    next_state.tasks_details[task]
                )
                del next_state.tasks_details[task]
                next_state.tasks_complete_progress.push_front(
                    next_state.tasks_progress[task]
                )
                del next_state.tasks_progress[task]
                next_state.tasks_complete_mode.push_front(
                    (task, next_state.tasks_mode[task])
                )
                del next_state.tasks_mode[task]
                next_state.resource_used_for_task.pop(task)
        return next_state

    def update_complete_dummy_tasks_uncertain(
        self, states: DiscreteDistribution[State], action: SchedulingAction
    ):
        """In an uncertain scheduling environment, update the status of newly started tasks whose duration is 0
        from ongoing to complete."""
        next_states = DiscreteDistribution(
            [(state, prob) for (state, prob) in states.get_values()]
        )
        if action.action == SchedulingActionEnum.START:
            task = action.task
            for next_state, _ in next_states.get_values():
                if next_state.tasks_details[task].sampled_duration == 0:
                    next_state.tasks_complete.add(task)
                    next_state.tasks_progress[task] = 1
                    next_state.tasks_ongoing.remove(task)
                    next_state.tasks_details[task].end = next_state.t
                    next_state.tasks_complete_details.push_front(
                        next_state.tasks_details[task]
                    )
                    del next_state.tasks_details[task]
                    next_state.tasks_complete_progress.push_front(
                        next_state.tasks_progress[task]
                    )
                    del next_state.tasks_progress[task]
                    next_state.tasks_complete_mode.push_front(
                        (task, next_state.tasks_mode[task])
                    )
                    del next_state.tasks_mode[task]
                    next_state.resource_used_for_task.pop(task)
        return next_states

    def update_complete_dummy_tasks_simulation(
        self, state: State, action: SchedulingAction
    ):
        """In a simulated scheduling environment, update the status of newly started tasks whose duration is 0
        from ongoing to complete."""
        return self.update_complete_dummy_tasks(state, action)

    def update_pause_tasks(self, state: State, action: SchedulingAction):
        """Update the status of a task from ongoing to paused if specified in the action
        and update resource availability. This function will also log in task_details the time it was paused."""
        next_state = state  # .copy()
        if action.action == SchedulingActionEnum.PAUSE:
            paused_task = action.task
            next_state.tasks_paused.add(paused_task)
            next_state.tasks_ongoing.remove(paused_task)
            # time_since_start = next_state.tasks_details[paused_task].get_task_active_time(next_state.t)
            for res in next_state.resource_used_for_task[paused_task]:
                res_consumption = next_state.resource_used_for_task[paused_task][res]
                if self.is_renewable(res) or (
                    not self.is_renewable(res)
                    and self.get_task_paused_non_renewable_resource_returned()[res]
                ):
                    next_state.resource_used[res] -= res_consumption
            next_state.resource_used_for_task.pop(paused_task)
            next_state.tasks_details[paused_task].paused.append(next_state.t)
            # Need to call this after get_task_active_time()
        return next_state

    def update_pause_tasks_uncertain(self, state: State, action: SchedulingAction):
        """In an uncertain scheduling environment, update the status of a task from ongoing to paused if
        specified in the action and update resource availability. This function will also log in task_details
        the time it was paused."""
        return self.update_pause_tasks(state, action)

    def update_pause_tasks_simulation(self, state: State, action: SchedulingAction):
        """In a simulation scheduling environment, update the status of a task from ongoing to paused if
        specified in the action and update resource availability. This function will also log in task_details
        the time it was paused."""
        return self.update_pause_tasks(state, action)

    def update_resume_tasks(self, state: State, action: SchedulingAction):
        """Update the status of a task from paused to ongoing if specified in the action
        and update resource availability. This function will also log in task_details the time it was resumed"""
        next_state = state  # .copy()
        if action.action == SchedulingActionEnum.RESUME:
            resumed_task = action.task
            mode = action.mode  # use mode specified in action if any
            if mode is None:
                mode = state.tasks_mode[resumed_task]  # or use previous mode
            next_state.tasks_ongoing.add(resumed_task)
            next_state.tasks_paused.remove(resumed_task)
            next_state.tasks_details[resumed_task].resumed.append(
                next_state.t
            )  # Need to call this before get_task_active_time()
            b, resource_to_use = self.check_if_action_can_be_started(
                next_state, action=action
            )
            if not b:
                return next_state
            for res in resource_to_use:
                next_state.resource_used[res] += resource_to_use[res]
            next_state.resource_used_for_task[resumed_task] = resource_to_use
            # self.get_latest_sampled_duration(task=resumed_task , mode=mode,
            #                                 progress_from=next_state.tasks_progress[resumed_task])  # TODO: what to do with this, so far the sample is stored and then used by get_task_progress()
            # Out of scope normally, there shouldn't be a "get_resource_need_at_time" function
            # for ressource unit, or it doesn't make completely sense as of today
            # for res in self.get_resource_units_names():
            #     res_consumption = self.get_tasks_modes()[resumed_task][next_state.tasks_mode[resumed_task]]\
            #         .get_resource_need_at_time(resource_name=res, time=time_since_start)
            #     # next_state.resource_availability[res] -= res_consumption
            #     next_state.resource_used[res] += res_consumption
            #     if resumed_task not in next_state.resource_used_for_task:
            #         next_state.resource_used_for_task[resumed_task] = {}
            #     next_state.resource_used_for_task[resumed_task][res] = res_consumption
            # for res in self.get_resource_types_names():
            #     res_consumption = self.get_tasks_modes()[resumed_task][next_state.tasks_mode[resumed_task]]\
            #         .get_resource_need_at_time(resource_name=res, time=time_since_start)
            #     # next_state.resource_availability[res] -= res_consumption
            #     next_state.resource_used[res] += res_consumption
            #     if resumed_task not in next_state.resource_used_for_task:
            #         next_state.resource_used_for_task[resumed_task] = {}
            #     next_state.resource_used_for_task[resumed_task][res] = res_consumption
            # if action.resource_unit_names is not None:
            #     for resource_unit_name in action.resource_unit_names:
            #         next_state.resource_used[resource_unit_name] = 1
            #         next_state.resource_used_for_task[resumed_task][resource_unit_name] = 1
        return next_state

    def update_resume_tasks_uncertain(self, state: State, action: SchedulingAction):
        """In an uncertain scheduling environment, update the status of a task from paused to ongoing if specified
        in the action and update resource availability. This function will also log in task_details the time it was
        resumed."""
        return self.update_resume_tasks(state, action)

    def update_resume_tasks_simulation(self, state: State, action: SchedulingAction):
        """In a simulationn scheduling environment, update the status of a task from paused to ongoing if specified
        in the action and update resource availability. This function will also log in task_details the time it was
        resumed."""
        return self.update_resume_tasks(state, action)

    def check_if_action_can_be_started(
        self, state: State, action: SchedulingAction
    ) -> Tuple[bool, Dict[str, int]]:
        """Check if a start or resume action can be applied. It returns a boolean and a dictionary of resources to use."""
        started_task = action.task
        if action.action == SchedulingActionEnum.START:
            time_since_start = state.t
        elif action.action == SchedulingActionEnum.RESUME:
            time_since_start = state.tasks_details[started_task].get_task_active_time(
                state.t
            )
        else:
            return True, {}
        resource_to_use = self.get_resource_used(
            task=started_task,
            mode=action.mode,
            resource_unit_names=action.resource_unit_names,
            time_since_start=time_since_start,
        )
        if any(
            resource_to_use[r] > state.resource_availability[r] - state.resource_used[r]
            for r in resource_to_use
        ):
            return False, resource_to_use
        b = self.check_if_skills_are_fulfilled(
            task=started_task, mode=action.mode, resource_used=resource_to_use
        )
        return b, resource_to_use

    def get_resource_used(
        self, task: int, mode: int, resource_unit_names: Set[str], time_since_start: int
    ):
        r_used = {}
        mode_details = self.get_tasks_modes()
        for res in self.get_resource_types_names():
            res_consumption = mode_details[task][mode].get_resource_need_at_time(
                resource_name=res, time=time_since_start
            )
            # next_state.resource_availability[res] -= res_consumption
            r_used[res] = res_consumption
        if resource_unit_names is not None:
            for resource_unit_name in resource_unit_names:
                r_used[resource_unit_name] = 1
        return r_used

    def update_start_tasks(self, state: State, action: SchedulingAction):
        """Update the status of a task from remaining to ongoing if specified in the action
        and update resource availability. This function will also log in task_details the time it was started."""
        if action.action == SchedulingActionEnum.START:
            can_be_started, resource_to_use = self.check_if_action_can_be_started(
                state, action
            )
            if not can_be_started:
                return state
            next_state: State = state  # .copy()
            started_task = action.task
            mode = action.mode
            next_state.tasks_mode[started_task] = mode
            next_state.tasks_ongoing.add(started_task)
            sampled_duration = self.get_latest_sampled_duration(
                task=started_task, mode=mode, progress_from=0.0
            )  # TODO: what to do with this, so far the sample is stored and then used by get_task_progress()
            next_state.tasks_details[started_task] = Task(
                started_task, next_state.t, sampled_duration
            )
            for res in resource_to_use:
                next_state.resource_used[res] += resource_to_use[res]
            next_state.resource_used_for_task[started_task] = resource_to_use
            next_state.tasks_progress[started_task] = 0.0
            return next_state
        return state

    def update_start_tasks_uncertain(self, state: State, action: SchedulingAction):
        """In an uncertain scheduling environment, update the status of a task from remaining to ongoing
        if specified in the action and update resource availability.
        This function returns a DsicreteDistribution of State.
        This function will also log in task_details the time it was started."""

        if action.action == SchedulingActionEnum.START:
            can_be_started, resource_to_use = self.check_if_action_can_be_started(
                state, action
            )
            if not can_be_started:
                return DiscreteDistribution([(state, 1)])
            started_task = action.task
            mode = action.mode
            duration_distrib = self.get_task_duration_distribution(
                started_task, mode, multivariate_settings={"t": state.t}
            )
            states_and_proba = []

            for value_duration in duration_distrib.get_values():
                next_state: State = state.copy()
                next_state.tasks_mode[started_task] = mode
                next_state.tasks_ongoing.add(started_task)
                next_state.tasks_details[started_task] = Task(
                    started_task, state.t, value_duration[0]
                )
                for res in resource_to_use:
                    next_state.resource_used[res] += resource_to_use[res]
                next_state.resource_used_for_task[started_task] = resource_to_use
                next_state.tasks_progress[started_task] = 0.0

                states_and_proba += [(next_state, value_duration[1])]

            return DiscreteDistribution(states_and_proba)
        return DiscreteDistribution([(state, 1)])

    def update_start_tasks_simulation(self, state: State, action: SchedulingAction):
        """In a simulated scheduling environment, update the status of a task from remaining to ongoing if
        specified in the action and update resource availability. This function will also log in task_details the
        time it was started."""
        return self.update_start_tasks(state, action)

    def get_possible_starting_tasks(self, state: State):

        mode_details = self.get_tasks_modes()
        possible_task_precedence = [
            (n, mode_details[n])
            for n in state.tasks_remaining
            if all(
                m in state.tasks_complete
                for m in set(self.ancestors[n]).intersection(
                    self.get_available_tasks(state)
                )
            )
        ]

        possible_task_with_ressource = [
            (n, mode, mode_consumption)
            for n, modes in possible_task_precedence
            for mode, mode_consumption in modes.items()
            if all(
                state.resource_availability[key]
                - state.resource_used[key]
                - mode_consumption.get_resource_need_at_time(
                    resource_name=key, time=state.t
                )
                >= 0
                for key in self.get_resource_types_names()
            )
        ]
        # print("Possible task with ressource : ", possible_task_with_ressource)
        return {
            n: {mode: mode_consumption.get_non_zero_ressource_need_names(0)}
            for n, mode, mode_consumption in possible_task_with_ressource
        }

    def get_possible_resume_tasks(self, state: State):
        mode_details = self.get_tasks_modes()
        possible_task_precedence = [
            (n, mode_details[n])
            for n in state.tasks_paused
            if all(m in state.tasks_complete for m in self.ancestors[n])
        ]
        # print("Possible task precedence : ", possible_task_precedence)
        possible_task_with_ressource = [
            (n, mode, mode_consumption)
            for n, modes in possible_task_precedence
            for mode, mode_consumption in modes.items()
            if all(
                state.resource_availability[key]
                - state.resource_used[key]
                - mode_consumption.get_resource_need_at_time(
                    resource_name=key, time=state.t
                )
                >= 0
                for key in self.get_resource_types_names()
            )
        ]
        # print("Possible task with ressource : ", possible_task_with_ressource)
        return {
            n: {mode: mode_consumption.get_non_zero_ressource_need_names(0)}
            for n, mode, mode_consumption in possible_task_with_ressource
        }

    def state_is_overconsuming(self, state: State):
        return any(
            state.resource_used[k] > state.resource_availability[k]
            for k in state.resource_used
        )

    def update_resource_availability(self, state: State, action: SchedulingAction):
        """Update resource availability for next time step. This should be called after update_time()."""
        if action.time_progress:
            next_state: State = state  # .copy()
            # update the resource used / resource availability function on the possible new availability
            # and consumption of ongoing task -> quite boring to code and debug probably
            for res in self.get_resource_units_names():
                next_state.resource_availability[res] = self.sample_quantity_resource(
                    resource=res, time=next_state.t
                )
            for res in self.get_resource_types_names():
                next_state.resource_availability[res] = self.sample_quantity_resource(
                    resource=res, time=next_state.t
                )
            # TODO :
            # Here, if the resource_used[res] is > resource_availability[res] we should be forced to pause some task??
            # If yes which one ? all ? and we let the algorithm resume the one of its choice in the next time step ?
            return next_state
        return state

    def update_resource_availability_uncertain(
        self, states: DiscreteDistribution[State], action: SchedulingAction
    ):
        """Update resource availability for next time step. This should be called after update_time()."""
        next_states = DiscreteDistribution(
            [(state, prob) for (state, prob) in states.get_values()]
        )

        if action.time_progress:
            for next_state, _ in next_states.get_values():
                # update the resource used / resource availability function on the possible new availability
                # and consumption of ongoing task -> quite boring to code and debug probably
                for res in self.get_resource_units_names():
                    next_state.resource_availability[
                        res
                    ] = self.sample_quantity_resource(resource=res, time=next_state.t)
                for res in self.get_resource_types_names():
                    next_state.resource_availability[
                        res
                    ] = self.sample_quantity_resource(resource=res, time=next_state.t)
                # TODO :
                # Here, if the resource_used[res] is > resource_availability[res] we should be forced to pause some task??
                # If yes which one ? all ? and we let the algorithm resume the one of its choice in the next time step ?
        return next_states

    def update_resource_availability_simulation(
        self, state: State, action: SchedulingAction
    ):
        """In a simulated scheduling environment, update resource availability for next time step.
        This should be called after update_time()."""
        return self.update_resource_availability(state, action)

    def update_conditional_tasks(self, state: State, action: SchedulingAction):
        """Update remaining tasks by checking conditions and potentially adding conditional tasks."""
        return state

    def update_conditional_tasks_uncertain(
        self, states: DiscreteDistribution[State], action: SchedulingAction
    ):
        """Update remaining tasks by checking conditions and potentially adding conditional tasks."""
        next_states = DiscreteDistribution(
            [(state, prob) for (state, prob) in states.get_values()]
        )

        if action.time_progress:
            for next_state, _ in next_states.get_values():
                all_available_tasks = self.get_available_tasks(next_state)
                all_considered_tasks = next_state.task_ids.difference(
                    next_state.tasks_unsatisfiable
                )
                new_tasks = all_available_tasks.symmetric_difference(
                    all_considered_tasks
                )
                for task in new_tasks:
                    next_state.tasks_unsatisfiable.remove(task)
            return next_states
        return next_states

    def update_conditional_tasks_simulation(
        self, state: State, action: SchedulingAction
    ):
        """In a simulated scheduling environment, update remaining tasks by checking conditions and potentially
        adding conditional tasks."""
        next_state: State = state
        next_state = self.update_conditional_tasks_uncertain(
            states=DiscreteDistribution([(next_state, 1.0)]), action=action
        ).sample()
        return next_state

    # WARNING : the two following functions are not required with the current signature of the Domain
    # BUT i feel it's necessary to code it somewhere when i tested it in skdecide/hub/domain/rcpsp_pi2/rcpsp where
    # i had to code those two functions.
    #
    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:

        transition_makespan = 0.0
        transition_cost = 0.0

        if SchedulingObjectiveEnum.MAKESPAN in self.get_objectives():
            transition_makespan = next_state.t - memory.t

        if SchedulingObjectiveEnum.COST in self.get_objectives():
            mode_cost_val = 0.0
            if action.action == SchedulingActionEnum.START:
                mode_cost_val += self.get_mode_costs()[action.task][action.mode]
            renewable_type_cost_val = 0.0
            nonrenewable_type_cost_val = 0.0
            for res in self.get_resource_types_names():
                if self.is_renewable(res):
                    renewable_type_cost_val += (
                        self.get_resource_cost_per_time_unit()[res]
                        * next_state.resource_used[res]
                    )
                else:
                    nonrenewable_type_cost_val += (
                        next_state.resource_used[res] - memory.resource_used[res]
                    )  # res used not decreased for NR resources so need to compute difference from previous state
            renewable_unit_cost_val = 0.0
            nonrenewable_unit_cost_val = 0.0
            for res in self.get_resource_units_names():
                if self.is_renewable(res):
                    renewable_unit_cost_val += (
                        self.get_resource_cost_per_time_unit()[res]
                        * next_state.resource_used[res]
                    )
                else:
                    nonrenewable_unit_cost_val += (
                        next_state.resource_used[res] - memory.resource_used[res]
                    )  # res used not decreased for NR resources so need to compute difference from previous state
            transition_cost = (
                mode_cost_val
                + renewable_type_cost_val
                + nonrenewable_type_cost_val
                + renewable_unit_cost_val
                + nonrenewable_unit_cost_val
            )

        # TODO: Handle more than 1 objective in the transition value (need weights ?)
        weighed_transition_cost = 1.0 * transition_makespan + 1.0 * transition_cost
        return Value(cost=weighed_transition_cost)

    def _is_terminal(self, state: D.T_state) -> bool:
        all_task_possible = self.all_tasks_possible(state)
        return (
            state.t > self.get_max_horizon()
            or (not all_task_possible)
            or (
                len(state.task_ids)
                == len(state.tasks_complete) + len(state.tasks_unsatisfiable)
                and (len(state.tasks_ongoing) == 0)
                and (len(state.tasks_paused) == 0)
            )
        )
        # TODO, is there a cleaner way ? We can check completion of the sink task

    def get_objectives(self) -> List[SchedulingObjectiveEnum]:
        """Return the objectives to consider as a list. The items should be of SchedulingObjectiveEnum type."""
        return self._get_objectives()

    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        """Return the objectives to consider as a list. The items should be of SchedulingObjectiveEnum type."""
        raise NotImplementedError


class D_det(
    Domain,
    SingleAgent,
    Sequential,
    DeterministicTransitions,
    DeterministicInitialized,
    Actions,
    FullyObservable,
    Markovian,
    Goals,
):
    """Base class for deterministic scheduling problems"""

    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = SchedulingAction  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class D_uncertain(
    Domain,
    SingleAgent,
    Sequential,
    UncertainTransitions,
    DeterministicInitialized,
    Actions,
    FullyObservable,
    Markovian,
    Goals,
):
    """Base class for uncertain scheduling problems where we can compute distributions"""

    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = SchedulingAction  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


# Specific for uncertain domain
class UncertainSchedulingDomain(SchedulingDomain, D_uncertain):
    """This is the highest level scheduling domain class (inheriting top-level class for each mandatory
    domain characteristic).
    """

    def _state_sample(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> TransitionOutcome[D.T_state, D.T_agent[Value[D.T_value]], D.T_agent[D.T_info]]:
        """This function will be used if the domain is defined as a Simulation (i.e. transitions are defined by call to
        a simulation). This function may also be used by simulation-based solvers on non-Simulation domains."""
        return UncertainTransitions._state_sample(self, memory, action)


# Specific for deterministic domain
class DeterministicSchedulingDomain(SchedulingDomain, D_det):
    """This is the highest level scheduling domain class (inheriting top-level class for each mandatory
    domain characteristic).
    """

    def _state_sample(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> TransitionOutcome[D.T_state, D.T_agent[Value[D.T_value]], D.T_agent[D.T_info]]:
        """This function will be used if the domain is defined as a Simulation (i.e. transitions are defined by call to
        a simulation). This function may also be used by simulation-based solvers on non-Simulation domains."""
        return DeterministicTransitions._state_sample(self, memory, action)

    def _get_next_state_distribution(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> Distribution[D.T_state]:
        """This function will be used if the domain is defined with UncertainTransitions. This function will be ignored
        if the domain is defined as a Simulation. This function may also be used by uncertainty-specialised solvers
         on deterministic domains."""
        return DeterministicTransitions._get_next_state_distribution(
            self, memory, action
        )


"""
Scheduling action space that will work for domains that don't require any
ressource unit allocation in its formulation.
"""


class SchedulingActionSpace(
    EnumerableSpace[SchedulingAction], SamplableSpace[SchedulingAction]
):
    def __init__(self, domain: SchedulingDomain, state: State):
        self.domain = domain
        self.state = state
        self.elements = self._get_elements()

    def _get_elements(self) -> Iterable[T]:
        choices = [
            SchedulingActionEnum.START,
            SchedulingActionEnum.PAUSE,
            SchedulingActionEnum.RESUME,
            SchedulingActionEnum.TIME_PR,
        ]
        list_action = []
        if self.domain.state_is_overconsuming(self.state):
            choices = [
                SchedulingActionEnum.PAUSE,
                SchedulingActionEnum.TIME_PR,
            ]  # we have to pause some task before doing anything
        for choice in choices:
            if choice == SchedulingActionEnum.START:
                # task, mode, list of Ressources
                task_possible_to_start: Dict[
                    int, Dict[int, List[str]]
                ] = self.domain.get_possible_starting_tasks(self.state)
                list_action += [
                    SchedulingAction(
                        task=t,
                        action=SchedulingActionEnum.START,
                        mode=m,
                        time_progress=False,
                    )
                    for t in task_possible_to_start
                    for m in task_possible_to_start[t]
                ]
            if choice == SchedulingActionEnum.PAUSE:
                task_possible_to_pause = self.state.tasks_ongoing
                list_action += [
                    SchedulingAction(
                        task=t,
                        action=SchedulingActionEnum.PAUSE,
                        mode=None,
                        time_progress=False,
                    )
                    for t in task_possible_to_pause
                    if self.domain.get_task_preemptivity()[t]
                ]
            if choice == SchedulingActionEnum.RESUME:
                task_possible_to_resume = self.state.tasks_paused
                list_action += [
                    SchedulingAction(
                        task=t,
                        action=SchedulingActionEnum.RESUME,
                        mode=self.state.tasks_mode[t],
                        time_progress=False,
                    )
                    for t in task_possible_to_resume
                ]
            if choice == SchedulingActionEnum.TIME_PR:
                list_action.append(
                    SchedulingAction(
                        task=None,
                        action=SchedulingActionEnum.TIME_PR,
                        mode=None,
                        time_progress=True,
                    )
                )
        return list_action

    def get_elements(self) -> Iterable[T]:
        return self.elements

    def sample(self) -> T:
        return random.choice(self.elements)


# TODO : will work well for domains where a task can be done by one ressource unit (and not a combination of ressource units)
# In this case we can enumerate the "worker" ressource. This is the case of the MultiSkill RCPSP of the Imopse benchmark
class SchedulingActionSpaceWithResourceUnit(
    EnumerableSpace[SchedulingAction], SamplableSpace[SchedulingAction]
):
    def __init__(self, domain: SchedulingDomain, state: State):
        self.domain = domain
        self.state = state
        self.elements = self._get_elements()

    def _get_elements(self) -> Iterable[T]:
        choices = [
            SchedulingActionEnum.START,
            SchedulingActionEnum.PAUSE,
            SchedulingActionEnum.RESUME,
            SchedulingActionEnum.TIME_PR,
        ]
        list_action = []
        if self.domain.state_is_overconsuming(self.state):
            choices = [
                SchedulingActionEnum.PAUSE,
                SchedulingActionEnum.TIME_PR,
            ]  # we have to pause some task before doing anything
        for choice in choices:
            if choice == SchedulingActionEnum.START:
                # task, mode, list of Ressources
                task_possible_to_start: Dict[
                    int, Dict[int, List[str]]
                ] = self.domain.get_possible_starting_tasks(self.state)
                for possible_to_start in task_possible_to_start:
                    for mode in task_possible_to_start[possible_to_start]:
                        possible = self.domain.find_one_ressource_to_do_one_task(
                            task=possible_to_start, mode=mode
                        )
                        list_action += [
                            SchedulingAction(
                                task=possible_to_start,
                                action=SchedulingActionEnum.START,
                                mode=mode,
                                time_progress=False,
                                resource_unit_names={r} if r is not None else None,
                            )
                            for r in possible
                            if r is None
                            or self.state.resource_used[r] + 1
                            <= self.state.resource_availability[r]
                        ]

            if choice == SchedulingActionEnum.PAUSE:
                task_possible_to_pause = self.state.tasks_ongoing
                list_action += [
                    SchedulingAction(
                        task=t,
                        action=SchedulingActionEnum.PAUSE,
                        mode=None,
                        time_progress=False,
                    )
                    for t in task_possible_to_pause
                    if self.domain.get_task_preemptivity()[t]
                ]
            if choice == SchedulingActionEnum.RESUME:
                task_possible_to_resume = self.state.tasks_paused

                for possible_to_resume in task_possible_to_resume:
                    possible = self.domain.find_one_ressource_to_do_one_task(
                        task=possible_to_resume,
                        mode=self.state.tasks_mode[possible_to_resume],
                    )

                    list_action += [
                        SchedulingAction(
                            task=possible_to_resume,
                            action=SchedulingActionEnum.START,
                            mode=self.state.tasks_mode[possible_to_resume],
                            time_progress=False,
                            resource_unit_names={r} if r is not None else None,
                        )
                        for r in possible
                        if r is None
                        or self.state.resource_used[r] + 1
                        <= self.state.resource_availability[r]
                    ]
            if choice == SchedulingActionEnum.TIME_PR:
                list_action.append(
                    SchedulingAction(
                        task=None,
                        action=SchedulingActionEnum.TIME_PR,
                        mode=None,
                        time_progress=True,
                    )
                )
        return list_action

    def get_elements(self) -> Iterable[T]:
        return self.elements

    def sample(self) -> T:
        return random.choice(self.elements)


class SchedulingActionSpaceWithResourceUnitSamplable(SamplableSpace[SchedulingAction]):
    def __init__(self, domain: SchedulingDomain, state: State):
        self.domain = domain
        self.state = state

    def sample(self) -> T:
        choices = [
            SchedulingActionEnum.START,
            SchedulingActionEnum.PAUSE,
            SchedulingActionEnum.RESUME,
            SchedulingActionEnum.TIME_PR,
        ]
        if self.domain.state_is_overconsuming(self.state):
            choices = [
                SchedulingActionEnum.PAUSE
            ]  # we have to pause some task before doing anything
        random_choice = random.choice(choices)
        if random_choice in {SchedulingActionEnum.START, SchedulingActionEnum.RESUME}:
            if random_choice == SchedulingActionEnum.START:
                task_possible_to_start: Dict[
                    int, Dict[int, List[str]]
                ] = self.domain.get_possible_starting_tasks(self.state)
            else:
                task_possible_to_start: Dict[
                    int, Dict[int, List[str]]
                ] = self.domain.get_possible_resume_tasks(self.state)
            task_modes = [
                (t, m)
                for t in task_possible_to_start
                for m in task_possible_to_start[t]
            ]
            if len(task_modes) > 0:
                random_ch = random.choice(task_modes)
                possible_to_start = random_ch[0]
                mode = random_ch[1]
                skill_of_task = self.domain.get_skills_of_task(possible_to_start, mode)
                resources = []
                if len(skill_of_task) == 0:
                    return SchedulingAction(
                        task=possible_to_start,
                        action=random_choice,
                        mode=mode,
                        time_progress=False,
                    )
                resources_skills = list(self.domain.get_all_resources_skills().keys())
                random.shuffle(resources_skills)
                cur_usage = {s: 0 for s in skill_of_task}
                success = False
                for resource in resources_skills:
                    if (
                        self.state.resource_used[resource] + 1
                        > self.state.resource_availability[resource]
                    ):
                        continue
                    if any(
                        self.domain.get_skills_of_resource(resource=resource).get(s, 0)
                        > 0
                        for s in skill_of_task
                    ):
                        resources += [resource]
                        for s in cur_usage:
                            cur_usage[s] += self.domain.get_skills_of_resource(
                                resource=resource
                            ).get(s, 0)
                        if all(cur_usage[s] >= skill_of_task[s] for s in skill_of_task):
                            success = True
                            break
                if success:
                    return SchedulingAction(
                        task=possible_to_start,
                        action=random_choice,
                        mode=mode,
                        time_progress=False,
                        resource_unit_names=resources,
                    )
                else:
                    return SchedulingAction(
                        task=None,
                        action=SchedulingActionEnum.TIME_PR,
                        mode=None,
                        time_progress=True,
                    )
            else:
                return SchedulingAction(
                    task=None,
                    action=SchedulingActionEnum.TIME_PR,
                    mode=None,
                    time_progress=True,
                )

        if random_choice == SchedulingActionEnum.PAUSE:
            task_possible_to_pause = [
                t
                for t in self.state.tasks_ongoing
                if self.domain.get_task_preemptivity()[t]
            ]
            if len(task_possible_to_pause) > 0:
                return SchedulingAction(
                    task=random.choice(task_possible_to_pause),
                    action=SchedulingActionEnum.PAUSE,
                    mode=None,
                    time_progress=False,
                )
            else:

                return SchedulingAction(
                    task=None,
                    action=SchedulingActionEnum.TIME_PR,
                    mode=None,
                    time_progress=True,
                )

        if random_choice == SchedulingActionEnum.TIME_PR:
            return SchedulingAction(
                task=None,
                action=SchedulingActionEnum.TIME_PR,
                mode=None,
                time_progress=True,
            )

    def contains(self, x: T) -> bool:
        return True


class SingleModeRCPSP(
    DeterministicSchedulingDomain,
    SingleMode,
    DeterministicTaskDuration,
    DeterministicTaskProgress,
    WithoutResourceUnit,
    WithoutPreallocations,
    WithoutTimeLag,
    WithoutTimeWindow,
    WithoutResourceSkills,
    WithoutResourceAvailabilityChange,
    WithoutConditionalTasks,
    RenewableOnly,
    ConstantResourceConsumption,  # problem with unimplemented classes with this
    WithoutPreemptivity,  # problem with unimplemented classes with this
    WithoutModeCosts,
    WithoutResourceCosts,
):
    """
    Single mode (classic) Resource project scheduling problem template.
    It consists in :
    - a deterministic scheduling problem with precedence constraint between task
    - a set of renewable resource with constant availability (capacity)
    - task having deterministic resource consumption
    The goal is to minimize the overall makespan, respecting the cumulative resource consumption constraint
    """

    pass


class SingleModeRCPSPCalendar(
    DeterministicSchedulingDomain,
    SingleMode,
    DeterministicTaskDuration,
    DeterministicTaskProgress,
    WithoutResourceUnit,
    WithoutPreallocations,
    WithoutTimeLag,
    WithoutTimeWindow,
    WithoutResourceSkills,
    # WithoutResourceAvailabilityChange,
    DeterministicResourceAvailabilityChanges,
    WithoutConditionalTasks,
    RenewableOnly,
    ConstantResourceConsumption,  # problem with unimplemented classes with this
    WithoutPreemptivity,  # problem with unimplemented classes with this
    WithoutModeCosts,
    WithoutResourceCosts,
):
    """
    Single mode Resource project scheduling problem with varying resource availability template.
    It consists in :
    - a deterministic scheduling problem with precedence constraint between task
    - a set of renewable resource with variable availability through time
    - task having deterministic resource consumption
    The goal is to minimize the overall makespan, respecting the cumulative resource consumption constraint
    at any time
    """

    pass


class MultiModeRCPSP(
    DeterministicSchedulingDomain,
    MultiMode,  # this changed from Single mode RCPSP
    DeterministicTaskDuration,
    DeterministicTaskProgress,
    WithoutResourceUnit,
    WithoutPreemptivity,
    WithoutPreallocations,
    WithoutTimeLag,
    WithoutTimeWindow,
    WithoutResourceSkills,
    WithoutResourceAvailabilityChange,
    WithoutConditionalTasks,
    ConstantResourceConsumption,
    MixedRenewable,  # this changed from Single mode RCPSP
    WithoutModeCosts,
    WithoutResourceCosts,
):
    """
    Multimode (classic) Resource project scheduling problem template.
    It consists in :
    - a deterministic scheduling problem with precedence constraint between task
    - a set of renewable resource with constant availability (capacity)
    - a set of non-renewable resource (consumable)
    - task having several modes of execution, giving for each mode a deterministic resource consumption and duration
    The goal is to minimize the overall makespan, respecting the cumulative resource consumption constraint
    """

    pass


class MultiModeRCPSPWithCost(
    DeterministicSchedulingDomain,
    MultiMode,
    DeterministicTaskDuration,
    DeterministicTaskProgress,
    WithoutResourceUnit,
    WithoutPreemptivity,
    WithoutPreallocations,
    WithoutTimeLag,
    WithoutTimeWindow,
    WithoutResourceSkills,
    WithoutResourceAvailabilityChange,
    WithoutConditionalTasks,
    ConstantResourceConsumption,
    MixedRenewable,
    WithModeCosts,
    WithResourceCosts,
):
    """
    Multimode (classic) Resource project scheduling problem template with cost based on modes.
    It consists in :
    - a deterministic scheduling problem with precedence constraint between task
    - a set of renewable resource with constant availability (capacity)
    - a set of non-renewable resource (consumable)
    - task having several modes of execution, giving for each mode a deterministic resource consumption and duration
    The goal is to minimize the overall cost that is function of the mode chosen for each task
    """

    pass


class MultiModeRCPSPCalendar(
    DeterministicSchedulingDomain,
    MultiMode,  # this changed from Single mode RCPSP
    DeterministicTaskDuration,
    DeterministicTaskProgress,
    WithoutResourceUnit,
    WithoutPreemptivity,
    WithoutPreallocations,
    WithoutTimeLag,
    WithoutTimeWindow,
    WithoutResourceSkills,
    DeterministicResourceAvailabilityChanges,
    WithoutConditionalTasks,
    ConstantResourceConsumption,
    MixedRenewable,  # this changed from Single mode RCPSP
    WithoutModeCosts,
    WithoutResourceCosts,
):
    """
    Multimode (classic) Resource project scheduling problem template with cost based on modes.
    It consists in :
    - a deterministic scheduling problem with precedence constraint between task
    - a set of renewable resource with variable availability (capacity)
    - a set of non-renewable resource (consumable)
    - task having several modes of execution, giving for each mode a deterministic resource consumption and duration
    The goal is to minimize the overall makespan
    """

    pass


class MultiModeRCPSPCalendar_Stochastic_Durations(
    UncertainSchedulingDomain,
    MultiMode,  # this changed from Single mode RCPSP
    UncertainUnivariateTaskDuration,
    DeterministicTaskProgress,
    WithoutResourceUnit,
    WithoutPreemptivity,
    WithoutPreallocations,
    WithoutTimeLag,
    WithoutTimeWindow,
    WithoutResourceSkills,
    # WithoutResourceAvailabilityChange,
    DeterministicResourceAvailabilityChanges,
    WithoutConditionalTasks,
    ConstantResourceConsumption,
    MixedRenewable,  # this changed from Single mode RCPSP
    WithoutModeCosts,
    WithoutResourceCosts,
):
    """
    Multimode (classic) Resource project scheduling problem template.
    It consists in :
    - a deterministic scheduling problem with precedence constraint between task
    - a set of renewable resource with variable availability (capacity)
    - a set of non-renewable resource (consumable)
    - task having several modes of execution, giving for each mode a deterministic resource consumption and
    a stochastic duration
    The goal is to minimize the overall makespan
    """

    pass


class MultiModeMultiSkillRCPSP(
    DeterministicSchedulingDomain,
    MultiMode,
    DeterministicTaskDuration,
    DeterministicTaskProgress,
    # WithResourceUnits,
    # WithResourceTypes,
    WithoutPreemptivity,
    WithoutPreallocations,
    WithoutTimeLag,
    WithoutTimeWindow,
    # WithResourceSkills, # This change from MultiModeRCPSP
    WithoutResourceAvailabilityChange,
    WithoutConditionalTasks,
    ConstantResourceConsumption,
    MixedRenewable,
    WithoutModeCosts,
    WithoutResourceCosts,
):
    """
    Multimode multiskill Resource project scheduling problem template
    It consists in :
    - a deterministic scheduling problem with precedence constraint between task
    - a set of renewable resource with constant availability (capacity)
    - resource can be unitary and have skills
    - a set of non-renewable resource (consumable)
    - task having several modes of execution, giving for each mode a deterministic resource consumption,
    deterministic duration and skills needed
    The goal is to minimize the overall makespan, allocating unit resource to tasks fulfilling the skills requirement.
    """

    pass


class MultiModeMultiSkillRCPSPCalendar(
    DeterministicSchedulingDomain,
    MultiMode,
    DeterministicTaskDuration,
    DeterministicTaskProgress,
    # WithResourceUnits,
    # WithResourceTypes,
    WithoutPreemptivity,
    WithoutPreallocations,
    WithoutTimeLag,
    WithoutTimeWindow,
    # WithResourceSkills, # This change from MultiModeRCPSP
    # WithoutResourceAvailabilityChange,
    DeterministicResourceAvailabilityChanges,
    WithoutConditionalTasks,
    ConstantResourceConsumption,
    MixedRenewable,
    WithoutModeCosts,
    WithoutResourceCosts,
):
    """
    Multimode multiskill Resource project scheduling problem with resource variability template
    It consists in :
    - a deterministic scheduling problem with precedence constraint between task
    - a set of renewable resource with variable availability
    - resource can be unitary and have skills
    - a set of non-renewable resource (consumable)
    - task having several modes of execution, giving for each mode a deterministic resource consumption,
    deterministic duration and skills needed
    The goal is to minimize the overall makespan, allocating unit resource to tasks fulfilling the skills requirement.
    """

    pass


class MultiModeRCPSP_Stochastic_Durations(
    UncertainSchedulingDomain,
    UncertainUnivariateTaskDuration,  # this changed from Single mode RCPSP
    DeterministicTaskProgress,
    WithoutResourceUnit,
    WithoutPreemptivity,
    WithoutPreallocations,
    WithoutTimeLag,
    WithoutTimeWindow,
    WithoutResourceSkills,
    WithoutResourceAvailabilityChange,
    WithoutConditionalTasks,
    ConstantResourceConsumption,
    RenewableOnly,
    WithoutModeCosts,
    WithoutResourceCosts,
):
    """
    Multimode Resource project scheduling problem with stochastic durations template.
    It consists in :
    - a scheduling problem with precedence constraint between task
    - a set of renewable resource with constant availability (capacity)
    - a set of non-renewable resource (consumable)
    - task having several modes of execution, giving for each mode a deterministic resource consumption and
    a stochastic duration
    The goal is to minimize the overall expected makespan
    """

    pass


class SingleModeRCPSP_Stochastic_Durations(
    UncertainSchedulingDomain,
    SingleMode,
    UncertainUnivariateTaskDuration,  # this changed from Single mode RCPSP
    DeterministicTaskProgress,
    WithoutResourceUnit,
    WithoutPreemptivity,
    WithoutPreallocations,
    WithoutTimeLag,
    WithoutTimeWindow,
    WithoutResourceSkills,
    WithoutResourceAvailabilityChange,
    WithoutConditionalTasks,
    ConstantResourceConsumption,
    RenewableOnly,
    WithoutModeCosts,
    WithoutResourceCosts,
):
    """
    Resource project scheduling problem template.
    It consists in :
    - a scheduling problem with precedence constraint between task
    - a set of renewable resource with constant availability (capacity)
    - task having a deterministic resource consumption and a stochastic duration
    The goal is to minimize the overall expected makespan
    """

    pass


class SingleModeRCPSP_Stochastic_Durations_WithConditionalTasks(
    UncertainSchedulingDomain,
    SingleMode,
    UncertainUnivariateTaskDuration,  # this changed from Single mode RCPSP
    DeterministicTaskProgress,
    WithoutResourceUnit,
    WithoutPreemptivity,
    WithoutPreallocations,
    WithoutTimeLag,
    WithoutTimeWindow,
    WithoutResourceSkills,
    WithoutResourceAvailabilityChange,
    # WithConditionalTasks,
    ConstantResourceConsumption,
    RenewableOnly,
    WithoutModeCosts,
    WithoutResourceCosts,
):
    """
    Resource project scheduling problem with stochastic duration and conditional tasks template.
    It consists in :
    - a deterministic scheduling problem with precedence constraint between task
    - a set of renewable resource with constant availability (capacity)
    - task having a deterministic resource consumption and a stochastic duration given as a distribution
    - based on duration of tasks, some optional tasks have to be executed.
    The goal is to minimize the overall expected makespan
    """

    pass


class SingleModeRCPSP_Simulated_Stochastic_Durations_WithConditionalTasks(
    SchedulingDomain,
    SingleMode,
    # SimulatedTaskDuration,
    DeterministicTaskProgress,
    WithoutResourceUnit,
    WithoutPreemptivity,
    WithoutPreallocations,
    WithoutTimeLag,
    WithoutTimeWindow,
    WithoutResourceSkills,
    WithoutResourceAvailabilityChange,
    # WithConditionalTasks,
    ConstantResourceConsumption,
    RenewableOnly,
    WithoutModeCosts,
    WithoutResourceCosts,
):
    """
    Resource project scheduling problem with stochastic duration and conditional tasks template.
    It consists in :
    - a deterministic scheduling problem with precedence constraint between task
    - a set of renewable resource with constant availability (capacity)
    - task having a deterministic resource consumption and a stochastic duration that is simulated as blackbox
    - based on duration of tasks, some optional tasks have to be executed.
    The goal is to minimize the overall expected makespan
    """

    pass
