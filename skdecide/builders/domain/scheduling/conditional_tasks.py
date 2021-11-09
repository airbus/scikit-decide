# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, List, Set, Tuple

from skdecide.core import Distribution

# from skdecide.builders.scheduling.scheduling_domains import State

__all__ = ["WithConditionalTasks", "WithoutConditionalTasks"]


class WithConditionalTasks:
    """A domain must inherit this class if some tasks only need be executed under some conditions
    and that the condition model can be expressed with Distribution objects."""

    # def __init__(self):
    #     self._current_conditions = set()

    # def _get_current_conditions(self) -> Set[int]:
    #     return self._current_conditions

    # def _reset_current_conditions(self):
    #     self._current_conditions = set()

    def get_all_condition_items(self) -> Enum:
        """Return an Enum with all the elements that can be used to define a condition.

        Example:
            return
                ConditionElementsExample(Enum):
                    OK = 0
                    NC_PART_1_OPERATION_1 = 1
                    NC_PART_1_OPERATION_2 = 2
                    NC_PART_2_OPERATION_1 = 3
                    NC_PART_2_OPERATION_2 = 4
                    HARDWARE_ISSUE_MACHINE_A = 5
                    HARDWARE_ISSUE_MACHINE_B = 6
        """
        return self._get_all_condition_items()

    def _get_all_condition_items(self) -> Enum:
        raise NotImplementedError

    def get_task_on_completion_added_conditions(self) -> Dict[int, List[Distribution]]:
        """Return a dict of list. The key of the dict is the task id and each list is composed of a list of tuples.
        Each tuple contains the probability (first item in tuple) that the conditionElement (second item in tuple)
        is True. The probabilities in the inner list should sum up to 1. The dictionary should only contains the keys
        of tasks that can create conditions.

        Example:
             return
                {
                    12:
                        [
                        DiscreteDistribution([(ConditionElementsExample.NC_PART_1_OPERATION_1, 0.1), (ConditionElementsExample.OK, 0.9)]),
                        DiscreteDistribution([(ConditionElementsExample.HARDWARE_ISSUE_MACHINE_A, 0.05), ('paper', 0.1), (ConditionElementsExample.OK, 0.95)])
                        ]
                }
        """
        return self._get_task_on_completion_added_conditions()

    def _get_task_on_completion_added_conditions(self) -> Dict[int, List[Distribution]]:
        raise NotImplementedError

    def _sample_completion_conditions(self, task: int) -> List[int]:
        """Samples the condition distributions associated with the given task and return a list of sampled
        conditions."""
        conditions_to_add = []
        tests = self.get_task_on_completion_added_conditions()[task]
        for test in tests:
            conditions_to_add.append(test.sample())
        return conditions_to_add

    def sample_completion_conditions(self, task: int) -> List[int]:
        """Samples the condition distributions associated with the given task and return a list of sampled
        conditions."""
        return self._sample_completion_conditions(task=task)

    def _get_task_existence_conditions(self) -> Dict[int, List[int]]:
        """Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)
        for the task to be part of the schedule. If a task has no entry in the dictionary,
        there is no conditions for that task.

        Example:
            return
                 {
                    20: [get_all_condition_items().NC_PART_1_OPERATION_1],
                    21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]
                    22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]
                 }e

        """
        raise NotImplementedError

    def get_task_existence_conditions(self) -> Dict[int, List[int]]:
        """Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)
        for the task to be part of the schedule. If a task has no entry in the dictionary,
        there is no conditions for that task.

        Example:
            return
                 {
                    20: [get_all_condition_items().NC_PART_1_OPERATION_1],
                    21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]
                    22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]
                 }e

        """
        return self._get_task_existence_conditions()

    def _add_to_current_conditions(self, task: int, state):
        """Samples completion conditions for a given task and add these conditions to the list of conditions in the
        given state. This function should be called when a task complete."""
        conditions_to_add = self.sample_completion_conditions(task)
        for x in conditions_to_add:
            state._current_conditions.add(x)

    def add_to_current_conditions(self, task: int, state):
        """Samples completion conditions for a given task and add these conditions to the list of conditions in the
        given state. This function should be called when a task complete."""
        return self._add_to_current_conditions(task=task, state=state)

    def _get_available_tasks(self, state) -> Set[int]:
        """Returns the set of all task ids that can be considered under the conditions defined in the given state.
        Note that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks
        that are remaining, or that have been completed, paused or started / resumed."""
        all_ids = self.get_tasks_ids()
        available_ids = set()
        # [for id in all_ids if id not in self.get_task_existence_conditions().keys() and ]
        for id in all_ids:
            if id not in self.get_task_existence_conditions().keys():
                available_ids.add(id)
            else:
                test = True
                for cond in self.get_task_existence_conditions()[id]:
                    if cond not in state._current_conditions:
                        test = False
                if test:
                    available_ids.add(id)
        return available_ids

    def get_available_tasks(self, state) -> Set[int]:
        """Returns the set of all task ids that can be considered under the conditions defined in the given state.
        Note that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks
        that are remaining, or that have been completed, paused or started / resumed."""
        return self._get_available_tasks(state=state)

    def _get_all_unconditional_tasks(self) -> Set[int]:
        """Returns the set of all task ids for which there are no conditions. These tasks are to be considered at
        the start of a project (i.e. in the initial state)."""
        all_ids = self.get_tasks_ids()
        available_ids = set()
        for id in all_ids:
            if (id not in self.get_task_existence_conditions().keys()) or (
                len(self.get_task_existence_conditions()[id]) == 0
            ):
                available_ids.add(id)
        return available_ids

    def get_all_unconditional_tasks(self) -> Set[int]:
        """Returns the set of all task ids for which there are no conditions. These tasks are to be considered at
        the start of a project (i.e. in the initial state)."""
        return self._get_all_unconditional_tasks()


class WithoutConditionalTasks(WithConditionalTasks):
    """A domain must inherit this class if all tasks need be executed without conditions."""

    def _get_all_condition_items(self) -> Enum:
        return None

    def _get_task_on_completion_added_conditions(
        self,
    ) -> Dict[int, List[List[Tuple[float, int]]]]:
        return {}

    def _get_task_existence_conditions(self) -> Dict[int, List[int]]:
        return {}
