# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Collection
from copy import copy, deepcopy
from enum import Enum
from typing import Dict, Iterable, Optional, Set, Tuple, Union

from skdecide.builders.domain.scheduling.task import Task


def rebuild_tasks_complete_details_dict(state: State) -> Dict[int, Task]:
    tasks_complete_details = {p.value.id: p.value for p in state.tasks_complete_details}
    return tasks_complete_details


def rebuild_all_tasks_dict(state: State) -> Dict[int, Task]:
    tasks_details = {
        task_id: Task(id=task_id, start=None, sampled_duration=None)
        for task_id in state.task_ids
    }
    tasks_details.update({p.value.id: p.value for p in state.tasks_complete_details})
    tasks_details.update(state.tasks_details)
    return tasks_details


def rebuild_tasks_modes_dict(state: State) -> Dict[int, int]:
    tasks_modes = {p.value[0]: p.value[1] for p in state.tasks_complete_mode}
    return tasks_modes


def rebuild_schedule_dict(state: State) -> Dict[int, Dict[str, int]]:
    schedule = {
        p.value.id: {"start_time": p.value.start, "end_time": p.value.end}
        for p in state.tasks_complete_details
    }
    return schedule


class Node:
    __slots__ = ("value", "next_node")

    def __init__(self, value: Task = None, next_node=None):
        self.value = value
        self.next_node = next_node


class SinglyLinkedList(Collection):
    def __init__(self, head=None):
        self.head = head

    def push_front(self, value: Union[int, float, Task, Tuple[int, int]]):
        self.head = Node(value, self.head)

    def __iter__(self):
        current = self.head
        while current:
            yield current
            current = current.next_node

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    def __contains__(self, value: Union[int, float, Task]) -> bool:
        for node in iter(self):
            if node.value == value:
                return True
        return False

    def __copy__(self):
        return SinglyLinkedList(self.head)


class SchedulingActionEnum(Enum):
    """Enum defining the different types of scheduling actions"""

    START = 0
    PAUSE = 1
    RESUME = 2
    TIME_PR = 3


SchedulingActionEnum.START.__doc__ = "start a task"
SchedulingActionEnum.PAUSE.__doc__ = "pause a task"
SchedulingActionEnum.RESUME.__doc__ = "resume a task"
SchedulingActionEnum.TIME_PR.__doc__ = (
    "do not apply actions on tasks and progress in time"
)


class State:
    """Class modelling a scheduling state and used by sk-decide scheduling domains.

    It contains the following information:
        t: the timestamp.
        task_ids: a list of all task ids in the scheduling domain.
        tasks_unsatisfiable: a set containing the ids of tasks for which canditions are not fulfilled
        tasks_ongoing: a set containing the ids of tasks started and not paused and still to be completed
        tasks_complete: a set containing the ids of tasks that have been completed
        tasks_paused: a set containing the ids of tasks that have been started and paused but not resumed yet
        tasks_progress: a dictionary where the key is a task id (int) and
            the value the progress of the task between 0 and 1 (float)
        tasks_mode: a dictionary where the key is a task id (int) and
            the value the mode used to execute the task (int)
        resource_to_task: dictionary where the key is the name of a resource (str) and the value a task
            it is currently assigned to (int)
        resource_availability: dictionary where the key is the name of a resource (str) and the value the number of
            resource units available for this type of resource regardless of the task assignments (int). Where the
            resource name is a resource unit itself, the availability value takes a value of either 1 (available)
            or 0 (unavailable)
        resource_used: dictionary where the key is the name of a resource (str) and the value the number of
            resource units for this resource type used/assigned on tasks at this time (int). Where the resource
            name is a resource unit itself, the value takes a value of either 1 (used) or 0 (not used)
        resource_used_for_task: nested dictionary where the first key is a task id (int), the second key is the name of
            a resource (str) and the value is the number of resource units for this resource type used/assigned on tasks
            at this time (int). Where the resource name is a resource unit itself, the value takes a value of either 1
            (used) or 0 (not used).
        tasks_details: dictionary where the key is the id of a task (int) and the value a Task object. This Task object
            contains information about the task execution and can be used to post-process the run. It is only used
            by the domain to store execution information and not used by scheduling solvers.
        tasks_full_details: like taks_details but containing all taks, even the ones not completed.
        _current_conditions: set of conditions observed so far, to be used by domains with WithConditionalTask
            properties

    """

    # TODO : code efficient hash/eq functions. will probably be mandatory in some planning algo.
    t: int
    tasks_unsatisfiable: Set[int]
    tasks_ongoing: Set[int]
    tasks_complete: Set[int]
    tasks_paused: Set[int]
    tasks_progress: Dict[int, float]
    tasks_mode: Dict[int, int]
    resource_to_task: Dict[str, int]
    resource_availability: Dict[str, int]
    resource_used: Dict[str, int]
    resource_used_for_task: Dict[int, Dict[str, int]]
    tasks_details: Dict[
        int, Task
    ]  # Use to store task stats, resource used etc... for post-processing purposes
    tasks_complete_details: SinglyLinkedList
    tasks_complete_progress: SinglyLinkedList
    tasks_complete_mode: SinglyLinkedList
    _current_conditions: Set

    # TODO : put the attributes in the __init__ ?!
    def __init__(self, task_ids: Iterable[int], tasks_available: Set[int] = None):
        """Initialize a scheduling state.

        # Parameters
        task_ids: a list of all task ids in the scheduling domain.
        tasks_available: a set of task ids that are available for scheduling. This may differ from task_ids if the
         domain contains conditional tasks.
        """
        self.t = 0
        self.task_ids = task_ids if isinstance(task_ids, set) else set(task_ids)
        self.tasks_unsatisfiable = (
            set()
            if tasks_available is None
            else set(t for t in self.task_ids if t not in tasks_available)
        )
        self.tasks_ongoing = set()
        self.tasks_complete = set()
        self.tasks_paused = set()
        self.tasks_progress = {}
        self.tasks_mode = {}
        self.resource_to_task = {}
        self.resource_availability = {}
        self.resource_used = {}
        self.resource_used_for_task = {}
        self.tasks_details = {}
        self.tasks_complete_details = SinglyLinkedList()
        self.tasks_complete_progress = SinglyLinkedList()
        self.tasks_complete_mode = SinglyLinkedList()
        self._current_conditions = set()

    def copy(self):
        s = State(task_ids=self.task_ids)
        s.t = self.t
        s.tasks_unsatisfiable = deepcopy(self.tasks_unsatisfiable)
        s.tasks_ongoing = deepcopy(self.tasks_ongoing)
        s.tasks_complete = deepcopy(self.tasks_complete)
        s.tasks_paused = deepcopy(self.tasks_paused)
        s.tasks_progress = deepcopy(self.tasks_progress)
        s.tasks_mode = deepcopy(self.tasks_mode)
        s.resource_to_task = deepcopy(self.resource_to_task)
        s.resource_availability = deepcopy(self.resource_availability)
        s.resource_used = deepcopy(self.resource_used)
        s.resource_used_for_task = deepcopy(self.resource_used_for_task)
        s.tasks_details = deepcopy(self.tasks_details)
        s._current_conditions = deepcopy(self._current_conditions)
        s.tasks_complete_details = copy(self.tasks_complete_details)
        s.tasks_complete_progress = copy(self.tasks_complete_progress)
        s.tasks_complete_mode = copy(self.tasks_complete_mode)
        return s

    @property
    def tasks_full_details(self) -> Dict[int, Task]:
        return rebuild_all_tasks_dict(self)

    @property
    def tasks_remaining(self):
        for task in self.task_ids:
            if (
                task not in self.tasks_complete
                and task not in self.tasks_progress
                and task not in self.tasks_unsatisfiable
            ):
                yield task

    def __str__(self):
        s = "State : " + "\n"
        for key in sorted(self.__dict__.keys()):
            if key == "tasks_details":
                for key2 in sorted(self.tasks_details.keys()):
                    s += str(self.tasks_details[key2]) + "\t"
            elif key.startswith("tasks_complete_"):
                for node in getattr(self, key):
                    s += str(node.value) + "\t"
            else:
                s += str(key) + ":" + str(getattr(self, key)) + "\n"
        return s

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class SchedulingAction:
    """
    Can be used to define actions on single task. Resource allocation can only be managed through changes in the mode.
    The time_progress attribute triggers a change in time (i.e. triggers the domain to increment its current time).
    It should thus be used as the last action to be applied at any point in time
    These actions are enumerable due to their coarse grain definition.

    E.g.
        task = 12 (start action 12 in mode 1)
        action = EnumerableActionEnum.START
        mode = 1
        time_progress = False

    E.g. (pause action 13, NB: mode info is not useful here)
        task = 13
        action = EnumerableActionEnum.PAUSE
        mode = None
        time_progress = False

    E.g. (do nothing and progress in time)
        task = None
        action = None
        mode = None
        time_progress = True
    """

    task: int
    action: SchedulingActionEnum
    mode: int
    time_progress: bool

    def __init__(
        self,
        task: Union[int, None],
        action: SchedulingActionEnum,
        mode: Union[int, None],
        time_progress: bool,
        resource_unit_names: Optional[Set[str]] = None,
    ):
        self.task = task
        self.action = action
        self.mode = mode
        self.time_progress = time_progress
        self.resource_unit_names = resource_unit_names

    def __str__(self):
        s = "Action \n"
        s += "Task : " + str(self.task) + "\n"
        s += "Mode : " + str(self.mode) + "\n"
        s += "Action type " + str(self.action.name) + "\n"
        s += "Time progress " + str(self.time_progress) + "\n"
        s += "Resource : " + str(self.resource_unit_names)
        return s

    def __repr__(self):
        return str(self)
