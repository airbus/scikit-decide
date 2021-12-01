# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Collection
from copy import copy, deepcopy
from enum import Enum
from time import time
from typing import Dict, List, Optional, Set, Union

from skdecide.builders.domain.scheduling.task import Task


class Node:
    __slots__ = ("value", "next_node")

    def __init__(self, value: Task = None, next_node=None):
        self.value = value
        self.next_node = next_node


class TaskLinkedList(Collection):
    def __init__(self, head=None):
        self.head = head

    def push_front(self, value: Task):
        self.head = Node(value, self.head)

    def __iter__(self):
        current = self.head
        while current:
            yield current
            current = current.next_node

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    def __contains__(self, value: Task) -> bool:
        for node in iter(self):
            if node.value == value:
                return True
        return False

    def __copy__(self):
        return TaskLinkedList(self.head)


class Timer:
    def __init__(self):
        self.c = [0] * 15
        self.v = [0] * 15

    def start(self):
        self.__i = 0
        self.__tic = time()

    def register(self, count: int = 0):
        self.v[self.__i] += time() - self.__tic
        self.c[self.__i] += count
        self.__i += 1
        self.__tic = time()

    def __str__(self):
        return f"Times: {self.v.__str__()} total = {sum(self.v)}\nCount: {self.c}"


class SchedulingActionEnum(Enum):
    """
    Enum defining the different types of scheduling actions:
    - START: start a task
    - PAUSE: pause a task
    - RESUME: resume a task
    - TIME_PR: do not apply actions on tasks and progress in time
    """

    START = 0
    PAUSE = 1
    RESUME = 2
    TIME_PR = 3


timer = Timer()


class State:
    """Class modelling a scheduling state and used by sk-decide scheduling domains.

    It contains the following information:
        t: the timestamp.
        task_ids: a list of all task ids in the scheduling domain.
        tasks_remaining: a set containing the ids of tasks still to be started
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
        _current_conditions: set of conditions observed so far, to be used by domains with WithConditionalTask
            properties

    """

    # TODO : code efficient hash/eq functions. will probably be mandatory in some planning algo.
    t: int
    tasks_remaining: Set[int]
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
    tasks_complete_details: TaskLinkedList
    _current_conditions: Set

    # TODO : put the attributes in the __init__ ?!
    def __init__(self, task_ids: List[int], tasks_available: Set[int] = None):
        """Initialize a scheduling state.

        # Parameters
        task_ids: a list of all task ids in the scheduling domain.
        tasks_available: a set of task ids that are available for scheduling. This may differ from task_ids if the
         domain contains conditional tasks.
        """
        self.t = 0
        self.task_ids = task_ids
        # self.tasks_remaining = set()
        self.tasks_remaining = tasks_available
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
        self.tasks_complete_details = TaskLinkedList()
        self._current_conditions = set()

    def copy(self):
        s = State(task_ids=self.task_ids)
        s.t = self.t
        global timer
        timer.start()
        s.tasks_remaining = deepcopy(self.tasks_remaining)
        timer.register(len(s.tasks_remaining))
        s.tasks_ongoing = deepcopy(self.tasks_ongoing)
        timer.register(len(s.tasks_ongoing))
        s.tasks_paused = deepcopy(self.tasks_paused)
        timer.register(len(s.tasks_paused))
        s.tasks_progress = deepcopy(self.tasks_progress)
        timer.register(len(s.tasks_progress))
        s.tasks_mode = deepcopy(self.tasks_mode)
        timer.register(len(s.tasks_mode))
        s.tasks_complete = deepcopy(self.tasks_complete)
        timer.register(len(s.tasks_complete))
        s.resource_to_task = deepcopy(self.resource_to_task)
        timer.register(len(s.resource_to_task))
        s.resource_availability = deepcopy(self.resource_availability)
        timer.register(len(s.resource_availability))
        s.resource_used = deepcopy(self.resource_used)
        timer.register(len(s.resource_used))
        s.resource_used_for_task = deepcopy(self.resource_used_for_task)
        timer.register(len(s.resource_used_for_task))
        s._current_conditions = deepcopy(self._current_conditions)
        timer.register(len(s._current_conditions))
        s.tasks_details = deepcopy(self.tasks_details)
        timer.register(len(s.tasks_details))
        s.tasks_complete_details = copy(self.tasks_complete_details)
        timer.register(len(s.tasks_complete_details))
        return s

    def __str__(self):
        s = "State : " + "\n"
        for key in sorted(self.__dict__.keys()):
            if key == "tasks_details":
                for key2 in sorted(self.tasks_details.keys()):
                    s += str(self.tasks_details[key2]) + "\t"
            elif key == "tasks_complete_details":
                for node in self.tasks_complete_details:
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
