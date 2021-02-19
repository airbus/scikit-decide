# builders.scheduling.scheduling_domains_modelling

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## SchedulingActionEnum

Enum defining the different types of scheduling actions:
- START: start a task
- PAUSE: pause a task
- RESUME: resume a task
- TIME_PR: do not apply actions on tasks and progress in time

## State

Class modelling a scheduling state and used by sk-decide scheduling domains.

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

### Constructor <Badge text="State" type="tip"/>

<skdecide-signature name= "State" :sig="{'params': [{'name': 'task_ids', 'annotation': 'List[int]'}, {'name': 'tasks_available', 'default': 'None', 'annotation': 'Set[int]'}]}"></skdecide-signature>

Initialize a scheduling state.

#### Parameters
- **task_ids**: a list of all task ids in the scheduling domain.
- **tasks_available**: a set of task ids that are available for scheduling. This may differ from task_ids if the
 domain contains conditional tasks.

## SchedulingAction

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

### Constructor <Badge text="SchedulingAction" type="tip"/>

<skdecide-signature name= "SchedulingAction" :sig="{'params': [{'name': 'task', 'annotation': 'Union[int, None]'}, {'name': 'action', 'annotation': 'SchedulingActionEnum'}, {'name': 'mode', 'annotation': 'Union[int, None]'}, {'name': 'time_progress', 'annotation': 'bool'}, {'name': 'resource_unit_names', 'default': 'None', 'annotation': 'Optional[Set[str]]'}]}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

