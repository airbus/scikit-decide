# builders.scheduling.task_progress

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## CustomTaskProgress

A domain must inherit this class if the task progress is uncertain.

### get\_task\_progress <Badge text="CustomTaskProgress" type="tip"/>

<skdecide-signature name= "get_task_progress" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 't_from', 'annotation': 'int'}, {'name': 't_to', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'Optional[int]'}, {'name': 'sampled_duration', 'default': 'None', 'annotation': 'Optional[int]'}], 'return': 'float'}"></skdecide-signature>

#### Returns
 The task progress (float) between t_from and t_to.
 

### \_get\_task\_progress <Badge text="CustomTaskProgress" type="tip"/>

<skdecide-signature name= "_get_task_progress" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 't_from', 'annotation': 'int'}, {'name': 't_to', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'Optional[int]'}, {'name': 'sampled_duration', 'default': 'None', 'annotation': 'Optional[int]'}], 'return': 'float'}"></skdecide-signature>

#### Returns
 The task progress (float) between t_from and t_to.
 

## DeterministicTaskProgress

A domain must inherit this class if the task progress is deterministic and can be considered as linear
over the duration of the task.

### get\_task\_progress <Badge text="CustomTaskProgress" type="warn"/>

<skdecide-signature name= "get_task_progress" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 't_from', 'annotation': 'int'}, {'name': 't_to', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'Optional[int]'}, {'name': 'sampled_duration', 'default': 'None', 'annotation': 'Optional[int]'}], 'return': 'float'}"></skdecide-signature>

#### Returns
 The task progress (float) between t_from and t_to.
 

### \_get\_task\_progress <Badge text="CustomTaskProgress" type="warn"/>

<skdecide-signature name= "_get_task_progress" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 't_from', 'annotation': 'int'}, {'name': 't_to', 'annotation': 'int'}, {'name': 'mode', 'annotation': 'Optional[int]'}, {'name': 'sampled_duration', 'default': 'None', 'annotation': 'Optional[int]'}], 'return': 'float'}"></skdecide-signature>

#### Returns
 The task progress (float) between t_from and t_to based on the task duration
and assuming linear progress.

