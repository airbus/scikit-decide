# builders.scheduling.preemptivity

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## ResumeType

An enumeration.

## WithPreemptivity

A domain must inherit this class if there exist at least 1 task that can be paused.

### get\_task\_paused\_non\_renewable\_resource\_returned <Badge text="WithPreemptivity" type="tip"/>

<skdecide-signature name= "get_task_paused_non_renewable_resource_returned" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type bool indicating
if the non-renewable resources are consumed when the task is paused (False) or made available again (True).
E.g. {
        2: False  # if paused, non-renewable resource will be consumed
        5: True  # if paused, the non-renewable resource will be available again
        }

### get\_task\_preemptivity <Badge text="WithPreemptivity" type="tip"/>

<skdecide-signature name= "get_task_preemptivity" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a boolean indicating
if the task can be paused or stopped.
E.g. {
        1: False
        2: True
        3: False
        4: False
        5: True
        6: False
        }

### get\_task\_resuming\_type <Badge text="WithPreemptivity" type="tip"/>

<skdecide-signature name= "get_task_resuming_type" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, ResumeType]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type ResumeType indicating
if the task can be resumed (restarted from where it was paused with no time loss)
or restarted (restarted from the start).
E.g. {
        1: ResumeType.NA
        2: ResumeType.Resume
        3: ResumeType.NA
        4: ResumeType.NA
        5: ResumeType.Restart
        6: ResumeType.NA
        }

### \_get\_task\_paused\_non\_renewable\_resource\_returned <Badge text="WithPreemptivity" type="tip"/>

<skdecide-signature name= "_get_task_paused_non_renewable_resource_returned" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type bool indicating
if the non-renewable resources are consumed when the task is paused (False) or made available again (True).
E.g. {
        2: False  # if paused, non-renewable resource will be consumed
        5: True  # if paused, the non-renewable resource will be available again
        }

### \_get\_task\_preemptivity <Badge text="WithPreemptivity" type="tip"/>

<skdecide-signature name= "_get_task_preemptivity" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a boolean indicating
if the task can be paused or stopped.
E.g. {
        1: False
        2: True
        3: False
        4: False
        5: True
        6: False
        }

### \_get\_task\_resuming\_type <Badge text="WithPreemptivity" type="tip"/>

<skdecide-signature name= "_get_task_resuming_type" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, ResumeType]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type ResumeType indicating
if the task can be resumed (restarted from where it was paused with no time loss)
or restarted (restarted from the start).
E.g. {
        1: ResumeType.NA
        2: ResumeType.Resume
        3: ResumeType.NA
        4: ResumeType.NA
        5: ResumeType.Restart
        6: ResumeType.NA
        }

## WithoutPreemptivity

A domain must inherit this class if none of the task can be paused.

### get\_task\_paused\_non\_renewable\_resource\_returned <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "get_task_paused_non_renewable_resource_returned" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type bool indicating
if the non-renewable resources are consumed when the task is paused (False) or made available again (True).
E.g. {
        2: False  # if paused, non-renewable resource will be consumed
        5: True  # if paused, the non-renewable resource will be available again
        }

### get\_task\_preemptivity <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "get_task_preemptivity" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a boolean indicating
if the task can be paused or stopped.
E.g. {
        1: False
        2: True
        3: False
        4: False
        5: True
        6: False
        }

### get\_task\_resuming\_type <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "get_task_resuming_type" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, ResumeType]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type ResumeType indicating
if the task can be resumed (restarted from where it was paused with no time loss)
or restarted (restarted from the start).
E.g. {
        1: ResumeType.NA
        2: ResumeType.Resume
        3: ResumeType.NA
        4: ResumeType.NA
        5: ResumeType.Restart
        6: ResumeType.NA
        }

### \_get\_task\_paused\_non\_renewable\_resource\_returned <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "_get_task_paused_non_renewable_resource_returned" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type bool indicating
if the non-renewable resources are consumed when the task is paused (False) or made available again (True).
E.g. {
        2: False  # if paused, non-renewable resource will be consumed
        5: True  # if paused, the non-renewable resource will be available again
        }

### \_get\_task\_preemptivity <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "_get_task_preemptivity" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, bool]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a boolean indicating
if the task can be paused or stopped.
E.g. {
        1: False
        2: True
        3: False
        4: False
        5: True
        6: False
        }

### \_get\_task\_resuming\_type <Badge text="WithPreemptivity" type="warn"/>

<skdecide-signature name= "_get_task_resuming_type" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, ResumeType]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is of type ResumeType indicating
if the task can be resumed (restarted from where it was paused with no time loss)
or restarted (restarted from the start).
E.g. {
        1: ResumeType.NA
        2: ResumeType.Resume
        3: ResumeType.NA
        4: ResumeType.NA
        5: ResumeType.Restart
        6: ResumeType.NA
        }

