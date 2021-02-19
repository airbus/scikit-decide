# builders.scheduling.modes

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## VaryingModeConsumption

Defines the most generic type of mode.

### Constructor <Badge text="VaryingModeConsumption" type="tip"/>

<skdecide-signature name= "VaryingModeConsumption" :sig="{'params': [{'name': 'mode_dict', 'annotation': 'Dict[str, List[int]]'}]}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

### get\_resource\_need\_at\_time <Badge text="ModeConsumption" type="warn"/>

<skdecide-signature name= "get_resource_need_at_time" :sig="{'params': [{'name': 'self'}, {'name': 'resource_name', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}]}"></skdecide-signature>

Return the resource consumption for the given resource at the given time.
Note that the time should be the time from the start of the execution of the task (starting from 0).

### \_get\_resource\_need\_at\_time <Badge text="ModeConsumption" type="warn"/>

<skdecide-signature name= "_get_resource_need_at_time" :sig="{'params': [{'name': 'self'}, {'name': 'resource_name', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}]}"></skdecide-signature>

Return the resource consumption for the given resource at the given time.
Note that the time should be the time from the start of the execution of the task (starting from 0).

## ConstantModeConsumption

Defines a mode where the resource consumption is constant throughout
the duration of the task.

### Constructor <Badge text="ConstantModeConsumption" type="tip"/>

<skdecide-signature name= "ConstantModeConsumption" :sig="{'params': [{'name': 'mode_dict', 'annotation': 'Dict[str, int]'}]}"></skdecide-signature>

Initialize self.  See help(type(self)) for accurate signature.

### get\_resource\_need <Badge text="ConstantModeConsumption" type="tip"/>

<skdecide-signature name= "get_resource_need" :sig="{'params': [{'name': 'self'}, {'name': 'resource_name', 'annotation': 'str'}]}"></skdecide-signature>

Return the resource consumption for the given resource.

### get\_resource\_need\_at\_time <Badge text="ModeConsumption" type="warn"/>

<skdecide-signature name= "get_resource_need_at_time" :sig="{'params': [{'name': 'self'}, {'name': 'resource_name', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}]}"></skdecide-signature>

Return the resource consumption for the given resource at the given time.
Note that the time should be the time from the start of the execution of the task (starting from 0).

### \_get\_resource\_need <Badge text="ConstantModeConsumption" type="tip"/>

<skdecide-signature name= "_get_resource_need" :sig="{'params': [{'name': 'self'}, {'name': 'resource_name', 'annotation': 'str'}]}"></skdecide-signature>

Return the resource consumption for the given resource.

### \_get\_resource\_need\_at\_time <Badge text="ModeConsumption" type="warn"/>

<skdecide-signature name= "_get_resource_need_at_time" :sig="{'params': [{'name': 'self'}, {'name': 'resource_name', 'annotation': 'str'}, {'name': 'time', 'annotation': 'int'}]}"></skdecide-signature>

Return the resource consumption for the given resource at the given time.
Note that the time should be the time from the start of the execution of the task (starting from 0).

## MultiMode

A domain must inherit this class if tasks can be done in 1 or more modes.

### \_get\_tasks\_ids <Badge text="MultiMode" type="tip"/>

<skdecide-signature name= "_get_tasks_ids" :sig="{'params': [{'name': 'self'}], 'return': 'Union[Set[int], Dict[int, Any], List[int]]'}"></skdecide-signature>

Return a set or dict of int = id of tasks

### \_get\_tasks\_modes <Badge text="MultiMode" type="tip"/>

<skdecide-signature name= "_get_tasks_modes" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, ModeConsumption]]'}"></skdecide-signature>

Return a nested dictionary where the first key is a task id and the second key is a mode id.
 The value is a Mode object defining the resource consumption.
If the domain is an instance of VariableResourceConsumption, VaryingModeConsumption objects should be used.
If this is not the case (i.e. the domain is an instance of ConstantResourceConsumption),
then ConstantModeConsumption should be used.

E.g. with constant resource consumption
    {
        12: {
                1: ConstantModeConsumption({'rt_1': 2, 'rt_2': 0, 'ru_1': 1}),
                2: ConstantModeConsumption({'rt_1': 0, 'rt_2': 3, 'ru_1': 1}),
            }
    }

E.g. with time varying resource consumption
    {
    12: {
        1: VaryingModeConsumption({'rt_1': [2,2,2,2,3], 'rt_2': [0,0,0,0,0], 'ru_1': [1,1,1,1,1]}),
        2: VaryingModeConsumption({'rt_1': [1,1,1,1,2,2,2], 'rt_2': [0,0,0,0,0,0,0], 'ru_1': [1,1,1,1,1,1,1]}),
        }
    }

## SingleMode

A domain must inherit this class if ALL tasks only have 1 possible execution mode.

### \_get\_tasks\_ids <Badge text="MultiMode" type="warn"/>

<skdecide-signature name= "_get_tasks_ids" :sig="{'params': [{'name': 'self'}], 'return': 'Union[Set[int], Dict[int, Any], List[int]]'}"></skdecide-signature>

Return a set or dict of int = id of tasks

### \_get\_tasks\_mode <Badge text="SingleMode" type="tip"/>

<skdecide-signature name= "_get_tasks_mode" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, ModeConsumption]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value is a ModeConsumption object defining
the resource consumption.
If the domain is an instance of VariableResourceConsumption, VaryingModeConsumption objects should be used.
If this is not the case (i.e. the domain is an instance of ConstantResourceConsumption),
then ConstantModeConsumption should be used.

E.g. with constant resource consumption
    {
        12: ConstantModeConsumption({'rt_1': 2, 'rt_2': 0, 'ru_1': 1})
    }

E.g. with time varying resource consumption
    {
        12: VaryingModeConsumption({'rt_1': [2,2,2,2,3], 'rt_2': [0,0,0,0,0], 'ru_1': [1,1,1,1,1]})
    }

### \_get\_tasks\_modes <Badge text="MultiMode" type="warn"/>

<skdecide-signature name= "_get_tasks_modes" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, ModeConsumption]]'}"></skdecide-signature>

Return a nested dictionary where the first key is a task id and the second key is a mode id.
The value is a Mode object defining the resource consumption.

