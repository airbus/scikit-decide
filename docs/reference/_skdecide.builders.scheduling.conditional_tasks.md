# builders.scheduling.conditional_tasks

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## WithConditionalTasks

A domain must inherit this class if some tasks only need be executed under some conditions
and that the condition model can be expressed with Distribution objects.

### add\_to\_current\_conditions <Badge text="WithConditionalTasks" type="tip"/>

<skdecide-signature name= "add_to_current_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'state'}]}"></skdecide-signature>

Samples completion conditions for a given task and add these conditions to the list of conditions in the
given state. This function should be called when a task complete.

### get\_all\_condition\_items <Badge text="WithConditionalTasks" type="tip"/>

<skdecide-signature name= "get_all_condition_items" :sig="{'params': [{'name': 'self'}], 'return': 'Enum'}"></skdecide-signature>

Return an Enum with all the elements that can be used to define a condition.

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
    

### get\_all\_unconditional\_tasks <Badge text="WithConditionalTasks" type="tip"/>

<skdecide-signature name= "get_all_unconditional_tasks" :sig="{'params': [{'name': 'self'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids for which there are no conditions. These tasks are to be considered at
the start of a project (i.e. in the initial state). 

### get\_available\_tasks <Badge text="WithConditionalTasks" type="tip"/>

<skdecide-signature name= "get_available_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids that can be considered under the conditions defined in the given state.
Note that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks
that are remaining, or that have been completed, paused or started / resumed.

### get\_task\_existence\_conditions <Badge text="WithConditionalTasks" type="tip"/>

<skdecide-signature name= "get_task_existence_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)
for the task to be part of the schedule. If a task has no entry in the dictionary,
there is no conditions for that task.

Example:
    return
         {
            20: [get_all_condition_items().NC_PART_1_OPERATION_1],
            21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]
            22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]
         }e

 

### get\_task\_on\_completion\_added\_conditions <Badge text="WithConditionalTasks" type="tip"/>

<skdecide-signature name= "get_task_on_completion_added_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[Distribution]]'}"></skdecide-signature>

Return a dict of list. The key of the dict is the task id and each list is composed of a list of tuples.
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
    

### sample\_completion\_conditions <Badge text="WithConditionalTasks" type="tip"/>

<skdecide-signature name= "sample_completion_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}], 'return': 'List[int]'}"></skdecide-signature>

Samples the condition distributions associated with the given task and return a list of sampled
conditions.

### \_add\_to\_current\_conditions <Badge text="WithConditionalTasks" type="tip"/>

<skdecide-signature name= "_add_to_current_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'state'}]}"></skdecide-signature>

Samples completion conditions for a given task and add these conditions to the list of conditions in the
given state. This function should be called when a task complete.

### \_get\_all\_unconditional\_tasks <Badge text="WithConditionalTasks" type="tip"/>

<skdecide-signature name= "_get_all_unconditional_tasks" :sig="{'params': [{'name': 'self'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids for which there are no conditions. These tasks are to be considered at
the start of a project (i.e. in the initial state). 

### \_get\_available\_tasks <Badge text="WithConditionalTasks" type="tip"/>

<skdecide-signature name= "_get_available_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids that can be considered under the conditions defined in the given state.
Note that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks
that are remaining, or that have been completed, paused or started / resumed.

### \_get\_task\_existence\_conditions <Badge text="WithConditionalTasks" type="tip"/>

<skdecide-signature name= "_get_task_existence_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)
for the task to be part of the schedule. If a task has no entry in the dictionary,
there is no conditions for that task.

Example:
    return
         {
            20: [get_all_condition_items().NC_PART_1_OPERATION_1],
            21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]
            22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]
         }e

### \_sample\_completion\_conditions <Badge text="WithConditionalTasks" type="tip"/>

<skdecide-signature name= "_sample_completion_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}], 'return': 'List[int]'}"></skdecide-signature>

Samples the condition distributions associated with the given task and return a list of sampled
conditions.

## WithoutConditionalTasks

A domain must inherit this class if all tasks need be executed without conditions.

### add\_to\_current\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "add_to_current_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'state'}]}"></skdecide-signature>

Samples completion conditions for a given task and add these conditions to the list of conditions in the
given state. This function should be called when a task complete.

### get\_all\_condition\_items <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_all_condition_items" :sig="{'params': [{'name': 'self'}], 'return': 'Enum'}"></skdecide-signature>

Return an Enum with all the elements that can be used to define a condition.

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
    

### get\_all\_unconditional\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_all_unconditional_tasks" :sig="{'params': [{'name': 'self'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids for which there are no conditions. These tasks are to be considered at
the start of a project (i.e. in the initial state). 

### get\_available\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_available_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids that can be considered under the conditions defined in the given state.
Note that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks
that are remaining, or that have been completed, paused or started / resumed.

### get\_task\_existence\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_task_existence_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)
for the task to be part of the schedule. If a task has no entry in the dictionary,
there is no conditions for that task.

Example:
    return
         {
            20: [get_all_condition_items().NC_PART_1_OPERATION_1],
            21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]
            22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]
         }e

 

### get\_task\_on\_completion\_added\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "get_task_on_completion_added_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[Distribution]]'}"></skdecide-signature>

Return a dict of list. The key of the dict is the task id and each list is composed of a list of tuples.
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
    

### sample\_completion\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "sample_completion_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}], 'return': 'List[int]'}"></skdecide-signature>

Samples the condition distributions associated with the given task and return a list of sampled
conditions.

### \_add\_to\_current\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_add_to_current_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}, {'name': 'state'}]}"></skdecide-signature>

Samples completion conditions for a given task and add these conditions to the list of conditions in the
given state. This function should be called when a task complete.

### \_get\_all\_unconditional\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_get_all_unconditional_tasks" :sig="{'params': [{'name': 'self'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids for which there are no conditions. These tasks are to be considered at
the start of a project (i.e. in the initial state). 

### \_get\_available\_tasks <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_get_available_tasks" :sig="{'params': [{'name': 'self'}, {'name': 'state'}], 'return': 'Set[int]'}"></skdecide-signature>

Returns the set of all task ids that can be considered under the conditions defined in the given state.
Note that the set will contains all ids for all tasks in the domain that meet the conditions, that is tasks
that are remaining, or that have been completed, paused or started / resumed.

### \_get\_task\_existence\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_get_task_existence_conditions" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return a dictionary where the key is a task id and the value a list of conditions to be respected (True)
for the task to be part of the schedule. If a task has no entry in the dictionary,
there is no conditions for that task.

Example:
    return
         {
            20: [get_all_condition_items().NC_PART_1_OPERATION_1],
            21: [get_all_condition_items().HARDWARE_ISSUE_MACHINE_A]
            22: [get_all_condition_items().NC_PART_1_OPERATION_1, get_all_condition_items().NC_PART_1_OPERATION_2]
         }e

### \_sample\_completion\_conditions <Badge text="WithConditionalTasks" type="warn"/>

<skdecide-signature name= "_sample_completion_conditions" :sig="{'params': [{'name': 'self'}, {'name': 'task', 'annotation': 'int'}], 'return': 'List[int]'}"></skdecide-signature>

Samples the condition distributions associated with the given task and return a list of sampled
conditions.

