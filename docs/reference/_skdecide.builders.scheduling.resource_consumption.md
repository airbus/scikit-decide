# builders.scheduling.resource_consumption

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## VariableResourceConsumption

A domain must inherit this class if the amount of resource needed by some tasks vary in time.

### get\_variable\_resource\_consumption <Badge text="VariableResourceConsumption" type="tip"/>

<skdecide-signature name= "get_variable_resource_consumption" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return true if the domain has variable resource consumption,
false if the consumption of resource does not vary in time for any of the tasks

### \_get\_variable\_resource\_consumption <Badge text="VariableResourceConsumption" type="tip"/>

<skdecide-signature name= "_get_variable_resource_consumption" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return true if the domain has variable resource consumption,
false if the consumption of resource does not vary in time for any of the tasks

## ConstantResourceConsumption

A domain must inherit this class if the amount of resource needed by all tasks do not vary in time.

### get\_variable\_resource\_consumption <Badge text="VariableResourceConsumption" type="warn"/>

<skdecide-signature name= "get_variable_resource_consumption" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return true if the domain has variable resource consumption,
false if the consumption of resource does not vary in time for any of the tasks

### \_get\_variable\_resource\_consumption <Badge text="VariableResourceConsumption" type="warn"/>

<skdecide-signature name= "_get_variable_resource_consumption" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Return true if the domain has variable resource consumption,
false if the consumption of resource does not vary in time for any of the tasks

