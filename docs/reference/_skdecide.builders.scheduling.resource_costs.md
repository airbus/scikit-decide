# builders.scheduling.resource_costs

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## WithModeCosts

A domain must inherit this class if there are some mode costs to consider.

### get\_mode\_costs <Badge text="WithModeCosts" type="tip"/>

<skdecide-signature name= "get_mode_costs" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, float]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode
and the value indicates the cost of execution the task in the mode.

### \_get\_mode\_costs <Badge text="WithModeCosts" type="tip"/>

<skdecide-signature name= "_get_mode_costs" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, float]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode
and the value indicates the cost of execution the task in the mode.

## WithoutModeCosts

A domain must inherit this class if there are no mode cost to consider.

### get\_mode\_costs <Badge text="WithModeCosts" type="warn"/>

<skdecide-signature name= "get_mode_costs" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, float]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode
and the value indicates the cost of execution the task in the mode.

### \_get\_mode\_costs <Badge text="WithModeCosts" type="warn"/>

<skdecide-signature name= "_get_mode_costs" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, Dict[int, float]]'}"></skdecide-signature>

Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode
and the value indicates the cost of execution the task in the mode.

## WithResourceCosts

A domain must inherit this class if there are some resource costs to consider.

### get\_resource\_cost\_per\_time\_unit <Badge text="WithResourceCosts" type="tip"/>

<skdecide-signature name= "get_resource_cost_per_time_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, float]'}"></skdecide-signature>

Return a dictionary where the key is the name of a resource (str)
and the value indicates the cost of using this resource per time unit.

### \_get\_resource\_cost\_per\_time\_unit <Badge text="WithResourceCosts" type="tip"/>

<skdecide-signature name= "_get_resource_cost_per_time_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, float]'}"></skdecide-signature>

Return a dictionary where the key is the name of a resource (str)
and the value indicates the cost of using this resource per time unit.

## WithoutResourceCosts

A domain must inherit this class if there are no resource cost to consider.

### get\_resource\_cost\_per\_time\_unit <Badge text="WithResourceCosts" type="warn"/>

<skdecide-signature name= "get_resource_cost_per_time_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, float]'}"></skdecide-signature>

Return a dictionary where the key is the name of a resource (str)
and the value indicates the cost of using this resource per time unit.

### \_get\_resource\_cost\_per\_time\_unit <Badge text="WithResourceCosts" type="warn"/>

<skdecide-signature name= "_get_resource_cost_per_time_unit" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, float]'}"></skdecide-signature>

Return a dictionary where the key is the name of a resource (str)
and the value indicates the cost of using this resource per time unit.

