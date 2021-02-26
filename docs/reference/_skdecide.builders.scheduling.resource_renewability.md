# builders.scheduling.resource_renewability

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## MixedRenewable

A domain must inherit this class if the resource available are non-renewable and renewable.

### all\_tasks\_possible <Badge text="MixedRenewable" type="tip"/>

<skdecide-signature name= "all_tasks_possible" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}], 'return': 'bool'}"></skdecide-signature>

Return a True is for each task there is at least one mode in which the task can be executed, given the
resource configuration in the state provided as argument. Returns False otherwise.
If this function returns False, the scheduling problem is unsolvable from this state.
This is to cope with the use of non-renable resources that may lead to state from which a
task will not be possible anymore.

### get\_resource\_renewability <Badge text="MixedRenewable" type="tip"/>

<skdecide-signature name= "get_resource_renewability" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, bool]'}"></skdecide-signature>

Return a dictionary where the key is a resource name (string)
and the value whether this resource is renewable (True) or not (False).

### \_get\_resource\_renewability <Badge text="MixedRenewable" type="tip"/>

<skdecide-signature name= "_get_resource_renewability" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, bool]'}"></skdecide-signature>

Return a dictionary where the key is a resource name (string)
and the value whether this resource is renewable (True) or not (False).

## RenewableOnly

A domain must inherit this class if the resource available are ALL renewable.

### all\_tasks\_possible <Badge text="MixedRenewable" type="warn"/>

<skdecide-signature name= "all_tasks_possible" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'State'}], 'return': 'bool'}"></skdecide-signature>

Return a True is for each task there is at least one mode in which the task can be executed, given the
resource configuration in the state provided as argument. Returns False otherwise.
If this function returns False, the scheduling problem is unsolvable from this state.
This is to cope with the use of non-renable resources that may lead to state from which a
task will not be possible anymore.

### get\_resource\_renewability <Badge text="MixedRenewable" type="warn"/>

<skdecide-signature name= "get_resource_renewability" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, bool]'}"></skdecide-signature>

Return a dictionary where the key is a resource name (string)
and the value whether this resource is renewable (True) or not (False).

### \_get\_resource\_renewability <Badge text="MixedRenewable" type="warn"/>

<skdecide-signature name= "_get_resource_renewability" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[str, bool]'}"></skdecide-signature>

Return a dictionary where the key is a resource name (string)
and the value whether this resource is renewable (True) or not (False).

