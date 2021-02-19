# builders.scheduling.preallocations

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## WithPreallocations

A domain must inherit this class if there are some pre-allocations to consider.

### get\_preallocations <Badge text="WithPreallocations" type="tip"/>

<skdecide-signature name= "get_preallocations" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[str]]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value indicates the pre-allocated resources for this task (as a list of str)

### \_get\_preallocations <Badge text="WithPreallocations" type="tip"/>

<skdecide-signature name= "_get_preallocations" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[str]]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value indicates the pre-allocated resources for this task (as a list of str)

## WithoutPreallocations

A domain must inherit this class if there are no pre-allocations to consider.

### get\_preallocations <Badge text="WithPreallocations" type="warn"/>

<skdecide-signature name= "get_preallocations" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[str]]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value indicates the pre-allocated resources for this task (as a list of str)

### \_get\_preallocations <Badge text="WithPreallocations" type="warn"/>

<skdecide-signature name= "_get_preallocations" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[str]]'}"></skdecide-signature>

Return a dictionary where the key is the id of a task (int)
and the value indicates the pre-allocated resources for this task (as a list of str)

