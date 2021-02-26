# builders.scheduling.precedence

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## WithPrecedence

A domain must inherit this class if there exist some predecence constraints between tasks.

### get\_predecessors <Badge text="WithPrecedence" type="tip"/>

<skdecide-signature name= "get_predecessors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the predecessors of the task. Successors are given as a list for a task given as a key.

### get\_successors <Badge text="WithPrecedence" type="tip"/>

<skdecide-signature name= "get_successors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the successors of the tasks. Successors are given as a list for a task given as a key.

### \_get\_predecessors <Badge text="WithPrecedence" type="tip"/>

<skdecide-signature name= "_get_predecessors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the predecessors of the task. Successors are given as a list for a task given as a key.

### \_get\_successors <Badge text="WithPrecedence" type="tip"/>

<skdecide-signature name= "_get_successors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the successors of the tasks. Successors are given as a list for a task given as a key.

## WithoutPrecedence

A domain must inherit this class if there are no predecence constraints between tasks.

### get\_predecessors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "get_predecessors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the predecessors of the task. Successors are given as a list for a task given as a key.

### get\_successors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "get_successors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the successors of the tasks. Successors are given as a list for a task given as a key.

### \_get\_predecessors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "_get_predecessors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the successors of the tasks. Successors are given as a list for a task given as a key.

### \_get\_successors <Badge text="WithPrecedence" type="warn"/>

<skdecide-signature name= "_get_successors" :sig="{'params': [{'name': 'self'}], 'return': 'Dict[int, List[int]]'}"></skdecide-signature>

Return the successors of the tasks. Successors are given as a list for a task given as a key.

