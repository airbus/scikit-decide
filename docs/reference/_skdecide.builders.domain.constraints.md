# builders.domain.constraints

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## Constrained

A domain must inherit this class if it has constraints.

### get\_constraints <Badge text="Constrained" type="tip"/>

<skdecide-signature name= "get_constraints" :sig="{'params': [{'name': 'self'}], 'return': 'List[Constraint[D.T_memory[D.T_state], D.T_agent[D.T_concurrency[D.T_event]], D.T_state]]'}"></skdecide-signature>

Get the (cached) domain constraints.

By default, `Constrained.get_constraints()` internally calls `Constrained._get_constraints_()` the first time and
automatically caches its value to make future calls more efficient (since the list of constraints is assumed to
be constant).

#### Returns
The list of constraints.

### \_get\_constraints <Badge text="Constrained" type="tip"/>

<skdecide-signature name= "_get_constraints" :sig="{'params': [{'name': 'self'}], 'return': 'List[Constraint[D.T_memory[D.T_state], D.T_agent[D.T_concurrency[D.T_event]], D.T_state]]'}"></skdecide-signature>

Get the (cached) domain constraints.

By default, `Constrained._get_constraints()` internally calls `Constrained._get_constraints_()` the first time and
automatically caches its value to make future calls more efficient (since the list of constraints is assumed to
be constant).

#### Returns
The list of constraints.

### \_get\_constraints\_ <Badge text="Constrained" type="tip"/>

<skdecide-signature name= "_get_constraints_" :sig="{'params': [{'name': 'self'}], 'return': 'List[Constraint[D.T_memory[D.T_state], D.T_agent[D.T_concurrency[D.T_event]], D.T_state]]'}"></skdecide-signature>

Get the domain constraints.

This is a helper function called by default from `Constrained.get_constraints()`, the difference being that the
result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The list of constraints.

