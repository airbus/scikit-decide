# builders.solver.restorability

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## Restorable

A solver must inherit this class if its state can be saved and reloaded (to continue computation later on or
reuse its solution).

### load <Badge text="Restorable" type="tip"/>

<skdecide-signature name= "load" :sig="{'params': [{'name': 'self'}, {'name': 'path', 'annotation': 'str'}, {'name': 'domain_factory', 'annotation': 'Callable[[], D]'}], 'return': 'None'}"></skdecide-signature>

Restore the solver state from given path.

#### Parameters
- **path**: The path where the solver state was saved.
- **domain_factory**: A callable with no argument returning the domain to solve (useful in some implementations).

### save <Badge text="Restorable" type="tip"/>

<skdecide-signature name= "save" :sig="{'params': [{'name': 'self'}, {'name': 'path', 'annotation': 'str'}], 'return': 'None'}"></skdecide-signature>

Save the solver state to given path.

#### Parameters
- **path**: The path to store the saved state.

### \_load <Badge text="Restorable" type="tip"/>

<skdecide-signature name= "_load" :sig="{'params': [{'name': 'self'}, {'name': 'path', 'annotation': 'str'}, {'name': 'domain_factory', 'annotation': 'Callable[[], D]'}], 'return': 'None'}"></skdecide-signature>

Restore the solver state from given path.

#### Parameters
- **path**: The path where the solver state was saved.
- **domain_factory**: A callable with no argument returning the domain to solve (useful in some implementations).

### \_save <Badge text="Restorable" type="tip"/>

<skdecide-signature name= "_save" :sig="{'params': [{'name': 'self'}, {'name': 'path', 'annotation': 'str'}], 'return': 'None'}"></skdecide-signature>

Save the solver state to given path.

#### Parameters
- **path**: The path to store the saved state.

