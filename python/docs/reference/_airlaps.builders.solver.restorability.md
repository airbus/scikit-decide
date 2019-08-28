# builders.solver.restorability

[[toc]]

## Restorable

A solver must inherit this class if its state can be saved and reloaded (to continue computation later on or
reuse its solution).

### load <Badge text="Restorable" type="tip"/>

<airlaps-signature name= "load" :sig="{'params': [{'name': 'self'}, {'name': 'path', 'annotation': 'str'}], 'return': 'None'}"></airlaps-signature>

Restore the solver state from given path.

#### Parameters
- **path**: The path where the solver state was saved.

### save <Badge text="Restorable" type="tip"/>

<airlaps-signature name= "save" :sig="{'params': [{'name': 'self'}, {'name': 'path', 'annotation': 'str'}], 'return': 'None'}"></airlaps-signature>

Save the solver state to given path.

#### Parameters
- **path**: The path to store the saved state.

### \_load <Badge text="Restorable" type="tip"/>

<airlaps-signature name= "_load" :sig="{'params': [{'name': 'self'}, {'name': 'path', 'annotation': 'str'}], 'return': 'None'}"></airlaps-signature>

Restore the solver state from given path.

#### Parameters
- **path**: The path where the solver state was saved.

### \_save <Badge text="Restorable" type="tip"/>

<airlaps-signature name= "_save" :sig="{'params': [{'name': 'self'}, {'name': 'path', 'annotation': 'str'}], 'return': 'None'}"></airlaps-signature>

Save the solver state to given path.

#### Parameters
- **path**: The path to store the saved state.

