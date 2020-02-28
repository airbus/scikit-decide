---
navbar: false
sidebar: false
---

<skdecide-spec isSolver>

<template v-slot:Solver>

This is the highest level solver class (inheriting top-level class for each mandatory solver characteristic).

This helper class can be used as the main base class for solvers.

Typical use:
```python
class MySolver(Solver, ...)
```

with "..." replaced when needed by a number of classes from following domain characteristics (the ones in
parentheses are optional):

- **(assessability)**: Utilities -> QValues
- **(policy)**: Policies -> UncertainPolicies -> DeterministicPolicies
- **(restorability)**: Restorable

</template>

<template v-slot:DeterministicPolicySolver>

This is a typical deterministic policy solver class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Solver
- DeterministicPolicies

Typical use:
```python
class MySolver(DeterministicPolicySolver)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class MySolver(DeterministicPolicySolver, QValues)
```
:::

</template>

<template v-slot:Utilities>

A solver must inherit this class if it can provide the utility function (i.e. value function).

</template>

<template v-slot:QValues>

A solver must inherit this class if it can provide the Q function (i.e. action-value function).

</template>

<template v-slot:Policies>

A solver must inherit this class if it computes a stochastic policy as part of the solving process.

</template>

<template v-slot:UncertainPolicies>

A solver must inherit this class if it computes a stochastic policy (providing next action distribution
explicitly) as part of the solving process.

</template>

<template v-slot:DeterministicPolicies>

A solver must inherit this class if it computes a deterministic policy as part of the solving process.

</template>

<template v-slot:Restorable>

A solver must inherit this class if its state can be saved and reloaded (to continue computation later on or
reuse its solution).

</template>

</skdecide-spec>

