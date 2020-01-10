---
navbar: false
sidebar: false
---

<skdecide-spec>

<template v-slot:Domain>

This is the highest level domain class (inheriting top-level class for each mandatory domain characteristic).

This helper class can be used as the main base class for domains.

Typical use:
```python
class D(Domain, ...)
```

with "..." replaced when needed by a number of classes from following domain characteristics (the ones in
parentheses are optional):

- **agent**: MultiAgent -> SingleAgent
- **concurrency**: Parallel -> Sequential
- **(constraints)**: Constrained
- **dynamics**: Environment -> Simulation -> UncertainTransitions -> EnumerableTransitions
  -> DeterministicTransitions
- **events**: Events -> Actions -> UnrestrictedActions
- **(goals)**: Goals
- **(initialization)**: Initializable -> UncertainInitialized -> DeterministicInitialized
- **memory**: History -> FiniteHistory -> Markovian -> Memoryless
- **observability**: PartiallyObservable -> TransformedObservable -> FullyObservable
- **(renderability)**: Renderable
- **value**: Rewards -> PositiveCosts

</template>

<template v-slot:RLDomain>

This is a typical Reinforcement Learning domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- Environment
- Actions
- Initializable
- Markovian
- TransformedObservable
- Rewards

Typical use:
```python
class D(RLDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

</template>

<template v-slot:MultiAgentRLDomain>

This is a typical multi-agent Reinforcement Learning domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- MultiAgent
- Sequential
- Environment
- Actions
- Initializable
- Markovian
- TransformedObservable
- Rewards

Typical use:
```python
class D(RLDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

</template>

<template v-slot:StatelessSimulatorDomain>

This is a typical stateless simulator domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- Simulation
- Actions
- Markovian
- TransformedObservable
- Rewards

Typical use:
```python
class D(StatelessSimulatorDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

</template>

<template v-slot:MDPDomain>

This is a typical Markov Decision Process domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- EnumerableTransitions
- Actions
- DeterministicInitialized
- Markovian
- FullyObservable
- Rewards

Typical use:
```python
class D(MDPDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

</template>

<template v-slot:POMDPDomain>

This is a typical Partially Observable Markov Decision Process domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- EnumerableTransitions
- Actions
- UncertainInitialized
- Markovian
- PartiallyObservable
- Rewards

Typical use:
```python
class D(POMDPDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

</template>

<template v-slot:GoalMDPDomain>

This is a typical Goal Markov Decision Process domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- EnumerableTransitions
- Actions
- Goals
- DeterministicInitialized
- Markovian
- FullyObservable
- PositiveCosts

Typical use:
```python
class D(GoalMDPDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

</template>

<template v-slot:GoalPOMDPDomain>

This is a typical Goal Partially Observable Markov Decision Process domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- EnumerableTransitions
- Actions
- Goals
- UncertainInitialized
- Markovian
- PartiallyObservable
- PositiveCosts

Typical use:
```python
class D(GoalPOMDPDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

</template>

<template v-slot:DeterministicPlanningDomain>

This is a typical deterministic planning domain class.

This helper class can be used as an alternate base class for domains, inheriting the following:

- Domain
- SingleAgent
- Sequential
- DeterministicTransitions
- Actions
- Goals
- DeterministicInitialized
- Markovian
- FullyObservable
- PositiveCosts

Typical use:
```python
class D(DeterministicPlanningDomain)
```

::: tip
It is also possible to refine any alternate base class, like for instance:
```python
class D(RLDomain, FullyObservable)
```
:::

</template>

<template v-slot:MultiAgent>

A domain must inherit this class if it is multi-agent (i.e hosting multiple independent agents).

Agents are identified by (string) agent names.

</template>

<template v-slot:SingleAgent>

A domain must inherit this class if it is single-agent (i.e hosting only one agent).

</template>

<template v-slot:Parallel>

A domain must inherit this class if multiple events/actions can happen in parallel.

</template>

<template v-slot:Sequential>

A domain must inherit this class if its events/actions are sequential (non-parallel).

</template>

<template v-slot:Constrained>

A domain must inherit this class if it has constraints.

</template>

<template v-slot:Environment>

A domain must inherit this class if agents interact with it like a black-box environment.

Black-box environment examples include: the real world, compiled ATARI games, etc.

::: tip
Environment domains are typically stateful: they must keep the current state or history in their memory to
compute next steps (automatically done by default in the `_memory` attribute).
:::

</template>

<template v-slot:Simulation>

A domain must inherit this class if agents interact with it like a simulation.

Compared to pure environment domains, simulation ones have the additional ability to sample transitions from any
given state.

::: tip
Simulation domains are typically stateless: they do not need to store the current state or history in memory
since it is usually passed as parameter of their functions. By default, they only become stateful whenever they
are used as environments (e.g. via `Initializable.reset()` and `Environment.step()` functions).
:::

</template>

<template v-slot:UncertainTransitions>

A domain must inherit this class if its dynamics is uncertain and provided as a white-box model.

Compared to pure simulation domains, uncertain transition ones provide in addition the full probability distribution
of next states given a memory and action.

::: tip
Uncertain transition domains are typically stateless: they do not need to store the current state or history in
memory since it is usually passed as parameter of their functions. By default, they only become stateful
whenever they are used as environments (e.g. via `Initializable.reset()` and `Environment.step()` functions).
:::

</template>

<template v-slot:EnumerableTransitions>

A domain must inherit this class if its dynamics is uncertain (with enumerable transitions) and provided as a
white-box model.

Compared to pure uncertain transition domains, enumerable transition ones guarantee that all probability
distributions of next state are discrete.

::: tip
Enumerable transition domains are typically stateless: they do not need to store the current state or history in
memory since it is usually passed as parameter of their functions. By default, they only become stateful
whenever they are used as environments (e.g. via `Initializable.reset()` and `Environment.step()` functions).
:::

</template>

<template v-slot:DeterministicTransitions>

A domain must inherit this class if its dynamics is deterministic and provided as a white-box model.

Compared to pure enumerable transition domains, deterministic transition ones guarantee that there is only one next
state for a given source memory (state or history) and action.

::: tip
Deterministic transition domains are typically stateless: they do not need to store the current state or history
in memory since it is usually passed as parameter of their functions. By default, they only become stateful
whenever they are used as environments (e.g. via `Initializable.reset()` and `Environment.step()` functions).
:::

</template>

<template v-slot:Events>

A domain must inherit this class if it handles events (controllable or not not by the agents).

</template>

<template v-slot:Actions>

A domain must inherit this class if it handles only actions (i.e. controllable events).

</template>

<template v-slot:UnrestrictedActions>

A domain must inherit this class if it handles only actions (i.e. controllable events), which are always all
applicable.

</template>

<template v-slot:Goals>

A domain must inherit this class if it has formalized goals.

</template>

<template v-slot:Initializable>

A domain must inherit this class if it can be initialized.

</template>

<template v-slot:UncertainInitialized>

A domain must inherit this class if its states are initialized according to a probability distribution known as
white-box.

</template>

<template v-slot:DeterministicInitialized>

A domain must inherit this class if it has a deterministic initial state known as white-box.

</template>

<template v-slot:History>

A domain must inherit this class if its full state history must be stored to compute its dynamics (non-Markovian
domain).

</template>

<template v-slot:FiniteHistory>

A domain must inherit this class if the last N states must be stored to compute its dynamics (Markovian
domain of order N).

N is specified by the return value of the `FiniteHistory._get_memory_maxlen()` function.

</template>

<template v-slot:Markovian>

A domain must inherit this class if only its last state must be stored to compute its dynamics (pure Markovian
domain).

</template>

<template v-slot:Memoryless>

A domain must inherit this class if it does not require any previous state(s) to be stored to compute its
dynamics.

A dice roll simulator is an example of memoryless domain (next states are independent of previous ones).

::: tip
Whenever an existing domain (environment, simulator...) needs to be wrapped instead of implemented fully in
scikit-decide (e.g. compiled ATARI games), Memoryless can be used because the domain memory (if any) would
be handled externally.
:::

</template>

<template v-slot:PartiallyObservable>

A domain must inherit this class if it is partially observable.

"Partially observable" means that the observation provided to the agent is computed from (but generally not equal
to) the internal state of the domain. Additionally, according to literature, a partially observable domain must
provide the probability distribution of the observation given a state and action.

</template>

<template v-slot:TransformedObservable>

A domain must inherit this class if it is transformed observable.

"Transformed observable" means that the observation provided to the agent is deterministically computed from (but
generally not equal to) the internal state of the domain.

</template>

<template v-slot:FullyObservable>

A domain must inherit this class if it is fully observable.

"Fully observable" means that the observation provided to the agent is equal to the internal state of the domain.

::: warning
In the case of fully observable domains, make sure that the observation type D.T_observation is equal to the
state type D.T_state.
:::

</template>

<template v-slot:Renderable>

A domain must inherit this class if it can be rendered with any kind of visualization.

</template>

<template v-slot:Rewards>

A domain must inherit this class if it sends rewards (positive and/or negative).

</template>

<template v-slot:PositiveCosts>

A domain must inherit this class if it sends only positive costs (i.e. negative rewards).

Having only positive costs is a required assumption for certain solvers to work, such as classical planners.

</template>

</skdecide-spec>

