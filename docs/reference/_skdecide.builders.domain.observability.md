# builders.domain.observability

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## PartiallyObservable

A domain must inherit this class if it is partially observable.

"Partially observable" means that the observation provided to the agent is computed from (but generally not equal
to) the internal state of the domain. Additionally, according to literature, a partially observable domain must
provide the probability distribution of the observation given a state and action.

### get\_observation\_distribution <Badge text="PartiallyObservable" type="tip"/>

<skdecide-signature name= "get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_space <Badge text="PartiallyObservable" type="tip"/>

<skdecide-signature name= "get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable.get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### is\_observation <Badge text="PartiallyObservable" type="tip"/>

<skdecide-signature name= "is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable.get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### \_get\_observation\_distribution <Badge text="PartiallyObservable" type="tip"/>

<skdecide-signature name= "_get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_space <Badge text="PartiallyObservable" type="tip"/>

<skdecide-signature name= "_get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable._get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### \_get\_observation\_space\_ <Badge text="PartiallyObservable" type="tip"/>

<skdecide-signature name= "_get_observation_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the observation space (finite or infinite set).

This is a helper function called by default from `PartiallyObservable._get_observation_space()`, the difference
being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The observation space.

### \_is\_observation <Badge text="PartiallyObservable" type="tip"/>

<skdecide-signature name= "_is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable._get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

## TransformedObservable

A domain must inherit this class if it is transformed observable.

"Transformed observable" means that the observation provided to the agent is deterministically computed from (but
generally not equal to) the internal state of the domain.

### get\_observation <Badge text="TransformedObservable" type="tip"/>

<skdecide-signature name= "get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable.get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable.get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### \_get\_observation <Badge text="TransformedObservable" type="tip"/>

<skdecide-signature name= "_get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable._get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### \_get\_observation\_space\_ <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the observation space (finite or infinite set).

This is a helper function called by default from `PartiallyObservable._get_observation_space()`, the difference
being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The observation space.

### \_is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable._get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

## FullyObservable

A domain must inherit this class if it is fully observable.

"Fully observable" means that the observation provided to the agent is equal to the internal state of the domain.

::: warning
In the case of fully observable domains, make sure that the observation type D.T_observation is equal to the
state type D.T_state.
:::

### get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable.get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable.get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

### \_get\_observation <Badge text="TransformedObservable" type="warn"/>

<skdecide-signature name= "_get_observation" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Get the deterministic observation given a state and action.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_distribution <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}, {'name': 'action', 'default': 'None', 'annotation': 'Optional[D.T_agent[D.T_concurrency[D.T_event]]]'}], 'return': 'Distribution[D.T_agent[D.T_observation]]'}"></skdecide-signature>

Get the probability distribution of the observation given a state and action.

In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
where $O$ is the random variable of the observation.

#### Parameters
- **state**: The state to be observed.
- **action**: The last applied action (or None if the state is an initial state).

#### Returns
The probability distribution of the observation.

### \_get\_observation\_space <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the (cached) observation space (finite or infinite set).

By default, `PartiallyObservable._get_observation_space()` internally
calls `PartiallyObservable._get_observation_space_()` the first time and automatically caches its value to make
future calls more efficient (since the observation space is assumed to be constant).

#### Returns
The observation space.

### \_get\_observation\_space\_ <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_get_observation_space_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[Space[D.T_observation]]'}"></skdecide-signature>

Get the observation space (finite or infinite set).

This is a helper function called by default from `PartiallyObservable._get_observation_space()`, the difference
being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The observation space.

### \_is\_observation <Badge text="PartiallyObservable" type="warn"/>

<skdecide-signature name= "_is_observation" :sig="{'params': [{'name': 'self'}, {'name': 'observation', 'annotation': 'D.T_agent[D.T_observation]'}], 'return': 'bool'}"></skdecide-signature>

Check that an observation indeed belongs to the domain observation space.

::: tip
By default, this function is implemented using the `skdecide.core.Space.contains()` function on the domain
observation space provided by `PartiallyObservable._get_observation_space()`, but it can be overridden for
faster implementations.
:::

#### Parameters
- **observation**: The observation to consider.

#### Returns
True if the observation belongs to the domain observation space (False otherwise).

