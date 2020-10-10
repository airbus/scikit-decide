# builders.domain.initialization

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## Initializable

A domain must inherit this class if it can be initialized.

### reset <Badge text="Initializable" type="tip"/>

<skdecide-signature name= "reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable.reset()` provides some boilerplate code and internally calls `Initializable._reset()`
(which returns an initial state). The boilerplate code automatically stores the initial state into the `_memory`
attribute and samples a corresponding observation.

#### Returns
An initial observation.

### \_reset <Badge text="Initializable" type="tip"/>

<skdecide-signature name= "_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable._reset()` provides some boilerplate code and internally
calls `Initializable._state_reset()` (which returns an initial state). The boilerplate code automatically stores
the initial state into the `_memory` attribute and samples a corresponding observation.

#### Returns
An initial observation.

### \_state\_reset <Badge text="Initializable" type="tip"/>

<skdecide-signature name= "_state_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Reset the state of the environment and return an initial state.

This is a helper function called by default from `Initializable._reset()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Returns
An initial state.

## UncertainInitialized

A domain must inherit this class if its states are initialized according to a probability distribution known as
white-box.

### get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="tip"/>

<skdecide-signature name= "get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized.get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable.reset()` provides some boilerplate code and internally calls `Initializable._reset()`
(which returns an initial state). The boilerplate code automatically stores the initial state into the `_memory`
attribute and samples a corresponding observation.

#### Returns
An initial observation.

### \_get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="tip"/>

<skdecide-signature name= "_get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized._get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### \_get\_initial\_state\_distribution\_ <Badge text="UncertainInitialized" type="tip"/>

<skdecide-signature name= "_get_initial_state_distribution_" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the probability distribution of initial states.

This is a helper function called by default from `UncertainInitialized._get_initial_state_distribution()`, the
difference being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The probability distribution of initial states.

### \_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable._reset()` provides some boilerplate code and internally
calls `Initializable._state_reset()` (which returns an initial state). The boilerplate code automatically stores
the initial state into the `_memory` attribute and samples a corresponding observation.

#### Returns
An initial observation.

### \_state\_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_state_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Reset the state of the environment and return an initial state.

This is a helper function called by default from `Initializable._reset()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Returns
An initial state.

## DeterministicInitialized

A domain must inherit this class if it has a deterministic initial state known as white-box.

### get\_initial\_state <Badge text="DeterministicInitialized" type="tip"/>

<skdecide-signature name= "get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized.get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized.get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable.reset()` provides some boilerplate code and internally calls `Initializable._reset()`
(which returns an initial state). The boilerplate code automatically stores the initial state into the `_memory`
attribute and samples a corresponding observation.

#### Returns
An initial observation.

### \_get\_initial\_state <Badge text="DeterministicInitialized" type="tip"/>

<skdecide-signature name= "_get_initial_state" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the (cached) initial state.

By default, `DeterministicInitialized._get_initial_state()` internally
calls `DeterministicInitialized._get_initial_state_()` the first time and automatically caches its value to make
future calls more efficient (since the initial state is assumed to be constant).

#### Returns
The initial state.

### \_get\_initial\_state\_ <Badge text="DeterministicInitialized" type="tip"/>

<skdecide-signature name= "_get_initial_state_" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the initial state.

This is a helper function called by default from `DeterministicInitialized._get_initial_state()`, the difference
being that the result is not cached here.

#### Returns
The initial state.

### \_get\_initial\_state\_distribution <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the (cached) probability distribution of initial states.

By default, `UncertainInitialized._get_initial_state_distribution()` internally
calls `UncertainInitialized._get_initial_state_distribution_()` the first time and automatically caches its value
to make future calls more efficient (since the initial state distribution is assumed to be constant).

#### Returns
The probability distribution of initial states.

### \_get\_initial\_state\_distribution\_ <Badge text="UncertainInitialized" type="warn"/>

<skdecide-signature name= "_get_initial_state_distribution_" :sig="{'params': [{'name': 'self'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the probability distribution of initial states.

This is a helper function called by default from `UncertainInitialized._get_initial_state_distribution()`, the
difference being that the result is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
The probability distribution of initial states.

### \_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_agent[D.T_observation]'}"></skdecide-signature>

Reset the state of the environment and return an initial observation.

By default, `Initializable._reset()` provides some boilerplate code and internally
calls `Initializable._state_reset()` (which returns an initial state). The boilerplate code automatically stores
the initial state into the `_memory` attribute and samples a corresponding observation.

#### Returns
An initial observation.

### \_state\_reset <Badge text="Initializable" type="warn"/>

<skdecide-signature name= "_state_reset" :sig="{'params': [{'name': 'self'}], 'return': 'D.T_state'}"></skdecide-signature>

Reset the state of the environment and return an initial state.

This is a helper function called by default from `Initializable._reset()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Returns
An initial state.

