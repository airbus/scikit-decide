# builders.domain.dynamics

[[toc]]

::: tip
<skdecide-summary></skdecide-summary>
:::

## Environment

A domain must inherit this class if agents interact with it like a black-box environment.

Black-box environment examples include: the real world, compiled ATARI games, etc.

::: tip
Environment domains are typically stateful: they must keep the current state or history in their memory to
compute next steps (automatically done by default in the `_memory` attribute).
:::

### step <Badge text="Environment" type="tip"/>

<skdecide-signature name= "step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment.step()` provides some boilerplate code and internally calls `Environment._step()` (which
returns a transition outcome). The boilerplate code automatically stores next state into the `_memory` attribute
and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment.step()` to call the external environment and not
use the `Environment._step()` helper function.
:::

::: warning
Before calling `Environment.step()` the first time or when the end of an episode is
reached, `Initializable.reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

### \_state\_step <Badge text="Environment" type="tip"/>

<skdecide-signature name= "_state_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one step of the transition's dynamics.

This is a helper function called by default from `Environment._step()`. It focuses on the state level, as opposed
to the observation one for the latter.

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The transition outcome of this step.

### \_step <Badge text="Environment" type="tip"/>

<skdecide-signature name= "_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment._step()` provides some boilerplate code and internally
calls `Environment._state_step()` (which returns a transition outcome). The boilerplate code automatically stores
next state into the `_memory` attribute and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment._step()` to call the external environment and not
use the `Environment._state_step()` helper function.
:::

::: warning
Before calling `Environment._step()` the first time or when the end of an episode is
reached, `Initializable._reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

## Simulation

A domain must inherit this class if agents interact with it like a simulation.

Compared to pure environment domains, simulation ones have the additional ability to sample transitions from any
given state.

::: tip
Simulation domains are typically stateless: they do not need to store the current state or history in memory
since it is usually passed as parameter of their functions. By default, they only become stateful whenever they
are used as environments (e.g. via `Initializable.reset()` and `Environment.step()` functions).
:::

### sample <Badge text="Simulation" type="tip"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation.sample()` provides some boilerplate code and internally calls `Simulation._sample()`
(which returns a transition outcome). The boilerplate code automatically samples an observation corresponding to
the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation.sample()` to call the external simulator and not use
the `Simulation._sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### set\_memory <Badge text="Simulation" type="tip"/>

<skdecide-signature name= "set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with successive `Environment.step()`
calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain.set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain.step(my_action)
```

### step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment.step()` provides some boilerplate code and internally calls `Environment._step()` (which
returns a transition outcome). The boilerplate code automatically stores next state into the `_memory` attribute
and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment.step()` to call the external environment and not
use the `Environment._step()` helper function.
:::

::: warning
Before calling `Environment.step()` the first time or when the end of an episode is
reached, `Initializable.reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

### \_sample <Badge text="Simulation" type="tip"/>

<skdecide-signature name= "_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation._sample()` provides some boilerplate code and internally
calls `Simulation._state_sample()` (which returns a transition outcome). The boilerplate code automatically
samples an observation corresponding to the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation._sample()` to call the external simulator and not use
the `Simulation._state_sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### \_set\_memory <Badge text="Simulation" type="tip"/>

<skdecide-signature name= "_set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with
successive `Environment._step()` calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain._set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain._step(my_action)
```

### \_state\_sample <Badge text="Simulation" type="tip"/>

<skdecide-signature name= "_state_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one sample of the transition's dynamics.

This is a helper function called by default from `Simulation._sample()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The transition outcome of the sampled transition.

### \_state\_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_state_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one step of the transition's dynamics.

This is a helper function called by default from `Environment._step()`. It focuses on the state level, as opposed
to the observation one for the latter.

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The transition outcome of this step.

### \_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment._step()` provides some boilerplate code and internally
calls `Environment._state_step()` (which returns a transition outcome). The boilerplate code automatically stores
next state into the `_memory` attribute and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment._step()` to call the external environment and not
use the `Environment._state_step()` helper function.
:::

::: warning
Before calling `Environment._step()` the first time or when the end of an episode is
reached, `Initializable._reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

## UncertainTransitions

A domain must inherit this class if its dynamics is uncertain and provided as a white-box model.

Compared to pure simulation domains, uncertain transition ones provide in addition the full probability distribution
of next states given a memory and action.

::: tip
Uncertain transition domains are typically stateless: they do not need to store the current state or history in
memory since it is usually passed as parameter of their functions. By default, they only become stateful
whenever they are used as environments (e.g. via `Initializable.reset()` and `Environment.step()` functions).
:::

### get\_next\_state\_distribution <Badge text="UncertainTransitions" type="tip"/>

<skdecide-signature name= "get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the probability distribution of next state given a memory and action.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The probability distribution of next state.

### get\_transition\_value <Badge text="UncertainTransitions" type="tip"/>

<skdecide-signature name= "get_transition_value" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_agent[TransitionValue[D.T_value]]'}"></skdecide-signature>

Get the value (reward or cost) of a transition.

The transition to consider is defined by the function parameters.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `UncertainTransitions._is_transition_value_dependent_on_next_state_()` to return
False. This information can then be exploited by solvers to avoid computing next state to evaluate a
transition value (more efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
The transition value (reward or cost).

### is\_terminal <Badge text="UncertainTransitions" type="tip"/>

<skdecide-signature name= "is_terminal" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether a state is terminal.

A terminal state is a state with no outgoing transition (except to itself with value 0).

#### Parameters
- **state**: The state to consider.

#### Returns
True if the state is terminal (False otherwise).

### is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="tip"/>

<skdecide-signature name= "is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions.is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation.sample()` provides some boilerplate code and internally calls `Simulation._sample()`
(which returns a transition outcome). The boilerplate code automatically samples an observation corresponding to
the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation.sample()` to call the external simulator and not use
the `Simulation._sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with successive `Environment.step()`
calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain.set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain.step(my_action)
```

### step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment.step()` provides some boilerplate code and internally calls `Environment._step()` (which
returns a transition outcome). The boilerplate code automatically stores next state into the `_memory` attribute
and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment.step()` to call the external environment and not
use the `Environment._step()` helper function.
:::

::: warning
Before calling `Environment.step()` the first time or when the end of an episode is
reached, `Initializable.reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

### \_get\_next\_state\_distribution <Badge text="UncertainTransitions" type="tip"/>

<skdecide-signature name= "_get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'Distribution[D.T_state]'}"></skdecide-signature>

Get the probability distribution of next state given a memory and action.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The probability distribution of next state.

### \_get\_transition\_value <Badge text="UncertainTransitions" type="tip"/>

<skdecide-signature name= "_get_transition_value" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_agent[TransitionValue[D.T_value]]'}"></skdecide-signature>

Get the value (reward or cost) of a transition.

The transition to consider is defined by the function parameters.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `UncertainTransitions._is_transition_value_dependent_on_next_state_()` to return
False. This information can then be exploited by solvers to avoid computing next state to evaluate a
transition value (more efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
The transition value (reward or cost).

### \_is\_terminal <Badge text="UncertainTransitions" type="tip"/>

<skdecide-signature name= "_is_terminal" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether a state is terminal.

A terminal state is a state with no outgoing transition (except to itself with value 0).

#### Parameters
- **state**: The state to consider.

#### Returns
True if the state is terminal (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="tip"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions._is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state\_ <Badge text="UncertainTransitions" type="tip"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state_" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation.

This is a helper function called by default
from `UncertainTransitions._is_transition_value_dependent_on_next_state()`, the difference being that the result
is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation._sample()` provides some boilerplate code and internally
calls `Simulation._state_sample()` (which returns a transition outcome). The boilerplate code automatically
samples an observation corresponding to the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation._sample()` to call the external simulator and not use
the `Simulation._state_sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### \_set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with
successive `Environment._step()` calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain._set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain._step(my_action)
```

### \_state\_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_state_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one sample of the transition's dynamics.

This is a helper function called by default from `Simulation._sample()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The transition outcome of the sampled transition.

### \_state\_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_state_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one step of the transition's dynamics.

This is a helper function called by default from `Environment._step()`. It focuses on the state level, as opposed
to the observation one for the latter.

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The transition outcome of this step.

### \_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment._step()` provides some boilerplate code and internally
calls `Environment._state_step()` (which returns a transition outcome). The boilerplate code automatically stores
next state into the `_memory` attribute and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment._step()` to call the external environment and not
use the `Environment._state_step()` helper function.
:::

::: warning
Before calling `Environment._step()` the first time or when the end of an episode is
reached, `Initializable._reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

## EnumerableTransitions

A domain must inherit this class if its dynamics is uncertain (with enumerable transitions) and provided as a
white-box model.

Compared to pure uncertain transition domains, enumerable transition ones guarantee that all probability
distributions of next state are discrete.

::: tip
Enumerable transition domains are typically stateless: they do not need to store the current state or history in
memory since it is usually passed as parameter of their functions. By default, they only become stateful
whenever they are used as environments (e.g. via `Initializable.reset()` and `Environment.step()` functions).
:::

### get\_next\_state\_distribution <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'DiscreteDistribution[D.T_state]'}"></skdecide-signature>

Get the discrete probability distribution of next state given a memory and action.

::: tip
In the Markovian case (memory only holds last state $s$), given an action $a$, this function can
be mathematically represented by $P(S'|s, a)$, where $S'$ is the next state random variable.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The discrete probability distribution of next state.

### get\_transition\_value <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "get_transition_value" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_agent[TransitionValue[D.T_value]]'}"></skdecide-signature>

Get the value (reward or cost) of a transition.

The transition to consider is defined by the function parameters.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `UncertainTransitions._is_transition_value_dependent_on_next_state_()` to return
False. This information can then be exploited by solvers to avoid computing next state to evaluate a
transition value (more efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
The transition value (reward or cost).

### is\_terminal <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "is_terminal" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether a state is terminal.

A terminal state is a state with no outgoing transition (except to itself with value 0).

#### Parameters
- **state**: The state to consider.

#### Returns
True if the state is terminal (False otherwise).

### is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions.is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation.sample()` provides some boilerplate code and internally calls `Simulation._sample()`
(which returns a transition outcome). The boilerplate code automatically samples an observation corresponding to
the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation.sample()` to call the external simulator and not use
the `Simulation._sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with successive `Environment.step()`
calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain.set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain.step(my_action)
```

### step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment.step()` provides some boilerplate code and internally calls `Environment._step()` (which
returns a transition outcome). The boilerplate code automatically stores next state into the `_memory` attribute
and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment.step()` to call the external environment and not
use the `Environment._step()` helper function.
:::

::: warning
Before calling `Environment.step()` the first time or when the end of an episode is
reached, `Initializable.reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

### \_get\_next\_state\_distribution <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'DiscreteDistribution[D.T_state]'}"></skdecide-signature>

Get the discrete probability distribution of next state given a memory and action.

::: tip
In the Markovian case (memory only holds last state $s$), given an action $a$, this function can
be mathematically represented by $P(S'|s, a)$, where $S'$ is the next state random variable.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The discrete probability distribution of next state.

### \_get\_transition\_value <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_get_transition_value" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_agent[TransitionValue[D.T_value]]'}"></skdecide-signature>

Get the value (reward or cost) of a transition.

The transition to consider is defined by the function parameters.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `UncertainTransitions._is_transition_value_dependent_on_next_state_()` to return
False. This information can then be exploited by solvers to avoid computing next state to evaluate a
transition value (more efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
The transition value (reward or cost).

### \_is\_terminal <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_terminal" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether a state is terminal.

A terminal state is a state with no outgoing transition (except to itself with value 0).

#### Parameters
- **state**: The state to consider.

#### Returns
True if the state is terminal (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions._is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state\_ <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state_" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation.

This is a helper function called by default
from `UncertainTransitions._is_transition_value_dependent_on_next_state()`, the difference being that the result
is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation._sample()` provides some boilerplate code and internally
calls `Simulation._state_sample()` (which returns a transition outcome). The boilerplate code automatically
samples an observation corresponding to the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation._sample()` to call the external simulator and not use
the `Simulation._state_sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### \_set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with
successive `Environment._step()` calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain._set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain._step(my_action)
```

### \_state\_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_state_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one sample of the transition's dynamics.

This is a helper function called by default from `Simulation._sample()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The transition outcome of the sampled transition.

### \_state\_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_state_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one step of the transition's dynamics.

This is a helper function called by default from `Environment._step()`. It focuses on the state level, as opposed
to the observation one for the latter.

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The transition outcome of this step.

### \_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment._step()` provides some boilerplate code and internally
calls `Environment._state_step()` (which returns a transition outcome). The boilerplate code automatically stores
next state into the `_memory` attribute and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment._step()` to call the external environment and not
use the `Environment._state_step()` helper function.
:::

::: warning
Before calling `Environment._step()` the first time or when the end of an episode is
reached, `Initializable._reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

## DeterministicTransitions

A domain must inherit this class if its dynamics is deterministic and provided as a white-box model.

Compared to pure enumerable transition domains, deterministic transition ones guarantee that there is only one next
state for a given source memory (state or history) and action.

::: tip
Deterministic transition domains are typically stateless: they do not need to store the current state or history
in memory since it is usually passed as parameter of their functions. By default, they only become stateful
whenever they are used as environments (e.g. via `Initializable.reset()` and `Environment.step()` functions).
:::

### get\_next\_state <Badge text="DeterministicTransitions" type="tip"/>

<skdecide-signature name= "get_next_state" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the next state given a memory and action.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The deterministic next state.

### get\_next\_state\_distribution <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'DiscreteDistribution[D.T_state]'}"></skdecide-signature>

Get the discrete probability distribution of next state given a memory and action.

::: tip
In the Markovian case (memory only holds last state $s$), given an action $a$, this function can
be mathematically represented by $P(S'|s, a)$, where $S'$ is the next state random variable.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The discrete probability distribution of next state.

### get\_transition\_value <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "get_transition_value" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_agent[TransitionValue[D.T_value]]'}"></skdecide-signature>

Get the value (reward or cost) of a transition.

The transition to consider is defined by the function parameters.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `UncertainTransitions._is_transition_value_dependent_on_next_state_()` to return
False. This information can then be exploited by solvers to avoid computing next state to evaluate a
transition value (more efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
The transition value (reward or cost).

### is\_terminal <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "is_terminal" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether a state is terminal.

A terminal state is a state with no outgoing transition (except to itself with value 0).

#### Parameters
- **state**: The state to consider.

#### Returns
True if the state is terminal (False otherwise).

### is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions.is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation.sample()` provides some boilerplate code and internally calls `Simulation._sample()`
(which returns a transition outcome). The boilerplate code automatically samples an observation corresponding to
the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation.sample()` to call the external simulator and not use
the `Simulation._sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with successive `Environment.step()`
calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain.set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain.step(my_action)
```

### step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment.step()` provides some boilerplate code and internally calls `Environment._step()` (which
returns a transition outcome). The boilerplate code automatically stores next state into the `_memory` attribute
and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment.step()` to call the external environment and not
use the `Environment._step()` helper function.
:::

::: warning
Before calling `Environment.step()` the first time or when the end of an episode is
reached, `Initializable.reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

### \_get\_next\_state <Badge text="DeterministicTransitions" type="tip"/>

<skdecide-signature name= "_get_next_state" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'D.T_state'}"></skdecide-signature>

Get the next state given a memory and action.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The deterministic next state.

### \_get\_next\_state\_distribution <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_get_next_state_distribution" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'SingleValueDistribution[D.T_state]'}"></skdecide-signature>

Get the discrete probability distribution of next state given a memory and action.

::: tip
In the Markovian case (memory only holds last state $s$), given an action $a$, this function can
be mathematically represented by $P(S'|s, a)$, where $S'$ is the next state random variable.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The discrete probability distribution of next state.

### \_get\_transition\_value <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_get_transition_value" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}, {'name': 'next_state', 'default': 'None', 'annotation': 'Optional[D.T_state]'}], 'return': 'D.T_agent[TransitionValue[D.T_value]]'}"></skdecide-signature>

Get the value (reward or cost) of a transition.

The transition to consider is defined by the function parameters.

::: tip
If this function never depends on the next_state parameter for its computation, it is recommended to
indicate it by overriding `UncertainTransitions._is_transition_value_dependent_on_next_state_()` to return
False. This information can then be exploited by solvers to avoid computing next state to evaluate a
transition value (more efficient).
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.
- **next_state**: The next state in which the transition ends (if needed for the computation).

#### Returns
The transition value (reward or cost).

### \_is\_terminal <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_terminal" :sig="{'params': [{'name': 'self'}, {'name': 'state', 'annotation': 'D.T_state'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether a state is terminal.

A terminal state is a state with no outgoing transition (except to itself with value 0).

#### Parameters
- **state**: The state to consider.

#### Returns
True if the state is terminal (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation (cached).

By default, `UncertainTransitions._is_transition_value_dependent_on_next_state()` internally
calls `UncertainTransitions._is_transition_value_dependent_on_next_state_()` the first time and automatically
caches its value to make future calls more efficient (since the returned value is assumed to be constant).

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_is\_transition\_value\_dependent\_on\_next\_state\_ <Badge text="UncertainTransitions" type="warn"/>

<skdecide-signature name= "_is_transition_value_dependent_on_next_state_" :sig="{'params': [{'name': 'self'}], 'return': 'bool'}"></skdecide-signature>

Indicate whether _get_transition_value() requires the next_state parameter for its computation.

This is a helper function called by default
from `UncertainTransitions._is_transition_value_dependent_on_next_state()`, the difference being that the result
is not cached here.

::: tip
The underscore at the end of this function's name is a convention to remind that its result should be
constant.
:::

#### Returns
True if the transition value computation depends on next_state (False otherwise).

### \_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Sample one transition of the simulator's dynamics.

By default, `Simulation._sample()` provides some boilerplate code and internally
calls `Simulation._state_sample()` (which returns a transition outcome). The boilerplate code automatically
samples an observation corresponding to the sampled next state.

::: tip
Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
simulator), it is recommended to overwrite `Simulation._sample()` to call the external simulator and not use
the `Simulation._state_sample()` helper function.
:::

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The environment outcome of the sampled transition.

### \_set\_memory <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_set_memory" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}], 'return': 'None'}"></skdecide-signature>

Set internal memory attribute `_memory` to given one.

This can be useful to set a specific "starting point" before doing a rollout with
successive `Environment._step()` calls.

#### Parameters
- **memory**: The memory to set internally.

#### Example
```python
# Set simulation_domain memory to my_state (assuming Markovian domain)
simulation_domain._set_memory(my_state)

# Start a 100-steps rollout from here (applying my_action at every step)
for _ in range(100):
    simulation_domain._step(my_action)
```

### \_state\_sample <Badge text="Simulation" type="warn"/>

<skdecide-signature name= "_state_sample" :sig="{'params': [{'name': 'self'}, {'name': 'memory', 'annotation': 'D.T_memory[D.T_state]'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one sample of the transition's dynamics.

This is a helper function called by default from `Simulation._sample()`. It focuses on the state level, as
opposed to the observation one for the latter.

#### Parameters
- **memory**: The source memory (state or history) of the transition.
- **action**: The action taken in the given memory (state or history) triggering the transition.

#### Returns
The transition outcome of the sampled transition.

### \_state\_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_state_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'TransitionOutcome[D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Compute one step of the transition's dynamics.

This is a helper function called by default from `Environment._step()`. It focuses on the state level, as opposed
to the observation one for the latter.

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The transition outcome of this step.

### \_step <Badge text="Environment" type="warn"/>

<skdecide-signature name= "_step" :sig="{'params': [{'name': 'self'}, {'name': 'action', 'annotation': 'D.T_agent[D.T_concurrency[D.T_event]]'}], 'return': 'EnvironmentOutcome[D.T_agent[D.T_observation], D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]'}"></skdecide-signature>

Run one step of the environment's dynamics.

By default, `Environment._step()` provides some boilerplate code and internally
calls `Environment._state_step()` (which returns a transition outcome). The boilerplate code automatically stores
next state into the `_memory` attribute and samples a corresponding observation.

::: tip
Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
ATARI games), it is recommended to overwrite `Environment._step()` to call the external environment and not
use the `Environment._state_step()` helper function.
:::

::: warning
Before calling `Environment._step()` the first time or when the end of an episode is
reached, `Initializable._reset()` must be called to reset the environment's state.
:::

#### Parameters
- **action**: The action taken in the current memory (state or history) triggering the transition.

#### Returns
The environment outcome of this step.

